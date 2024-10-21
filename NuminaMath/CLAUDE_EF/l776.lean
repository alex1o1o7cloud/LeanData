import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_average_weight_l776_77677

def original_team_size : ℕ := 7
def new_player1_weight : ℕ := 110
def new_player2_weight : ℕ := 60
def new_team_size : ℕ := original_team_size + 2
def new_average_weight : ℚ := 113

theorem original_average_weight (original_average : ℚ) :
  (original_average * original_team_size + new_player1_weight + new_player2_weight) / new_team_size = new_average_weight →
  original_average = 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_average_weight_l776_77677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_increase_when_absent_l776_77694

/-- Represents the increase in work per person when some members are absent -/
noncomputable def work_increase (W p : ℝ) : ℝ := W / (6 * p)

/-- Proves that when 1/7 of the members are absent, the work increase per person is W/(6p) -/
theorem work_increase_when_absent (W p : ℝ) (h1 : p > 0) (h2 : W > 0) :
  let total_persons := p
  let absent_persons := p / 7
  let present_persons := p - absent_persons
  let original_work_per_person := W / p
  let new_work_per_person := W / present_persons
  new_work_per_person - original_work_per_person = work_increase W p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_increase_when_absent_l776_77694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_smallest_n_sum_power_of_two_sum_1897_power_of_two_l776_77631

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  let k := (Nat.sqrt (8 * n + 1) - 1) / 2
  2 ^ (n - k * (k + 1) / 2)

-- Define the sum of the first N terms
def seq_sum (N : ℕ) : ℕ :=
  (List.range N).map seq |>.sum

-- Theorem 1: The 100th term of the sequence is 256
theorem hundredth_term : seq 100 = 256 := by
  sorry

-- Theorem 2: The smallest N > 1000 such that the sum of the first N terms is a power of 2 is 1897
theorem smallest_n_sum_power_of_two :
  ∀ N : ℕ, N > 1000 → (∃ k : ℕ, seq_sum N = 2^k) → N ≥ 1897 := by
  sorry

-- Theorem 3: The sum of the first 1897 terms is a power of 2
theorem sum_1897_power_of_two : ∃ k : ℕ, seq_sum 1897 = 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_smallest_n_sum_power_of_two_sum_1897_power_of_two_l776_77631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l776_77600

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point lies on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given ellipse, if a chord AB passes through a focus F and AF = 2, then BF = 4 -/
theorem ellipse_chord_theorem (e : Ellipse) (A B F : Point) :
  e.a = 6 ∧ e.b = 4 ∧
  onEllipse A e ∧ onEllipse B e ∧
  F.x = 2 * Real.sqrt 5 ∧ F.y = 0 ∧
  distance A F = 2 →
  distance B F = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l776_77600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l776_77691

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi

/-- The equation a(1-x^2) + 2bx + c(1+x^2) = 0 has two equal real roots -/
def has_equal_roots (t : Triangle) : Prop :=
  ∃ x : ℝ, t.a * (1 - x^2) + 2 * t.b * x + t.c * (1 + x^2) = 0 ∧
    ∀ y : ℝ, t.a * (1 - y^2) + 2 * t.b * y + t.c * (1 + y^2) = 0 → y = x

/-- The relation 3c = a + 3b holds -/
def side_relation (t : Triangle) : Prop :=
  3 * t.c = t.a + 3 * t.b

/-- The triangle is right-angled with the right angle at C -/
def is_right_angled_at_C (t : Triangle) : Prop :=
  t.C = Real.pi / 2 ∧ t.a^2 + t.b^2 = t.c^2

theorem triangle_properties (t : Triangle) 
  (h1 : has_equal_roots t) (h2 : side_relation t) : 
  is_right_angled_at_C t ∧ Real.sin t.A + Real.sin t.B = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l776_77691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l776_77660

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem function_properties :
  let S := Set.Ioo (-1 : ℝ) 1
  (∀ x y, x ∈ S → y ∈ S → f x + f y = f ((x + y) / (1 + x * y))) ∧
  (∀ x, x ∈ S → x < 0 → f x > 0) ∧
  (f (-1/2) = 1) →
  (∀ x, x ∈ S → f x = Real.log ((1 - x) / (1 + x))) ∧
  (∀ a b, a ∈ S → b ∈ S → |a| < 1 → |b| < 1 → 
    f ((a + b) / (1 + a * b)) = 1 → f ((a - b) / (1 - a * b)) = 2 → 
    f a = 3/2 ∧ f b = -1/2) ∧
  (∃ x, x ∈ S ∧ f x = -1/2 ∧ x = 2 - Real.sqrt 3) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l776_77660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_special_number_l776_77629

theorem at_most_one_special_number (p : ℕ) (h_prime : Nat.Prime p) 
  (A B : Finset ℕ) (h_partition : A ∪ B = Finset.range p ∧ A ∩ B = ∅ ∧ A.Nonempty ∧ B.Nonempty) :
  ∃! a, a ∈ Finset.range p ∧ 
    (∀ x ∈ A, ∀ y ∈ B, x + y ≠ a ∧ x + y ≠ a + p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_special_number_l776_77629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l776_77608

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + (Real.log x / Real.log 3)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- Define the domain of x
def domain : Set ℝ := Set.Icc 1 9

-- Theorem statement
theorem g_properties :
  ∃ (max_val : ℝ) (max_x : ℝ),
    (∀ x ∈ domain, g x ≤ max_val) ∧
    (g max_x = max_val) ∧
    (max_x = 3) ∧
    (max_val = 13) ∧
    (∀ x ∈ domain, g x = x → x ∈ Set.Icc 1 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l776_77608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_range_l776_77681

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c

-- Define the specific condition given in the problem
def specialCondition (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

-- Theorem statement
theorem angle_B_range (t : Triangle) 
  (h1 : validTriangle t) 
  (h2 : specialCondition t) : 
  0 < t.B ∧ t.B ≤ Real.pi/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_range_l776_77681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newer_car_travels_195_miles_l776_77699

/-- The distance traveled by the newer car given the distance of the older car and the percentage increase -/
noncomputable def newer_car_distance (older_car_distance : ℝ) (percentage_increase : ℝ) : ℝ :=
  older_car_distance * (1 + percentage_increase / 100)

/-- Theorem stating that the newer car travels 195 miles given the conditions -/
theorem newer_car_travels_195_miles (older_car_distance : ℝ) (percentage_increase : ℝ) 
  (h1 : older_car_distance = 150)
  (h2 : percentage_increase = 30) :
  newer_car_distance older_car_distance percentage_increase = 195 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_newer_car_travels_195_miles_l776_77699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l776_77609

theorem triangle_sin_A (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 6 →
  b = 4 →
  B = 2 * A →
  Real.sin A = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l776_77609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_given_area_l776_77641

/-- A figure consisting of a right triangle and two squares -/
structure Figure where
  x : ℝ
  triangle_vertical : ℝ := 4 * x
  triangle_horizontal : ℝ := 3 * x
  square1_side : ℝ := 3 * x
  square2_side : ℝ := 6 * x

/-- The total area of the figure -/
noncomputable def total_area (f : Figure) : ℝ :=
  (1/2) * f.triangle_vertical * f.triangle_horizontal +
  f.square1_side^2 + f.square2_side^2

/-- Theorem stating the value of x given the total area -/
theorem x_value_given_area (f : Figure) (h : total_area f = 700) :
  f.x = Real.sqrt (700/51) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_given_area_l776_77641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l776_77627

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -((-x)^2 + 2*(-x))

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f (-x) = -f x) → -- f is odd
  (f (2 - a^2) > f a) ↔ (-2 < a ∧ a < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l776_77627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inclination_positive_value_condition_l776_77620

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Theorem 1
theorem tangent_line_inclination (a : ℝ) :
  (deriv (f a)) 1 = 1 → a = 2 := by sorry

-- Theorem 2
theorem positive_value_condition (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ > 0) → a > 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inclination_positive_value_condition_l776_77620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l776_77683

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 + 2*x + 1

-- Define the line equation
def line (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Theorem statement
theorem min_distance_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 10 / 5 ∧
  ∀ (x y : ℝ), y = f x →
    ∀ (x' y' : ℝ), line x' y' →
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l776_77683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_leg_length_l776_77640

noncomputable def Triangle := ℝ → ℝ → ℝ → Prop

noncomputable def Triangle.IsIsosceles (t : Triangle) : Prop := sorry
noncomputable def Triangle.IsRight (t : Triangle) : Prop := sorry
noncomputable def Triangle.MedianToHypotenuse (t : Triangle) : ℝ := sorry
noncomputable def Triangle.Leg (t : Triangle) : ℝ := sorry

theorem isosceles_right_triangle_leg_length 
  (triangle : Triangle) 
  (h_isosceles : triangle.IsIsosceles)
  (h_right : triangle.IsRight)
  (h_median : triangle.MedianToHypotenuse = 12) : 
  triangle.Leg = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_leg_length_l776_77640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l776_77625

/-- The length of a train that crosses a pole -/
noncomputable def train_length (speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600) * crossing_time_s

/-- Theorem: A train with a speed of 72 km/hr that crosses a pole in 9 seconds has a length of 180 meters -/
theorem train_length_calculation :
  train_length 72 9 = 180 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Simplify the expression
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l776_77625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_roots_l776_77678

/-- The function f(x) = 3x^2 + 2ax + b -/
def f (a b x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

/-- The integral condition -/
def integral_condition (a b : ℝ) : Prop :=
  ∫ x in Set.Icc (-1) 1, |f a b x| < 2

/-- Theorem: If the integral condition is satisfied, then f(x) = 0 has distinct real roots -/
theorem distinct_roots (a b : ℝ) (h : integral_condition a b) :
  ∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_roots_l776_77678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_decreasing_iff_a_in_range_l776_77665

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * a * x^2 + 2 * a * x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := -x^2 + a * x + 2 * a

/-- Theorem stating that f(x) is increasing on (-1, 2) when a = 1 -/
theorem f_increasing_on_interval :
  ∀ x ∈ Set.Ioo (-1) 2, f_derivative 1 x > 0 := by sorry

/-- Theorem stating that f(x) is decreasing on ℝ iff a ∈ [-8, 0] -/
theorem f_decreasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, f_derivative a x ≤ 0) ↔ a ∈ Set.Icc (-8) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_decreasing_iff_a_in_range_l776_77665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l776_77619

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the left vertex A
def A : ℝ × ℝ := (-4, 0)

-- Define the right vertex B
def B : ℝ × ℝ := (4, 0)

-- Define a point M on the ellipse
noncomputable def M : ℝ × ℝ := sorry

-- Define the angle AMB
noncomputable def angle_AMB : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_area :
  E M.1 M.2 ∧ 
  angle_AMB = Real.arccos (-Real.sqrt 65 / 65) →
  abs ((A.1 - M.1) * (B.2 - M.2) - (B.1 - M.1) * (A.2 - M.2)) / 2 = 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l776_77619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_range_l776_77679

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x - a

-- State the theorem
theorem two_zeros_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) →
  a > 2 - 2 * Real.log 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_range_l776_77679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_theta_value_l776_77673

/-- The distance set between two point sets S and T -/
def distance_set (S T : Set (ℝ × ℝ)) : Set ℝ :=
  {d | ∃ P Q, P ∈ S ∧ Q ∈ T ∧ d = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)}

/-- The set S defined by the line equation -/
def S (k θ : ℝ) : Set (ℝ × ℝ) :=
  {P | P.2 = k * P.1 + Real.sqrt 5 * Real.tan θ}

/-- The set T defined by the hyperbola equation -/
def T : Set (ℝ × ℝ) :=
  {P | P.2 = Real.sqrt (4 * P.1^2 + 1)}

theorem exists_theta_value (k : ℝ) :
  ∃ θ : ℝ, θ = -π/4 ∧ distance_set (S k θ) T = Set.Ioi 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_theta_value_l776_77673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_senate_committee_seating_l776_77656

/-- The number of Democrats in the Senate committee -/
def num_democrats : ℕ := 6

/-- The number of Republicans in the Senate committee -/
def num_republicans : ℕ := 4

/-- The number of ways to arrange the committee members around a circular table
    where no two Republicans can sit next to each other -/
def valid_seating_arrangements : ℕ := 43200

/-- Theorem stating that the number of valid seating arrangements
    for the Senate committee is 43,200 -/
theorem senate_committee_seating :
  (num_democrats - 1).factorial * (Nat.choose num_democrats num_republicans) * num_republicans.factorial = valid_seating_arrangements :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_senate_committee_seating_l776_77656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_vector_sum_magnitude_l776_77662

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)
  arithmetic_seq : b * (Real.cos C) + c * (Real.cos B) = 2 * a * (Real.cos A)

theorem angle_A_measure (t : Triangle) : t.A = π / 3 := by
  sorry

theorem vector_sum_magnitude (t : Triangle) (h1 : t.a = 3 * Real.sqrt 2) (h2 : t.b + t.c = 6) :
  t.b^2 + t.c^2 + 2*t.b*t.c*(Real.cos t.A) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_vector_sum_magnitude_l776_77662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_misha_homework_probability_l776_77634

/-- The probability of exactly k successes in n independent Bernoulli trials --/
noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of exactly k successes in n trials, given m total successes in n + m trials --/
noncomputable def conditional_binomial_probability (n m k : ℕ) : ℝ :=
  (n.choose k : ℝ) * (m.choose (m - k) : ℝ) / ((n + m).choose m : ℝ)

theorem misha_homework_probability :
  let n_monday := 5
  let n_tuesday := 6
  let n_total := n_monday + n_tuesday
  let n_correct := 7
  let n_correct_monday := 3
  conditional_binomial_probability n_monday n_tuesday n_correct_monday = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_misha_homework_probability_l776_77634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_four_values_l776_77636

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the constant c
def c : ℝ := sorry

-- State the main theorem
theorem f_four_values :
  (∀ x y : ℝ, f (f x + y) = f (x - y) + 2 * f x * y) →
  (∀ x : ℝ, f x = c * x) →
  f 4 = 0 ∨ f 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_four_values_l776_77636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_inscribed_triangle_l776_77650

/-- A triangle inscribed in a semicircle with one side as the diameter -/
structure InscribedTriangle where
  r : ℝ  -- radius of the semicircle
  a : ℝ  -- length of one tangent side
  b : ℝ  -- length of the other tangent side
  h₁ : 0 < r
  h₂ : 0 < a
  h₃ : 0 < b

/-- The area of an inscribed triangle -/
noncomputable def area (t : InscribedTriangle) : ℝ := (t.a + t.b) * t.r / 2

/-- The theorem stating the minimum area of an inscribed triangle -/
theorem min_area_inscribed_triangle (t : InscribedTriangle) :
  area t ≥ t.r^2 ∧
  (area t = t.r^2 ↔ t.a = t.b ∧ t.a = Real.sqrt 2 * t.r) := by
  sorry

#check min_area_inscribed_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_inscribed_triangle_l776_77650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_square_on_EH_l776_77643

-- Define the triangles and their properties
structure Triangle (α : Type*) [LinearOrderedField α] where
  E : α × α
  F : α × α
  G : α × α
  H : α × α
  right_angle_G : (G.1 - F.1) * (G.2 - E.2) + (G.2 - F.2) * (E.1 - G.1) = 0
  right_angle_H : (H.1 - F.1) * (H.2 - E.2) + (H.2 - F.2) * (E.1 - H.1) = 0
  square_EG : (E.1 - G.1)^2 + (E.2 - G.2)^2 = 25
  square_EF : (E.1 - F.1)^2 + (E.2 - F.2)^2 = 64
  square_GF : (G.1 - F.1)^2 + (G.2 - F.2)^2 = 49

-- Theorem statement
theorem area_of_square_on_EH {α : Type*} [LinearOrderedField α] (t : Triangle α) :
  (t.E.1 - t.H.1)^2 + (t.E.2 - t.H.2)^2 = 113 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_square_on_EH_l776_77643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_calculation_l776_77624

theorem partnership_profit_calculation (mary_investment mike_investment : ℚ) 
  (h1 : mary_investment = 800)
  (h2 : mike_investment = 200)
  (h3 : mary_investment + mike_investment > 0) :
  ∃ P : ℚ, 
    let total_investment := mary_investment + mike_investment;
    let mary_ratio := mary_investment / total_investment;
    let mike_ratio := mike_investment / total_investment;
    let mary_share := P / 6 + mary_ratio * (2 * P / 3);
    let mike_share := P / 6 + mike_ratio * (2 * P / 3);
    mary_share - mike_share = 1200 ∧ P = 3000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_calculation_l776_77624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l776_77610

-- Define the function f
noncomputable def f (x : Real) : Real := Real.sin x + Real.sqrt 3 * Real.cos x

-- Define the theorem
theorem triangle_side_length 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle
  (h1 : 0 < A ∧ A < Real.pi/2) -- A is acute
  (h2 : 0 < B ∧ B < Real.pi/2) -- B is acute
  (h3 : 0 < C ∧ C < Real.pi/2) -- C is acute
  (h4 : A + B + C = Real.pi) -- Sum of angles in a triangle
  (h5 : c = Real.sqrt 6) -- Given condition
  (h6 : Real.cos B = 1/3) -- Given condition
  (h7 : f C = Real.sqrt 3) -- Given condition
  : b = 8/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l776_77610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expression_l776_77688

theorem range_of_expression (t : Real) (h : 0 ≤ t ∧ t ≤ 1) :
  ∃ (z : Real), 2/3 ≤ z ∧ z ≤ 2 ∧
  (2 * Real.sqrt (1 - t) + 2) / (Real.sqrt t + 2) = z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expression_l776_77688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l776_77642

/-- An even function that is increasing on [0, +∞) and f(-1) = 1/2 -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x ≤ f y) ∧
  (f (-1) = 1/2)

/-- The main theorem -/
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : EvenIncreasingFunction f) 
  (h : f (Real.log 3 / Real.log a) + f (Real.log (1/3) / Real.log a) ≤ 1) :
  a ≥ 3 ∨ (0 < a ∧ a ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l776_77642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l776_77649

theorem sin_minus_cos_value (α : Real) 
  (h1 : α > π/2 ∧ α < π) 
  (h2 : Real.sin (π - α) - Real.cos (π + α) = Real.sqrt 2 / 3) : 
  Real.sin α - Real.cos α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l776_77649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l776_77669

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * ω * x - Real.pi / 6)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi) = f ω x) 
  (h_smallest_period : ∀ T, T > 0 → (∀ x, f ω (x + T) = f ω x) → T ≥ Real.pi) :
  (ω = 1) ∧ 
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-Real.pi/6 + k * Real.pi) (Real.pi/3 + k * Real.pi), 
    ∀ y ∈ Set.Icc (-Real.pi/6 + k * Real.pi) (Real.pi/3 + k * Real.pi), 
    x ≤ y → f ω x ≤ f ω y) ∧
  (∀ x ∈ Set.Icc 0 (5*Real.pi/12), f ω x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 (5*Real.pi/12), f ω x = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l776_77669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l776_77668

theorem right_triangle_properties (XY YZ : ℝ) (h_right : XY > 0 ∧ YZ > 0 ∧ XY < YZ) 
  (h_XY : XY = 30) (h_YZ : YZ = 34) : 
  let XZ := Real.sqrt (YZ^2 - XY^2)
  (XZ = 16) ∧ 
  (XZ / XY = 8 / 15) ∧ 
  (XZ / YZ = 8 / 17) := by
  -- Introduce the local definition of XZ
  let XZ := Real.sqrt (YZ^2 - XY^2)
  
  -- Split the goal into three parts
  have h1 : XZ = 16 := by sorry
  have h2 : XZ / XY = 8 / 15 := by sorry
  have h3 : XZ / YZ = 8 / 17 := by sorry
  
  -- Combine the three parts to prove the main statement
  exact ⟨h1, h2, h3⟩

#check right_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l776_77668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_opposite_edge_distance_l776_77630

/-- Represents a tetrahedron with specific edge lengths -/
structure Tetrahedron where
  /-- One face is an equilateral triangle with side length 6 -/
  equilateral_face : ℝ
  /-- Length of one edge -/
  edge1 : ℝ
  /-- Length of another edge -/
  edge2 : ℝ
  /-- Length of the third edge -/
  edge3 : ℝ
  /-- The equilateral face has side length 6 -/
  eq_face_length : equilateral_face = 6
  /-- One edge has length 3 -/
  edge1_length : edge1 = 3
  /-- Another edge has length 4 -/
  edge2_length : edge2 = 4
  /-- The third edge has length 5 -/
  edge3_length : edge3 = 5

/-- The distance between the line of the 3-unit edge and its opposite edge in the tetrahedron -/
noncomputable def opposite_edge_distance (t : Tetrahedron) : ℝ :=
  Real.sqrt (3732 / 405)

/-- Theorem stating that the distance between the 3-unit edge and its opposite edge
    in the specified tetrahedron is sqrt(3732/405) -/
theorem tetrahedron_opposite_edge_distance (t : Tetrahedron) :
  opposite_edge_distance t = Real.sqrt (3732 / 405) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_opposite_edge_distance_l776_77630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_plane_angle_l776_77690

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

structure Plane3D where
  normal : Point3D
  d : ℝ

-- Define membership for Point3D in Line3D
def Point3D.mem (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk
    (l.point.x + t * l.direction.x)
    (l.point.y + t * l.direction.y)
    (l.point.z + t * l.direction.z)

instance : Membership Point3D Line3D where
  mem := Point3D.mem

-- Helper functions (definitions only, no implementation)
noncomputable def angle_with_plane (l : Line3D) (p : Plane3D) : ℝ := sorry
noncomputable def Point3D.dist (p q : Point3D) : ℝ := sorry

-- Define the problem
theorem intersection_line_plane_angle (A : Point3D) (a : Line3D) (s : Plane3D) (α : ℝ) :
  ∃ (solutions : Finset Line3D), 
    (solutions.card = 0 ∨ solutions.card = 1 ∨ solutions.card = 2) ∧
    (∀ l ∈ solutions, 
      (∃ p : Point3D, p ∈ a ∧ p ∈ l) ∧  -- l intersects a
      (A ∈ l) ∧                         -- l passes through A
      (angle_with_plane l s = α))       -- l forms angle α with s
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_plane_angle_l776_77690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_is_2_range_of_a_l776_77616

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin (2 * x) + (a - 1) * (Real.sin x + Real.cos x) + 2 * a - 8

def domain : Set ℝ := { x | -Real.pi / 2 ≤ x ∧ x ≤ 0 }

theorem range_when_a_is_2 :
  Set.Icc (-129/16) (-3) = { y | ∃ x ∈ domain, f 2 x = y } := by
  sorry

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ (a = 1 ∨ a ≥ (17 + Real.sqrt 257) / 16)) ↔
  (∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → |f a x₁ - f a x₂| ≤ a^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_is_2_range_of_a_l776_77616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_area_square_equals_area_triangle_l776_77651

structure Point where
  x : ℝ
  y : ℝ

def Square (O P Q R : Point) : Prop :=
  (O.x = 0 ∧ O.y = 0) ∧
  (Q.x = 3 ∧ Q.y = 3) ∧
  (P.x = Q.x ∧ P.y = O.y) ∧
  (R.x = O.x ∧ R.y = Q.y)

noncomputable def AreaSquare (O P Q R : Point) : ℝ :=
  (Q.x - O.x) * (Q.y - O.y)

noncomputable def AreaTriangle (P Q T : Point) : ℝ :=
  1/2 * (Q.x - P.x) * (T.y - P.y)

theorem twice_area_square_equals_area_triangle (O P Q R T : Point) :
  Square O P Q R →
  T.x = 3 ∧ T.y = 12 →
  AreaTriangle P Q T = 2 * AreaSquare O P Q R :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_area_square_equals_area_triangle_l776_77651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l776_77628

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2
  | n + 1 => sequence_a n + Real.log (1 + 1 / (n + 1))

theorem sequence_a_formula : ∀ n : ℕ, sequence_a n = Real.log (n + 1) + 2 := by
  intro n
  induction n with
  | zero => 
    simp [sequence_a]
  | succ n ih =>
    simp [sequence_a]
    sorry  -- The actual proof would go here

#check sequence_a_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l776_77628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_properties_l776_77653

-- Define the operation of taking the arithmetic mean
def arithmeticMean (a b : ℚ) : ℚ := (a + b) / 2

-- Define a set of numbers that can be obtained using the arithmetic mean operation
def obtainableSet (m n : ℤ) : Set ℚ :=
  {x | ∃ (sequence : ℕ → ℚ), 
    (sequence 0 = 0 ∨ sequence 0 = m ∨ sequence 0 = n) ∧
    (∀ i, ∃ j k, j < i ∧ k < i ∧ sequence i = arithmeticMean (sequence j) (sequence k)) ∧
    (∃ i, sequence i = x)}

-- Main theorem
theorem arithmetic_mean_properties (m n : ℤ) 
  (h_coprime : Nat.Coprime m.natAbs n.natAbs)
  (h_less : m < n) :
  ((1 : ℚ) ∈ obtainableSet m n) ∧ 
  (∀ k : ℤ, 1 ≤ k ∧ k ≤ n → (k : ℚ) ∈ obtainableSet m n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_properties_l776_77653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_optimization_l776_77676

noncomputable section

/-- Represents a circular subway system with inner and outer loops -/
structure SubwaySystem where
  loop_length : ℝ
  total_trains : ℕ
  inner_speed : ℝ
  outer_speed : ℝ

/-- Calculates the waiting time for a given loop -/
def waiting_time (num_trains : ℕ) (speed : ℝ) (loop_length : ℝ) : ℝ :=
  (loop_length / (speed * (num_trains : ℝ))) * 60

/-- Theorem stating the minimum speed for the inner loop and optimal train distribution -/
theorem subway_optimization (s : SubwaySystem)
    (h1 : s.loop_length = 35)
    (h2 : s.total_trains = 28)
    (h3 : s.inner_speed = 30)
    (h4 : s.outer_speed = 35) :
  (∃ (min_speed : ℝ), min_speed = 25 ∧
    ∀ (speed : ℝ), speed ≥ min_speed →
      waiting_time 14 speed s.loop_length ≤ 6) ∧
  (∃ (inner_trains : ℕ) (outer_trains : ℕ),
    inner_trains = 15 ∧ outer_trains = 13 ∧
    inner_trains + outer_trains = s.total_trains ∧
    |waiting_time inner_trains s.inner_speed s.loop_length -
     waiting_time outer_trains s.outer_speed s.loop_length| ≤ 0.5) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_optimization_l776_77676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_accessible_area_l776_77684

/-- The area accessible to Chuck the llama -/
noncomputable def chucks_area (shed_width shed_length leash_length : ℝ) : ℝ :=
  let full_sector_area := (3/4) * Real.pi * leash_length^2
  let quarter_sector_area := (1/4) * Real.pi * (leash_length - shed_length)^2
  full_sector_area + quarter_sector_area

/-- Theorem: Chuck's accessible area is 12.25π square meters -/
theorem chucks_accessible_area :
  chucks_area 3 4 4 = (49/4) * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_accessible_area_l776_77684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l776_77602

theorem sin_half_angle (θ : ℝ) (h1 : Real.sin θ = 3/5) (h2 : 5*Real.pi/2 < θ) (h3 : θ < 3*Real.pi) :
  Real.sin (θ/2) = -3*Real.sqrt 10/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l776_77602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l776_77697

/-- Proves that shifting the graph of y = sin(2x - π/3) to the right by π/3 units 
    results in the graph of y = -sin(2x) -/
theorem sin_shift_equivalence (x : ℝ) : 
  Real.sin (2 * (x - Real.pi / 3) - Real.pi / 3) = -Real.sin (2 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l776_77697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_catches_lucy_l776_77667

/-- The time it takes for Tom to catch up with Lucy -/
noncomputable def catchUpTime (lucy_speed tom_speed initial_distance : ℝ) : ℝ :=
  initial_distance / (tom_speed - lucy_speed) * 60

/-- Theorem stating that Tom catches up with Lucy in 60 minutes -/
theorem tom_catches_lucy :
  catchUpTime 4 6 2 = 60 := by
  -- Unfold the definition of catchUpTime
  unfold catchUpTime
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_catches_lucy_l776_77667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_twelve_l776_77644

/-- The function to be minimized -/
noncomputable def f (c : ℝ) : ℝ := (1/3) * c^2 + 8 * c - 7

/-- Theorem stating that f attains its minimum at c = -12 -/
theorem f_min_at_neg_twelve : 
  ∀ c : ℝ, f c ≥ f (-12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_twelve_l776_77644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_is_thirty_percent_l776_77637

noncomputable def net_salary : ℝ := 3300

noncomputable def discretionary_income : ℝ := net_salary / 5

noncomputable def savings_percentage : ℝ := 0.20
noncomputable def socializing_percentage : ℝ := 0.35
noncomputable def gifts_amount : ℝ := 99

noncomputable def vacation_fund_percentage : ℝ := 
  (discretionary_income - (savings_percentage * discretionary_income + 
   socializing_percentage * discretionary_income + gifts_amount)) / discretionary_income

theorem vacation_fund_is_thirty_percent :
  vacation_fund_percentage = 0.30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_is_thirty_percent_l776_77637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l776_77639

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

theorem point_not_on_graph :
  f (-1/2) ≠ -1 ∧
  f 0 = 0 ∧
  f (1/2) ≠ 1 ∧
  f (-1) = undefined ∧
  f (-2) ≠ -4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l776_77639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_product_l776_77682

theorem unique_prime_product (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ → Prime p₂ → Prime p₃ → Prime p₄ →
  p₁ ≠ p₂ → p₁ ≠ p₃ → p₁ ≠ p₄ → p₂ ≠ p₃ → p₂ ≠ p₄ → p₃ ≠ p₄ →
  let n := p₁ * p₂ * p₃ * p₄
  let divisors := (Finset.range (n + 1)).filter (· ∣ n)
  n < 2001 →
  (divisors.toList.nthLe 8 (by sorry) : ℕ) - (divisors.toList.nthLe 7 (by sorry) : ℕ) = 22 →
  n = 1995 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_product_l776_77682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balloons_is_60_l776_77611

/-- The maximum number of balloons that can be purchased under the given conditions -/
def max_balloons (regular_price : ℚ) : ℕ :=
  let total_money := 45 * regular_price
  let pair_cost := regular_price + (regular_price / 2)
  let num_pairs := (total_money / pair_cost).floor
  (2 * num_pairs).toNat

/-- Theorem stating that the maximum number of balloons that can be purchased is 60 -/
theorem max_balloons_is_60 (regular_price : ℚ) (h : regular_price > 0) :
  max_balloons regular_price = 60 := by
  sorry

#eval max_balloons 3  -- Should output 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balloons_is_60_l776_77611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l776_77698

-- Define the points
variable (A B C O D : ℝ × ℝ)

-- Define the vectors
def OA (A O : ℝ × ℝ) : ℝ × ℝ := (A.1 - O.1, A.2 - O.2)
def OB (B O : ℝ × ℝ) : ℝ × ℝ := (B.1 - O.1, B.2 - O.2)
def OC (C O : ℝ × ℝ) : ℝ × ℝ := (C.1 - O.1, C.2 - O.2)
def DB (D B : ℝ × ℝ) : ℝ × ℝ := (B.1 - D.1, B.2 - D.2)
def DC (D C : ℝ × ℝ) : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define vector addition and scalar multiplication
def add_vec (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def smul_vec (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Define the conditions
axiom inside_triangle : O ≠ A ∧ O ≠ B ∧ O ≠ C
axiom vector_sum : add_vec (add_vec (OA A O) (smul_vec 2 (OB B O))) (smul_vec 3 (OC C O)) = (0, 0)
axiom line_intersection : ∃ t : ℝ, D = (smul_vec (1 - t) A) + (smul_vec t O) ∧ 0 < t ∧ t < 1

-- State the theorem
theorem vector_equality : add_vec (smul_vec 2 (DB D B)) (smul_vec 3 (DC D C)) = (0, 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l776_77698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_product_is_half_l776_77605

def interval : Set ℝ := Set.Icc (-15) 15

def probability_positive_product (a b : ℝ) : Prop :=
  a ∈ interval ∧ b ∈ interval ∧ a * b > 0

theorem probability_positive_product_is_half :
  ∃ (P : Set (ℝ × ℝ) → ℝ),
    (∀ s, P s ≥ 0) ∧
    (P (Set.prod interval interval) = 1) ∧
    (∀ s t, Disjoint s t → P (s ∪ t) = P s + P t) ∧
    P {(a, b) | probability_positive_product a b} = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_product_is_half_l776_77605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l776_77670

/-- Given an ellipse with the equation x²/a² + y²/b² = 1, where a > b > 0,
    left focus F(-c, 0), vertices A(-a, 0) and B(0, b),
    and the distance from F to AB is b/√7,
    prove that the eccentricity of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let F : ℝ × ℝ := (-c, 0)
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (0, b)
  let d := b / Real.sqrt 7
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
  (∃ k : ℝ, k * (B.1 - A.1) = F.1 - A.1 ∧ k * (B.2 - A.2) = F.2 - A.2 ∧
             d = |k| * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
  c / a = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l776_77670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nadine_has_twenty_white_pebbles_l776_77613

/-- The number of white pebbles Nadine has -/
def white_pebbles : ℕ := sorry

/-- The number of red pebbles Nadine has -/
def red_pebbles : ℕ := sorry

/-- The total number of pebbles Nadine has -/
def total_pebbles : ℕ := 30

/-- The number of red pebbles is half the number of white pebbles -/
axiom red_half_of_white : red_pebbles = white_pebbles / 2

/-- The total number of pebbles is the sum of white and red pebbles -/
axiom total_is_sum : total_pebbles = white_pebbles + red_pebbles

/-- Theorem: Nadine has 20 white pebbles -/
theorem nadine_has_twenty_white_pebbles : white_pebbles = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nadine_has_twenty_white_pebbles_l776_77613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_nondegenerate_ellipse_l776_77617

/-- The equation of the conic section --/
noncomputable def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

/-- The distance between the foci --/
noncomputable def foci_distance : ℝ := Real.sqrt 72

/-- Theorem stating that the conic section is a non-degenerate ellipse --/
theorem conic_is_nondegenerate_ellipse :
  (∃ x y, conic_equation x y) →
  12 > foci_distance →
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    ∀ x y, conic_equation x y ↔ 
      (x^2 / a^2) + (y^2 / b^2) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_nondegenerate_ellipse_l776_77617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_in_interval_l776_77646

def is_valid_sequence (k : ℕ → ℕ) : Prop :=
  ∀ n, k n < k (n + 1) ∧ k (n + 1) - k n ≥ 2

def S (k : ℕ → ℕ) (m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (fun i => k (i + 1))

theorem perfect_square_in_interval (k : ℕ → ℕ) (h : is_valid_sequence k) :
  ∀ n : ℕ, ∃ m : ℕ, S k n ≤ m^2 ∧ m^2 < S k (n + 1) :=
by
  sorry

#check perfect_square_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_in_interval_l776_77646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l776_77621

theorem sin_double_angle_special_case (α : ℝ) (m : ℝ) :
  (∃ x y : ℝ, x = m ∧ y = Real.sqrt 3 * m ∧ x^2 + y^2 = 1) →
  Real.sin (2 * α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l776_77621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l776_77658

def sequence_a (n : ℕ) : ℚ := 3^(n-1) - 2^n

theorem sequence_a_properties :
  let S : ℕ → ℚ := λ n => (1/2) * sequence_a (n+1) - 2^n + 3/2
  ∀ n : ℕ, n > 0 → 
    (S n - S (n-1) = sequence_a n) ∧
    (n = 1 → sequence_a n = -1) ∧
    (n = 2 → sequence_a n = -1) ∧
    (n = 3 → sequence_a n = 1) ∧
    (∃ d : ℚ × ℚ, (sequence_a 1, sequence_a 2 + 1, sequence_a 3) = (d.1, d.1 + d.2, d.1 + 2*d.2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l776_77658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_prism_volume_l776_77638

/-- A prism with a trapezoidal base -/
structure TrapezoidalPrism where
  /-- Area of one parallel lateral face -/
  S₁ : ℝ
  /-- Area of the other parallel lateral face -/
  S₂ : ℝ
  /-- Distance between the parallel lateral faces -/
  h : ℝ
  /-- The base is a trapezoid -/
  is_trapezoid : Bool

/-- The volume of a trapezoidal prism -/
noncomputable def volume (p : TrapezoidalPrism) : ℝ := (p.S₁ + p.S₂) * p.h / 2

/-- Theorem stating that the volume of a trapezoidal prism is (S₁ + S₂) * h / 2 -/
theorem trapezoidal_prism_volume (p : TrapezoidalPrism) :
  p.is_trapezoid → volume p = (p.S₁ + p.S₂) * p.h / 2 := by
  intro h
  unfold volume
  rfl

#check trapezoidal_prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_prism_volume_l776_77638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_minus_y_l776_77626

theorem nearest_integer_to_x_minus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 2) (h2 : |x| * y - x^3 = 26) :
  ∃ (n : ℤ), n = -2 ∧ ∀ (m : ℤ), |↑m - (x - y)| ≥ |↑n - (x - y)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_minus_y_l776_77626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_x_power_ten_minus_x_squared_l776_77623

/-- The number of factors when x^10 - x^2 is completely factored into polynomials and monomials with integral coefficients -/
theorem factors_of_x_power_ten_minus_x_squared :
  ∃ (f₁ f₂ f₃ f₄ f₅ f₆ : Polynomial ℤ),
    (X^10 - X^2 : Polynomial ℤ) = f₁ * f₂ * f₃ * f₄ * f₅ * f₆ ∧
    (∀ i, i ∈ ({f₁, f₂, f₃, f₄, f₅, f₆} : Set (Polynomial ℤ)) → ∃ (n : ℕ), i.degree = n) ∧
    (∀ g₁ g₂ g₃ g₄ g₅ g₆ g₇ : Polynomial ℤ,
      (X^10 - X^2 : Polynomial ℤ) ≠ g₁ * g₂ * g₃ * g₄ * g₅ * g₆ * g₇) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_x_power_ten_minus_x_squared_l776_77623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_problem_l776_77633

theorem cyclist_speed_problem (distance_XY speed_difference meeting_distance : ℝ) :
  distance_XY = 80 →
  speed_difference = 6 →
  meeting_distance = 15 →
  ∃ (speed_C : ℝ),
    speed_C > 0 ∧
    let speed_D := speed_C + speed_difference
    let time := (distance_XY - meeting_distance) / speed_C
    time * speed_D = distance_XY + meeting_distance ∧
    speed_C = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_problem_l776_77633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l776_77622

open Real

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (|x - 1| + |x + 1| - m)

-- Define the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, f 2 x ≥ 0) ∧
  (2 / (3 * a + b) + 1 / (a + 2 * b) = 2) →
  (∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) → m ≤ 2) ∧
  (7 * a + 4 * b ≥ 9 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l776_77622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_algebraic_expression_l776_77686

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- Theorem for set operations
theorem set_operations :
  (A ∩ B = {x | 1 < x ∧ x < 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) := by sorry

-- Theorem for the algebraic expression
theorem algebraic_expression (x : ℝ) (h : x > 0) :
  (2 * x^(1/4 : ℝ) + 3^(3/2 : ℝ)) * (2 * x^(1/4 : ℝ) - 3^(3/2 : ℝ)) - 4 * x^(-(1/2 : ℝ)) * (x - x^(1/2 : ℝ)) = -23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_algebraic_expression_l776_77686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_train_length_l776_77666

/-- The length of the shorter train given the speeds of two trains, the length of the longer train, and the time they take to clear each other. -/
theorem shorter_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (longer_train_length : ℝ) 
  (clear_time : ℝ) 
  (h1 : speed1 = 42) 
  (h2 : speed2 = 30) 
  (h3 : longer_train_length = 280) 
  (h4 : clear_time = 16.998640108791296) : 
  ∃ shorter_train_length : ℝ, 
    abs (shorter_train_length - 59.9728021758259) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_train_length_l776_77666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l776_77603

theorem division_problem (x y : ℕ) (h1 : x % y = 5) (h2 : (x : ℝ) / (y : ℝ) = 96.2) : y = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l776_77603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l776_77604

theorem weight_loss_challenge (W : ℝ) (hw : W > 0) : 
  let weight_after_loss := W * (1 - 0.15)
  let weight_with_clothes := weight_after_loss * (1 + 0.02)
  let measured_weight_loss_percentage := (W - weight_with_clothes) / W * 100
  abs (measured_weight_loss_percentage - 13.3) < 0.01 := by
  sorry

#check weight_loss_challenge

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l776_77604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_attendance_percentage_l776_77696

/-- Calculate the percentage of workers present, rounded to the nearest tenth -/
def workerPercentage (total workers : ℕ) (present : ℕ) : ℚ :=
  ((present : ℚ) / (total : ℚ) * 100).floor / 10

theorem worker_attendance_percentage :
  workerPercentage 86 72 = 837/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_attendance_percentage_l776_77696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_981_l776_77648

/-- A sequence formed by all powers of 3 and sums of distinct powers of 3, arranged in ascending order -/
def powerOf3Sequence : ℕ → ℕ := sorry

/-- The nth term of the powerOf3Sequence -/
def nthTerm (n : ℕ) : ℕ := powerOf3Sequence n

/-- The sequence starts with 1, 3, 4, 9, 10, 12, 13, ... -/
axiom sequence_start :
  powerOf3Sequence 0 = 1 ∧
  powerOf3Sequence 1 = 3 ∧
  powerOf3Sequence 2 = 4 ∧
  powerOf3Sequence 3 = 9 ∧
  powerOf3Sequence 4 = 10 ∧
  powerOf3Sequence 5 = 12 ∧
  powerOf3Sequence 6 = 13

/-- The sequence is arranged in ascending order -/
axiom sequence_ascending :
  ∀ n m : ℕ, n < m → powerOf3Sequence n < powerOf3Sequence m

/-- Theorem: The 100th term of the sequence is 981 -/
theorem hundredth_term_is_981 : nthTerm 99 = 981 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_981_l776_77648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_goal_is_150_l776_77692

/-- The number of cans collected on a given day -/
def cans_collected (day : ℕ) : ℕ := 20 + 5 * (day - 1)

/-- The total number of cans collected in a week -/
def total_cans : ℕ := (List.range 5).map (λ i => cans_collected (i + 1)) |>.sum

theorem weekly_goal_is_150 : total_cans = 150 := by
  -- Expand the definition of total_cans
  unfold total_cans
  -- Expand the definition of cans_collected
  simp [cans_collected]
  -- Evaluate the sum
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_goal_is_150_l776_77692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_real_part_fourth_power_l776_77675

def z1 : ℂ := -3
def z2 : ℂ := -2 + Complex.I
def z3 : ℂ := -1 + 2 * Complex.I
def z4 : ℂ := -1 + 3 * Complex.I
def z5 : ℂ := 3 * Complex.I

theorem greatest_real_part_fourth_power :
  max (Complex.re (z1^4))
    (max (Complex.re (z2^4))
      (max (Complex.re (z3^4))
        (max (Complex.re (z4^4))
          (Complex.re (z5^4))))) = Complex.re (z1^4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_real_part_fourth_power_l776_77675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_difference_is_fifty_percent_l776_77689

/-- Represents the percentage of students who answered "Yes" at the beginning and end of the year -/
structure YesPercentages where
  initial : ℝ
  final : ℝ

/-- Calculates the difference between the maximum and minimum percentage of students who could have changed their answers -/
noncomputable def change_difference (p : YesPercentages) : ℝ :=
  let min_change := |p.final - p.initial|
  let max_change := min p.initial (1 - p.final) + min (1 - p.initial) p.final
  max_change - min_change

/-- Theorem stating that for the given percentages, the difference between max and min change is 50% -/
theorem change_difference_is_fifty_percent : 
  let p : YesPercentages := { initial := 0.5, final := 0.7 }
  change_difference p = 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_difference_is_fifty_percent_l776_77689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l776_77659

/-- A dilation in the complex plane -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (2 - 3*I) 3 (1 - 2*I) = -1 := by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.I, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l776_77659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_for_given_volume_and_height_l776_77601

/-- The volume of a cone given its radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Theorem stating that a cone with given height and volume has a specific radius -/
theorem cone_radius_for_given_volume_and_height :
  ∃ (r : ℝ), 
    r > 0 ∧
    cone_volume r 21 = 2199.114857512855 ∧
    r = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_for_given_volume_and_height_l776_77601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l776_77695

noncomputable def a : ℝ × ℝ := (4, 0)
noncomputable def b : ℝ × ℝ := (-1, Real.sqrt 3)

theorem angle_between_vectors : Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l776_77695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l776_77664

theorem triangle_side_count : 
  ∃! n : ℕ, n = (Finset.filter (λ x : ℕ => x > 0 ∧ x + 3 > 10 ∧ x + 10 > 3 ∧ 3 + 10 > x) (Finset.range 100)).card ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l776_77664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l776_77661

def U : Set Int := {-2, -1, 0, 3, 6}

def A : Set Int := {x : Int | x^2 - 5*x - 6 = 0}

def B : Set Int := {x : Int | x^2 - x - 6 = 0}

theorem set_operations :
  (A ∪ B = {-2, -1, 3, 6}) ∧
  (A ∩ B = ∅) ∧
  ((U \ A) ∩ (U \ B) = {0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l776_77661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_in_terms_of_area_l776_77618

-- Define the area of the ring between two concentric circles
noncomputable def ring_area (r R : ℝ) : ℝ := Real.pi * (R^2 - r^2)

-- Define the length of the longest chord within the ring
noncomputable def longest_chord (r R : ℝ) : ℝ := 2 * Real.sqrt (R^2 - r^2)

-- Theorem statement
theorem longest_chord_in_terms_of_area {r R A : ℝ} (h1 : R > r) (h2 : A = ring_area r R) :
  longest_chord r R = 2 * Real.sqrt (A / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_in_terms_of_area_l776_77618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subspace_bases_existence_l776_77672

variable (n : ℕ)

variable (A B C : Subspace ℝ (Fin (2 * n) → ℝ))

-- A, B, and C are n-dimensional
variable (hA : FiniteDimensional.finrank ℝ A = n)
variable (hB : FiniteDimensional.finrank ℝ B = n)
variable (hC : FiniteDimensional.finrank ℝ C = n)

-- Intersection conditions
variable (hAB : A ⊓ B = ⊥)
variable (hBC : B ⊓ C = ⊥)
variable (hCA : C ⊓ A = ⊥)

theorem subspace_bases_existence :
  ∃ (basisA : Basis (Fin n) ℝ A)
    (basisB : Basis (Fin n) ℝ B)
    (basisC : Basis (Fin n) ℝ C),
  ∀ i : Fin n, ∃ (a : A) (b : B) (c : C),
    basisA i = a ∧ basisB i = b ∧ basisC i = c ∧ c.1 = a.1 + b.1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subspace_bases_existence_l776_77672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julian_numbers_l776_77647

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def is_consecutive (a b : Nat) : Prop := b = a + 1

theorem julian_numbers :
  ∀ (pedro_nums ana_nums : Fin 3 → Nat),
  (∀ i : Fin 3, pedro_nums i ∈ Finset.range 10) →
  (∀ i : Fin 3, ana_nums i ∈ Finset.range 10) →
  (∀ i : Fin 2, is_consecutive (pedro_nums i) (pedro_nums (Fin.succ i))) →
  (pedro_nums 0 * pedro_nums 1 * pedro_nums 2 = 5 * (pedro_nums 0 + pedro_nums 1 + pedro_nums 2)) →
  (∃ i j : Fin 3, i ≠ j ∧ is_consecutive (ana_nums i) (ana_nums j)) →
  (∀ i : Fin 3, ¬is_prime (ana_nums i)) →
  (ana_nums 0 * ana_nums 1 * ana_nums 2 = 4 * (ana_nums 0 + ana_nums 1 + ana_nums 2)) →
  ∃ (julian_nums : Fin 3 → Nat),
    (∀ i : Fin 3, julian_nums i ∈ Finset.range 10) ∧
    (∀ n : Nat, n ∈ Finset.range 10 → 
      (n ∈ Finset.image julian_nums Finset.univ ↔ n ∉ Finset.image pedro_nums Finset.univ ∧ n ∉ Finset.image ana_nums Finset.univ)) ∧
    Finset.image julian_nums Finset.univ = {2, 6, 7} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julian_numbers_l776_77647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l776_77614

/-- The arc length of the curve given by ρ = 2φ for 0 ≤ φ ≤ 12/5 -/
noncomputable def arcLength : ℝ := 156/25 + Real.log 5

/-- The polar equation of the curve -/
def ρ (φ : ℝ) : ℝ := 2 * φ

theorem arc_length_polar_curve :
  ∫ φ in (0)..(12/5), Real.sqrt ((ρ φ)^2 + (deriv ρ φ)^2) = arcLength := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l776_77614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_form_l776_77655

-- Define the polynomial type
def MyPolynomial (R : Type*) [Semiring R] := R → R

-- Define the condition for the polynomial
def SatisfiesCondition (P : MyPolynomial ℝ) : Prop :=
  ∀ x : ℝ, x * P (x - 1) = (x - 2) * P x

-- State the theorem
theorem polynomial_form (P : MyPolynomial ℝ) (h : SatisfiesCondition P) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x * (x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_form_l776_77655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shiny_igneous_ratio_l776_77606

/-- Represents Cliff's rock collection --/
structure RockCollection where
  total : ℕ
  sedimentary : ℕ
  igneous : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- Conditions for Cliff's rock collection --/
def cliffsCollection : RockCollection where
  total := 180
  sedimentary := 120
  igneous := 60
  shinyIgneous := 40
  shinySedimentary := 24

/-- Theorem stating the ratio of shiny igneous rocks to total igneous rocks --/
theorem shiny_igneous_ratio (c : RockCollection) 
  (h1 : c.igneous = c.sedimentary / 2)
  (h2 : c.shinySedimentary = c.sedimentary / 5)
  (h3 : c.shinyIgneous = 40)
  (h4 : c.total = 180)
  (h5 : c.total = c.sedimentary + c.igneous) :
  c.shinyIgneous * 3 = c.igneous * 2 := by
  sorry

/-- Check the theorem with cliffsCollection --/
example : cliffsCollection.shinyIgneous * 3 = cliffsCollection.igneous * 2 := by
  apply shiny_igneous_ratio cliffsCollection
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shiny_igneous_ratio_l776_77606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_angles_acute_in_triangle_l776_77607

noncomputable def angle (P Q R : ℝ × ℝ) : ℝ := sorry

theorem two_angles_acute_in_triangle (A B C : ℝ × ℝ) : 
  ∃ (x y : ℝ), (x = angle A B C ∧ y = angle B C A ∧ 0 < x ∧ x < π / 2 ∧ 0 < y ∧ y < π / 2) ∨
                (x = angle B C A ∧ y = angle C A B ∧ 0 < x ∧ x < π / 2 ∧ 0 < y ∧ y < π / 2) ∨
                (x = angle C A B ∧ y = angle A B C ∧ 0 < x ∧ x < π / 2 ∧ 0 < y ∧ y < π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_angles_acute_in_triangle_l776_77607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_through_point_standard_form_and_directrix_l776_77657

-- Define a parabola type
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  axis : Bool -- true for x-axis, false for y-axis

-- Define the condition that the parabola passes through a point
noncomputable def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  if p.axis then
    y^2 = p.a * x + p.b * y + p.c
  else
    x^2 = p.a * x + p.b * y + p.c

-- Define the vertex of the parabola
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  if p.axis then
    (-p.b / (2 * p.a), -p.c + p.b^2 / (4 * p.a))
  else
    (-p.c + p.a^2 / (4 * p.b), -p.a / (2 * p.b))

-- Define the directrix of the parabola
noncomputable def directrix (p : Parabola) : ℝ :=
  if p.axis then
    -p.c / p.a - 1 / (4 * p.a)
  else
    -p.c / p.b - 1 / (4 * p.b)

-- Theorem statement
theorem parabola_through_point_standard_form_and_directrix :
  ∀ p : Parabola,
    vertex p = (0, 0) →
    passes_through p 1 (-2) →
    ((p.axis ∧ p.a = 4 ∧ p.b = 0 ∧ p.c = 0) ∨
     (¬p.axis ∧ p.a = 0 ∧ p.b = -1/2 ∧ p.c = 0)) ∧
    (directrix p = -1 ∨ directrix p = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_through_point_standard_form_and_directrix_l776_77657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l776_77652

/-- The distance from the origin to a line ax + by + c = 0 is |c| / √(a² + b²) -/
noncomputable def distanceFromOriginToLine (a b c : ℝ) : ℝ :=
  |c| / Real.sqrt (a^2 + b^2)

/-- The line equation coefficients -/
noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 3
def c : ℝ := -2

theorem distance_to_line : distanceFromOriginToLine a b c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l776_77652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l776_77632

/-- The time it takes for A to complete the work alone -/
noncomputable def time_A : ℝ := 3

/-- The time it takes for A and B to complete the work together -/
noncomputable def time_AB : ℝ := 2

/-- The time it takes for B to complete the work alone -/
noncomputable def time_B : ℝ := 6

/-- The work rate of A -/
noncomputable def rate_A : ℝ := 1 / time_A

/-- The work rate of B -/
noncomputable def rate_B : ℝ := 1 / time_B

/-- The combined work rate of A and B -/
noncomputable def rate_AB : ℝ := 1 / time_AB

/-- Theorem stating that the sum of individual rates equals the combined rate -/
theorem work_completion_time :
  rate_A + rate_B = rate_AB := by
  -- Expand the definitions
  unfold rate_A rate_B rate_AB
  unfold time_A time_AB time_B
  -- Simplify the equation
  simp [add_div]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l776_77632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_distributions_properties_l776_77680

noncomputable def P_X (a : ℝ) : ℝ → ℝ := fun x =>
  if x = -1 then 1/3
  else if x = 0 then (2-a)/3
  else if x = 1 then a/3
  else 0

noncomputable def P_Y (a : ℝ) : ℝ → ℝ := fun y =>
  if y = 0 then 1/2
  else if y = 1 then (1-a)/2
  else if y = 2 then a/2
  else 0

theorem probability_distributions_properties :
  (∀ a, (∀ x, P_X a x ≥ 0 ∧ P_X a x ≤ 1) ∧ (∀ y, P_Y a y ≥ 0 ∧ P_Y a y ≤ 1) ↔ 0 ≤ a ∧ a ≤ 1) ∧
  (let a := (1/2 : ℝ)
   let E_Y := 0 * P_Y a 0 + 1 * P_Y a 1 + 2 * P_Y a 2
   let D_Y := (0 - E_Y)^2 * P_Y a 0 + (1 - E_Y)^2 * P_Y a 1 + (2 - E_Y)^2 * P_Y a 2
   D_Y = 11/16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_distributions_properties_l776_77680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_skew_l776_77654

structure Plane where

structure Line where

def intersect (p1 p2 : Plane) (l : Line) : Prop :=
  sorry

def parallel (l : Line) (p : Plane) : Prop :=
  sorry

def intersect_line_plane (l : Line) (p : Plane) : Prop :=
  sorry

def skew (l1 l2 : Line) : Prop :=
  sorry

theorem line_plane_intersection_skew 
  (α β : Plane) (a c : Line) 
  (h1 : intersect α β c) 
  (h2 : parallel a α) 
  (h3 : intersect_line_plane a β) : 
  skew a c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_skew_l776_77654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_probability_l776_77671

-- Define the parabola
def f (x : ℝ) := x^2

-- Define the derivative of the parabola
def f' (x : ℝ) := 2 * x

-- Define the interval
def interval : Set ℝ := Set.Icc (-6) 6

-- Define the condition for the angle of inclination
def angle_condition (x : ℝ) : Prop :=
  Real.pi/4 ≤ Real.arctan (f' x) ∧ Real.arctan (f' x) ≤ 3*Real.pi/4

-- Define the set of x values satisfying the angle condition
def satisfying_set : Set ℝ := {x ∈ interval | angle_condition x}

-- State the theorem
theorem tangent_angle_probability :
  MeasureTheory.volume satisfying_set / MeasureTheory.volume interval = 11/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_probability_l776_77671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_expenses_theorem_l776_77674

-- Define the costs and frequencies
def vet_cost : ℚ := 400
def vet_visits : ℚ := 3
def medication_cost : ℚ := 50
def food_cost : ℚ := 30
def toy_cost : ℚ := 15
def grooming_cost : ℚ := 60
def grooming_frequency : ℚ := 4  -- 4 times a year
def irregular_health_cost : ℚ := 200
def irregular_health_frequency : ℚ := 2
def insurance_cost : ℚ := 100

-- Define insurance coverage percentages
def vet_coverage : ℚ := 80 / 100
def medication_coverage : ℚ := 50 / 100
def irregular_health_coverage : ℚ := 25 / 100

-- Define the total cost calculation
noncomputable def total_cost : ℚ := 
  vet_cost +  -- First vet visit (not covered)
  (vet_cost * (vet_visits - 1) * (1 - vet_coverage)) +  -- Subsequent vet visits
  (medication_cost * 12 * (1 - medication_coverage)) +  -- Medication for the year
  food_cost * 12 +  -- Food for the year
  toy_cost * 12 +  -- Toys for the year
  grooming_cost * grooming_frequency +  -- Grooming for the year
  (irregular_health_cost * irregular_health_frequency * (1 - irregular_health_coverage)) +  -- Irregular health issues
  insurance_cost  -- Insurance cost

-- Theorem statement
theorem dog_expenses_theorem : total_cost = 2040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_expenses_theorem_l776_77674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l776_77645

theorem problem_statement (a b : ℕ) (h_pos : 0 < b ∧ b < a) 
  (h_coprime : Nat.Coprime a b) (h_eq : (a^3 - b^3) / (a - b)^3 = 13) : 
  a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l776_77645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l776_77615

/-- Function to calculate the area of a quadrilateral given its side lengths -/
noncomputable def area_of_quadrilateral (a b c d : ℝ) : ℝ := sorry

/-- Predicate to check if a quadrilateral with given side lengths is inscribed in a circle -/
def is_inscribed_quadrilateral (a b c d : ℝ) : Prop := sorry

/-- Predicate to check if a quadrilateral with given side lengths has perpendicular diagonals -/
def has_perpendicular_diagonals (a b c d : ℝ) : Prop := sorry

/-- A quadrilateral with side lengths a, b, c, d and area S satisfies S ≤ (ac + bd) / 2,
    with equality if and only if it is inscribed in a circle with perpendicular diagonals -/
theorem quadrilateral_area_inequality (a b c d S : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (hS : S = area_of_quadrilateral a b c d) :
  S ≤ (a * c + b * d) / 2 ∧ 
  (S = (a * c + b * d) / 2 ↔ is_inscribed_quadrilateral a b c d ∧ has_perpendicular_diagonals a b c d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l776_77615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_graveling_roads_l776_77635

/-- Cost of graveling roads on a rectangular lawn -/
theorem cost_of_graveling_roads 
  (lawn_length lawn_width road_width cost_per_sqm : ℝ) 
  (h1 : lawn_length = 110)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : cost_per_sqm = 3) : 
  (lawn_length * road_width + (lawn_width - road_width) * road_width) * cost_per_sqm = 4800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_graveling_roads_l776_77635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l776_77612

theorem tan_difference (α β : ℝ) (h1 : Real.tan (α + Real.pi/4) = 3) (h2 : Real.tan β = 2) : 
  Real.tan (α - β) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l776_77612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_waddle_difference_l776_77687

/-- The number of telephone poles -/
def num_poles : ℕ := 51

/-- The total distance between the first and last pole in feet -/
def total_distance : ℝ := 6336

/-- The number of waddles Penelope takes between consecutive poles -/
def penelope_waddles : ℕ := 50

/-- The number of jumps Hector takes between consecutive poles -/
def hector_jumps : ℕ := 15

/-- Penelope's waddle length in feet -/
noncomputable def penelope_waddle_length : ℝ := 
  total_distance / (penelope_waddles * (num_poles - 1 : ℝ))

/-- Hector's jump length in feet -/
noncomputable def hector_jump_length : ℝ := 
  total_distance / (hector_jumps * (num_poles - 1 : ℝ))

/-- The theorem stating the difference between Hector's jump and Penelope's waddle -/
theorem jump_waddle_difference : 
  ∃ ε > 0, abs (hector_jump_length - penelope_waddle_length - 5.9136) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_waddle_difference_l776_77687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l776_77663

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.cos (x + Real.pi/6))

theorem f_range :
  let S := {y | ∃ x ∈ Set.Icc (Real.pi/12) (Real.pi/6), f x = y}
  S = Set.Icc ((Real.sqrt 3 - 1) / 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l776_77663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_K_l776_77685

/-- Two concentric circles with radii r and R, centered at the origin -/
def concentric_circles (r R : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = r^2 ∨ p.1^2 + p.2^2 = R^2}

/-- Point B on the circle with radius r -/
def point_B (r : ℝ) : ℝ × ℝ := (r, 0)

/-- Point C on the circle with radius R -/
def point_C (R : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = R^2

/-- Two equal circles mutually tangent at point K -/
def tangent_circles (K B C : ℝ × ℝ) : Prop :=
  ∃ (n : ℝ), n > 0 ∧
    (K.1 - B.1)^2 + (K.2 - B.2)^2 = n^2 ∧
    (K.1 - C.1)^2 + (K.2 - C.2)^2 = n^2

/-- Main theorem: Locus of points K -/
theorem locus_of_K (r R : ℝ) (hr : r > 0) (hR : R > 0) :
  {K : ℝ × ℝ | ∃ (C : ℝ × ℝ), point_C R C ∧
    tangent_circles K (point_B r) C} =
  {K : ℝ × ℝ | 2 * K.1^2 + 2 * K.2^2 = r^2 + R^2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_K_l776_77685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l776_77693

theorem revenue_change 
  (original_price original_visitors : ℝ) 
  (price_increase : ℝ) 
  (visitor_decrease : ℝ) 
  (h1 : price_increase = 0.5) 
  (h2 : visitor_decrease = 0.2) : 
  let new_price := original_price * (1 + price_increase)
  let new_visitors := original_visitors * (1 - visitor_decrease)
  let original_revenue := original_price * original_visitors
  let new_revenue := new_price * new_visitors
  (new_revenue - original_revenue) / original_revenue = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l776_77693
