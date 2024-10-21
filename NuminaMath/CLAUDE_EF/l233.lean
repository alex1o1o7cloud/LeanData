import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_pow_x_eq_x_pow_y_l233_23309

-- Define the variables and conditions
variable (t : ℝ)
variable (h : t > 2)

noncomputable def x (t : ℝ) : ℝ := t^(2 / (t - 2))
noncomputable def y (t : ℝ) : ℝ := t^(t / (t - 2))

-- State the theorem
theorem y_pow_x_eq_x_pow_y (t : ℝ) (h : t > 2) : (y t)^(x t) = (x t)^(y t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_pow_x_eq_x_pow_y_l233_23309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l233_23314

-- Define the constants
noncomputable def a : ℝ := (3 : ℝ) ^ (0.4 : ℝ)
noncomputable def b : ℝ := (0.4 : ℝ) ^ (3 : ℝ)
noncomputable def c : ℝ := Real.log 3 / Real.log 0.4

-- State the theorem
theorem ordering_abc : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l233_23314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_same_side_line_l233_23380

/-- The reflection of a point (x, y) across (x0, y0) -/
def reflect (x y x0 y0 : ℝ) : ℝ × ℝ :=
  (2 * x0 - x, 2 * y0 - y)

/-- Check if two points are on the same side of a line ax + by + c = 0 -/
def sameSide (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) > 0

theorem reflection_same_side_line (a : ℝ) :
  let A : ℝ × ℝ := (3, 1)
  let C : ℝ × ℝ := (-1/2, 7/2)
  let B : ℝ × ℝ := reflect A.1 A.2 C.1 C.2
  sameSide A.1 A.2 B.1 B.2 3 (-2) a →
  a ∈ Set.Ioi 24 ∪ Set.Iio (-7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_same_side_line_l233_23380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_probability_l233_23373

/-- Triangle with side lengths 1, 1, √2 -/
structure IsoscelesRightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  side1_eq : side1 = 1
  side2_eq : side2 = 1
  hypotenuse_eq : hypotenuse = Real.sqrt 2

/-- Two points chosen randomly on the sides of the triangle -/
structure RandomPoints (T : IsoscelesRightTriangle) where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  on_sides : (point1.1 = 0 ∨ point1.2 = 0 ∨ point1.1 + point1.2 = 1) ∧
             (point2.1 = 0 ∨ point2.2 = 0 ∨ point2.1 + point2.2 = 1)

/-- The probability that the straight-line distance between the points is at least √2/2 -/
noncomputable def probability (T : IsoscelesRightTriangle) : ℝ :=
  1 - Real.pi / 8

/-- The main theorem to prove -/
theorem distance_probability (T : IsoscelesRightTriangle) :
  probability T = 1 - Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_probability_l233_23373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l233_23350

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 3/2) ^ x

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → f a y < f a x

def q (a : ℝ) : Prop := ∀ x : ℝ, (1/2 : ℝ) ^ |x - 1| < a

theorem problem_solution (a : ℝ) :
  (p a ↔ 3/2 < a ∧ a < 5/2) ∧
  (q a ↔ 1 < a) ∧
  (¬(p a ∧ q a) ∧ (p a ∨ q a) ↔ (1 < a ∧ a ≤ 3/2) ∨ 5/2 ≤ a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l233_23350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_741_decimal_l233_23302

theorem smallest_n_with_741_decimal : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (m : ℕ), m > 0 ∧ m < n ∧ 
    Nat.Coprime m n ∧ 
    (741 : ℚ) / 1000 ≤ (m : ℚ) / n ∧ (m : ℚ) / n < (742 : ℚ) / 1000) ∧
  (∀ (k : ℕ), k > 0 ∧ k < n → 
    ¬∃ (j : ℕ), j > 0 ∧ j < k ∧ 
      Nat.Coprime j k ∧ 
      (741 : ℚ) / 1000 ≤ (j : ℚ) / k ∧ (j : ℚ) / k < (742 : ℚ) / 1000) ∧
  n = 999 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_741_decimal_l233_23302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_h_min_value_l233_23374

-- Define the logarithm base 3
noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

-- Define f(x) = log₃(x)
noncomputable def f (x : ℝ) : ℝ := log3 x

-- Define g(x) = f((x+1)/(x-1))
noncomputable def g (x : ℝ) : ℝ := f ((x + 1) / (x - 1))

-- Define h(x) = f(√x) * f(3x)
noncomputable def h (x : ℝ) : ℝ := f (Real.sqrt x) * f (3 * x)

-- Theorem 1: g is an odd function
theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

-- Theorem 2: h achieves its minimum value of 1 when x = 3 for x ∈ [3, 27]
theorem h_min_value :
  ∀ x ∈ Set.Icc 3 27, h x ≥ 1 ∧ h 3 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_h_min_value_l233_23374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_black_correct_l233_23387

/-- The number of white balls in the box -/
def white_balls : ℕ := 8

/-- The number of black balls in the box -/
def black_balls : ℕ := 7

/-- The number of balls drawn from the box -/
def drawn_balls : ℕ := 6

/-- The probability of drawing all black balls -/
def probability_all_black : ℚ := 1 / 715

theorem probability_all_black_correct : 
  (Nat.choose black_balls drawn_balls : ℚ) / (Nat.choose (white_balls + black_balls) drawn_balls) = probability_all_black :=
by sorry

#eval white_balls
#eval black_balls
#eval drawn_balls
#eval probability_all_black

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_black_correct_l233_23387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_digits_l233_23360

theorem count_divisible_digits : 
  (Finset.filter (fun n : Fin 9 => (25 * n.val.succ + n.val.succ) % n.val.succ = 0) Finset.univ).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_digits_l233_23360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_element_value_l233_23364

/-- Represents a 4x4 matrix where each row and column forms an arithmetic sequence -/
def ArithmeticMatrix := Matrix (Fin 4) (Fin 4) ℚ

/-- Checks if a sequence of 4 rational numbers forms an arithmetic progression -/
def isArithmeticSequence (seq : Fin 4 → ℚ) : Prop :=
  ∃ d, ∀ i : Fin 3, seq i.succ - seq i = d

/-- Properties of our specific arithmetic matrix -/
def isSpecialArithmeticMatrix (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 4, isArithmeticSequence (λ j ↦ M i j)) ∧ 
  (∀ j : Fin 4, isArithmeticSequence (λ i ↦ M i j)) ∧
  M 0 0 = 12 ∧ M 3 3 = 72

/-- The center element of a 4x4 matrix -/
def centerElement (M : ArithmeticMatrix) : ℚ :=
  (M 1 1 + M 1 2 + M 2 1 + M 2 2) / 4

theorem center_element_value (M : ArithmeticMatrix) 
  (h : isSpecialArithmeticMatrix M) : 
  centerElement M = 52 := by
  sorry

#check center_element_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_element_value_l233_23364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_rate_proof_l233_23375

noncomputable def sally_hourly_rate (total_amount : ℝ) (hours_worked : ℝ) : ℝ :=
  total_amount / hours_worked

theorem sally_rate_proof (total_amount : ℝ) (hours_worked : ℝ) 
  (h1 : total_amount = 150)
  (h2 : hours_worked = 12) :
  sally_hourly_rate total_amount hours_worked = 12.5 := by
  unfold sally_hourly_rate
  rw [h1, h2]
  norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_rate_proof_l233_23375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_laplace_of_F_is_f_l233_23382

/-- The Laplace transform of a function f(t) --/
noncomputable def laplace_transform (f : ℝ → ℝ) (p : ℝ) : ℝ := 
  ∫ (t : ℝ), f t * Real.exp (-p * t)

/-- The inverse Laplace transform of a function F(p) --/
noncomputable def inverse_laplace_transform (F : ℝ → ℝ) (t : ℝ) : ℝ :=
  sorry  -- Definition of inverse Laplace transform

/-- The given function in Laplace domain --/
noncomputable def F (p : ℝ) : ℝ := p / (p^2 - 2*p + 5)

/-- The claimed original function --/
noncomputable def f (t : ℝ) : ℝ := Real.exp t * (Real.cos (2*t) + (1/2) * Real.sin (2*t))

theorem inverse_laplace_of_F_is_f :
  ∀ t, inverse_laplace_transform F t = f t :=
by
  sorry  -- Proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_laplace_of_F_is_f_l233_23382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l233_23341

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 5

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop :=
  my_circle P.1 P.2

-- Distance function from a point to the origin
noncomputable def distance_to_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1^2 + P.2^2)

-- Theorem statement
theorem max_distance_to_origin :
  ∀ P : ℝ × ℝ, point_on_circle P →
  distance_to_origin P ≤ 3 * Real.sqrt 5 := by
  sorry

#check max_distance_to_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l233_23341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_parenthesizations_eq_catalan_l233_23349

/-- The number of distinct ways to fully parenthesize the product of n numbers -/
def num_parenthesizations (n : ℕ) : ℕ :=
  sorry

/-- The nth Catalan number -/
def nth_catalan (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinct parenthesizations for n numbers
    is equal to the (n-1)th Catalan number -/
theorem num_parenthesizations_eq_catalan (n : ℕ) :
  num_parenthesizations n = nth_catalan (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_parenthesizations_eq_catalan_l233_23349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_city_assignment_l233_23355

-- Define the cities
inductive City
| Yaransk
| Lalsk
| Khalturin
| Slobodskoy
| Kotelnych

-- Define the students
inductive Student
| First
| Second
| Third
| Fourth
| Fifth

-- Define a function to represent the city of each student
def cityOf : Student → City := sorry

-- Define the statements made by each student
def firstStudentStatements : (Student → City) → Prop :=
  fun c => (c Student.First = City.Yaransk) ≠ (c Student.Fourth = City.Lalsk)

def secondStudentStatements : (Student → City) → Prop :=
  fun c => (c Student.Second = City.Yaransk) ≠ (c Student.Third = City.Khalturin)

def thirdStudentStatements : (Student → City) → Prop :=
  fun c => (c Student.Third = City.Yaransk) ≠ (c Student.Fourth = City.Slobodskoy)

def fourthStudentStatements : (Student → City) → Prop :=
  fun c => (c Student.Fourth = City.Lalsk) ≠ (c Student.Fifth = City.Kotelnych)

def fifthStudentStatements : (Student → City) → Prop :=
  fun c => (c Student.Fifth = City.Kotelnych) ≠ (c Student.First = City.Slobodskoy)

-- Define the theorem
theorem olympiad_city_assignment :
  ∃! c : Student → City,
    firstStudentStatements c ∧
    secondStudentStatements c ∧
    thirdStudentStatements c ∧
    fourthStudentStatements c ∧
    fifthStudentStatements c ∧
    c Student.First = City.Slobodskoy ∧
    c Student.Second = City.Lalsk ∧
    c Student.Third = City.Yaransk ∧
    c Student.Fourth = City.Lalsk ∧
    c Student.Fifth = City.Kotelnych :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_city_assignment_l233_23355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_isosceles_right_triangles_l233_23332

/-- Represents a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is an isosceles right triangle -/
def IsoscelesRightTriangle (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2

/-- Checks if two triangles are congruent -/
def Congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

/-- Returns the length of the hypotenuse of a triangle -/
def Hypotenuse (t : Triangle) : ℝ := t.c

/-- Checks if the hypotenuses of two triangles coincide -/
def CoincidingHypotenuses (t1 t2 : Triangle) : Prop :=
  t1.c = t2.c

/-- Calculates the area of overlap between two triangles -/
noncomputable def AreaOfOverlap (t1 t2 : Triangle) : ℝ :=
  (t1.a^2) / 2

/-- The area of overlap between two congruent isosceles right triangles
    with coinciding hypotenuses of length 10 is 25 square units. -/
theorem overlap_area_of_isosceles_right_triangles :
  ∀ (t1 t2 : Triangle) (h : ℝ),
    IsoscelesRightTriangle t1 →
    IsoscelesRightTriangle t2 →
    Congruent t1 t2 →
    Hypotenuse t1 = h →
    Hypotenuse t2 = h →
    CoincidingHypotenuses t1 t2 →
    h = 10 →
    AreaOfOverlap t1 t2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_isosceles_right_triangles_l233_23332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_value_l233_23324

theorem abc_value (a b c : ℝ) 
  (h1 : a^(Real.log a) * b^(Real.log b) * c^(Real.log c) = 5)
  (h2 : a^(Real.log b) * b^(Real.log c) * c^(Real.log a) = Real.sqrt 2) :
  a * b * c = 10 ∨ a * b * c = (1 : ℝ) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_value_l233_23324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_area_ratio_proof_l233_23381

noncomputable def cube_tetrahedron_area_ratio : ℝ := Real.sqrt 3

def cube_side_length : ℝ := 2

noncomputable def tetrahedron_side_length : ℝ := 2 * Real.sqrt 2

def cube_surface_area : ℝ := 6 * cube_side_length ^ 2

noncomputable def tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length ^ 2

theorem cube_tetrahedron_area_ratio_proof :
  cube_surface_area / tetrahedron_surface_area = cube_tetrahedron_area_ratio :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_area_ratio_proof_l233_23381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l233_23308

/-- Calculates the length of a faster train given the speeds of two trains and the time it takes for the faster train to pass a person in the slower train. -/
theorem faster_train_length 
  (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : faster_speed = 220) 
  (h2 : slower_speed = 42) 
  (h3 : passing_time = 25) : 
  (faster_speed - slower_speed) * (5/18) * passing_time = 1236 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l233_23308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l233_23344

noncomputable def a (n : ℕ) : ℝ := ((2 + Real.sqrt 3) ^ n - (2 - Real.sqrt 3) ^ n) / (2 * Real.sqrt 3)

theorem sequence_properties :
  (∀ n : ℕ, ∃ k : ℤ, a n = k) ∧
  (∀ n : ℕ, (∃ k : ℤ, a n = 3 * k) ↔ (∃ m : ℕ, n = 3 * m)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l233_23344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l233_23389

/-- The number of ways to distribute candies to trick-or-treaters -/
def distribute_candies (total_candies : ℕ) (num_trick_or_treaters : ℕ) : ℕ :=
  sorry

/-- At least one candy per trick-or-treater -/
def at_least_one (distribution : List ℕ) : Prop :=
  ∀ x, x ∈ distribution → x ≥ 1

/-- No two trick-or-treaters receive the same number of candies -/
def all_different (distribution : List ℕ) : Prop :=
  ∀ x y, x ∈ distribution → y ∈ distribution → x ≠ y → x ≠ y

theorem candy_distribution_theorem :
  ∃ (valid_distributions : List (List ℕ)),
    (∀ dist, dist ∈ valid_distributions →
      dist.length = 3 ∧
      dist.sum = 15 ∧
      at_least_one dist ∧
      all_different dist) ∧
    valid_distributions.length = 72 :=
  sorry

#check candy_distribution_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l233_23389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_match_iff_p_plus_q_eq_one_l233_23335

/-- 
Represents a quadratic equation of the form ax² + bx + c = 0,
where a, b, and c are real numbers and a ≠ 0.
-/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- 
Given two quadratic equations f(x) and f(y), this theorem states that
their roots are identical if and only if p + q = 1, where p and q are
related to the coefficients of the equations as specified.
-/
theorem roots_match_iff_p_plus_q_eq_one
  (fx fy : QuadraticEquation)
  (p q : ℝ)
  (h1 : fx.a = 1)
  (h2 : fx.b = 1 + p + q)
  (h3 : fx.c = 1)
  (h4 : fy.a = p)
  (h5 : fy.b = 2 * (1 - q))
  (h6 : fy.c = q)
  (h7 : (1 + p + q) / 1 = (2 * (1 - q)) / p)
  (h8 : (2 * (1 - q)) / p = (1 - p + q) / q) :
  (∀ r : ℝ, fx.a * r^2 + fx.b * r + fx.c = 0 ↔ fy.a * r^2 + fy.b * r + fy.c = 0) ↔
  p + q = 1 := by
  sorry

#check roots_match_iff_p_plus_q_eq_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_match_iff_p_plus_q_eq_one_l233_23335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_max_value_f_reaches_max_l233_23303

-- Define the function f(x) = cos(x/3) + 2
noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 3) + 2

-- Theorem for the period of f(x)
theorem f_period : ∀ x : ℝ, f (x + 6 * Real.pi) = f x := by
  sorry

-- Theorem for the maximum value of f(x)
theorem f_max_value : ∀ x : ℝ, f x ≤ 3 := by
  sorry

-- Theorem that there exists a point where f(x) reaches its maximum value
theorem f_reaches_max : ∃ x : ℝ, f x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_max_value_f_reaches_max_l233_23303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l233_23394

/-- The distance between two points in the Cartesian plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The sum of distances from a point (x, y) to the given points -/
noncomputable def sum_distances (x y : ℝ) : ℝ :=
  distance x y 1 2 + distance x y 1 5 + distance x y 3 6 + distance x y 7 (-1)

/-- The point (2, 4) minimizes the sum of distances to the given points -/
theorem min_sum_distances :
  ∀ x y : ℝ, sum_distances 2 4 ≤ sum_distances x y := by
  sorry

#check min_sum_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l233_23394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_plus_abs_two_minus_three_power_zero_sqrt_two_times_sqrt_six_minus_sqrt_three_plus_sqrt_twelve_div_sqrt_two_l233_23388

-- Part 1
theorem cube_root_eight_plus_abs_two_minus_three_power_zero :
  (8 : ℝ) ^ (1/3) + Real.sqrt ((-2)^2) - (-3)^0 = 3 := by sorry

-- Part 2
theorem sqrt_two_times_sqrt_six_minus_sqrt_three_plus_sqrt_twelve_div_sqrt_two :
  Real.sqrt 2 * (Real.sqrt 6 - Real.sqrt 3) + Real.sqrt 12 / Real.sqrt 2 = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_plus_abs_two_minus_three_power_zero_sqrt_two_times_sqrt_six_minus_sqrt_three_plus_sqrt_twelve_div_sqrt_two_l233_23388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l233_23325

/-- A function g with specific properties -/
noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- Theorem stating the unique number not in the range of g -/
theorem unique_number_not_in_range_of_g 
  (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : g p q r s 31 = 31)
  (h2 : g p q r s 41 = 41)
  (h3 : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l233_23325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l233_23330

/-- The sum of the series from n = 3 to infinity of (5n^3 - 2n^2 - n + 3) / (n^6 - 2n^5 + 2n^4 - 2n^3 + 2n^2 - 2n + 1) is equal to 1. -/
theorem series_sum_equals_one :
  ∑' n : ℕ, (5 * n^3 - 2 * n^2 - n + 3) / (n^6 - 2*n^5 + 2*n^4 - 2*n^3 + 2*n^2 - 2*n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l233_23330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_to_y_axis_l233_23326

noncomputable section

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus
def focus : ℝ × ℝ := (1/2, 0)

-- Define the property that A and B are on the parabola
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  is_on_parabola A.1 A.2 ∧ is_on_parabola B.1 B.2

-- Define the distance condition
def distance_condition (A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) +
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 8

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (A B : ℝ × ℝ) 
  (h1 : points_on_parabola A B) 
  (h2 : distance_condition A B) :
  (A.1 + B.1) / 2 = 7/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_to_y_axis_l233_23326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l233_23317

theorem trigonometric_identity (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 2 / 2) : 
  Real.cos (2 * α) / Real.cos (α - π / 4) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l233_23317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l233_23311

/-- The number of ways to assign roles in a play. -/
def assignRoles (numMen numWomen : ℕ) (numMaleRoles numFemaleRoles numNeutralRoles : ℕ) : ℕ :=
  (numMen.choose numMaleRoles * numMaleRoles.factorial) * 
  (numWomen.choose numFemaleRoles * numFemaleRoles.factorial) * 
  ((numMen + numWomen - numMaleRoles - numFemaleRoles).choose numNeutralRoles * numNeutralRoles.factorial)

/-- Theorem stating the number of ways to assign roles in the given scenario. -/
theorem role_assignment_count :
  assignRoles 6 7 3 3 2 = 1058400 := by
  rfl

#eval assignRoles 6 7 3 3 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l233_23311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_l233_23347

-- Define the polar coordinates of point M
noncomputable def r : ℝ := 2
noncomputable def θ : ℝ := Real.pi / 3

-- Define the conversion functions from polar to rectangular coordinates
noncomputable def x (r θ : ℝ) : ℝ := r * Real.cos θ
noncomputable def y (r θ : ℝ) : ℝ := r * Real.sin θ

-- State the theorem
theorem polar_to_rectangular :
  (x r θ, y r θ) = (1, Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_l233_23347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l233_23397

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem
theorem triangle_problem (t : Triangle) 
    (h1 : (2 * t.b - t.a) * Real.cos t.C = t.c * Real.cos t.A)
    (h2 : t.c = Real.sqrt 3) 
    (h3 : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2) :
  t.C = π / 3 ∧ t.a + t.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l233_23397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_n_values_l233_23327

/-- Represents the number of vertices on one side of the diagonal --/
def x : ℕ → ℕ := sorry

/-- Represents a convex n-gon --/
structure ConvexNGon (n : ℕ) where
  vertices_ge_three : n ≥ 3

/-- Represents a diagonal in the n-gon --/
structure Diagonal (n : ℕ) where
  intersecting_diagonals : ℕ
  intersection_relation : ∀ (x : ℕ), intersecting_diagonals = (x - 1) * (n - x - 1)

/-- The theorem statement --/
theorem sum_of_possible_n_values (n : ℕ) (ngon : ConvexNGon n) (d : Diagonal n) :
  d.intersecting_diagonals = 14 →
  (∃ n₁ n₂ : ℕ, (∃ _ : ConvexNGon n₁, ∃ _ : Diagonal n₁, d.intersecting_diagonals = 14) ∧
                (∃ _ : ConvexNGon n₂, ∃ _ : Diagonal n₂, d.intersecting_diagonals = 14) ∧
                n₁ + n₂ = 28 ∧
                (∀ m : ℕ, (∃ _ : ConvexNGon m, ∃ _ : Diagonal m, d.intersecting_diagonals = 14) → m = n₁ ∨ m = n₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_n_values_l233_23327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l233_23312

-- Define a triangle with angles in arithmetic sequence and sides in geometric sequence
structure SpecialTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_arithmetic_seq : 2 * B = A + C
  angle_sum : A + B + C = Real.pi
  side_geometric_seq : b^2 = a * c
  law_of_sines : a / Real.sin A = b / Real.sin B
  side_order : 0 < a ∧ 0 < b ∧ 0 < c

-- State the theorem
theorem special_triangle_properties (t : SpecialTriangle) :
  Real.cos t.B = 1/2 ∧
  Real.sin t.A = Real.sqrt 6 / 4 ∧
  Real.sin t.C = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l233_23312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l233_23392

-- Define the function representing the equation
noncomputable def f (θ : ℝ) : ℝ := 1 - 3 * Real.sin θ + 5 * Real.cos (3 * θ)

-- Theorem statement
theorem equation_solutions :
  ∃ (S : Finset ℝ), 
    (∀ θ ∈ S, 0 < θ ∧ θ < 2 * Real.pi ∧ f θ = 0) ∧ 
    (∀ θ, 0 < θ → θ < 2 * Real.pi → f θ = 0 → θ ∈ S) ∧
    Finset.card S = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l233_23392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_order_l233_23386

-- Define the circles
noncomputable def circle_X : ℝ := Real.sqrt 10
def circle_Y : ℝ := 4  -- derived from circumference = 8π
def circle_Z : ℝ := 4  -- derived from area = 16π

-- Theorem statement
theorem circle_radius_order : 
  circle_X < circle_Y ∧ circle_X < circle_Z ∧ circle_Y = circle_Z :=
by
  -- Split the conjunction into three parts
  constructor
  · -- Prove circle_X < circle_Y
    sorry
  constructor
  · -- Prove circle_X < circle_Z
    sorry
  · -- Prove circle_Y = circle_Z
    rfl  -- reflexivity, since they are defined to be equal


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_order_l233_23386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_one_l233_23348

-- Define the function g first
noncomputable def g (x : ℝ) : ℝ := -x^2 + 3*x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 3*x else g x

-- State the theorem
theorem f_g_minus_one (h : ∀ x, f (-x) = -f x) : f (g (-1)) = -28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_one_l233_23348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_properties_l233_23305

/-- A distribution with given properties --/
structure Distribution where
  mean : ℝ
  std_dev : ℝ
  skewness : ℝ
  kurtosis : ℝ
  symmetric : Bool
  within_one_std_dev : ℝ

/-- The percentage of the distribution below mean + std_dev --/
def j_percentage (d : Distribution) : ℝ := 84

/-- The third standardized moment (skewness) --/
def third_moment (d : Distribution) : ℝ := d.skewness

/-- The fourth standardized moment (kurtosis) --/
def fourth_moment (d : Distribution) : ℝ := d.kurtosis

theorem distribution_properties (d : Distribution) 
  (h_symmetric : d.symmetric = true) 
  (h_within_one_std : d.within_one_std_dev = 68) : 
  j_percentage d = 84 ∧ 
  third_moment d = d.skewness ∧ 
  fourth_moment d = d.kurtosis := by
  sorry  -- Proof skipped

#check distribution_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_properties_l233_23305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_erased_numbers_l233_23329

def erasureRule (S : Finset Nat) : Finset Nat :=
  S.filter (fun n => ∃ m ∈ S, m ≠ n ∧ n % m = 0)

def erasureProcess (S : Finset Nat) : List (Finset Nat) :=
  let rec process (current : Finset Nat) (acc : List (Finset Nat)) (fuel : Nat) : List (Finset Nat) :=
    match fuel with
    | 0 => acc.reverse
    | fuel'+1 =>
      let next := erasureRule current
      if next = current then acc.reverse
      else process next (next :: acc) fuel'
  process S [S] 100

theorem last_erased_numbers :
  let initial_set := Finset.range 100
  let process := erasureProcess initial_set
  let final_step := process.getLast?
  final_step = some {64, 96} := by sorry

#eval erasureProcess (Finset.range 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_erased_numbers_l233_23329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_example_l233_23378

/-- The curved surface area of a cone with given slant height and base radius. -/
noncomputable def coneCurvedSurfaceArea (slantHeight : ℝ) (baseRadius : ℝ) : ℝ :=
  Real.pi * baseRadius * slantHeight

/-- Theorem: The curved surface area of a cone with slant height 10 cm and base radius 5 cm is 50π cm². -/
theorem cone_curved_surface_area_example :
  coneCurvedSurfaceArea 10 5 = 50 * Real.pi := by
  unfold coneCurvedSurfaceArea
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_example_l233_23378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagram_area_ratio_l233_23354

/-- Represents a hexagram formed by overlapping two equilateral triangles -/
structure Hexagram where
  vertices : Finset (ℝ × ℝ)
  is_hexagon : vertices.card = 6
  is_regular : Bool -- Simplified for now, can be expanded later
  shaded_triangles : Finset (Finset (ℝ × ℝ))
  unshaded_triangles : Finset (Finset (ℝ × ℝ))
  shaded_count : shaded_triangles.card = 18
  unshaded_count : unshaded_triangles.card = 6

/-- The area of a triangle -/
noncomputable def triangle_area (t : Finset (ℝ × ℝ)) : ℝ := 
  sorry

/-- The total area of a set of triangles -/
noncomputable def total_area (triangles : Finset (Finset (ℝ × ℝ))) : ℝ :=
  (triangles.toList.map triangle_area).sum

/-- Theorem: The ratio of shaded to unshaded area in a hexagram is 3:1 -/
theorem hexagram_area_ratio (h : Hexagram) :
  (total_area h.shaded_triangles) / (total_area h.unshaded_triangles) = 3 := by
  sorry

#check hexagram_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagram_area_ratio_l233_23354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_increasing_interval_l233_23320

-- Define the cosine function
noncomputable def f (x : ℝ) := Real.cos x

-- Define what it means for a function to be increasing on an interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem cosine_increasing_interval (a : ℝ) :
  increasing_on f π a → π < a ∧ a ≤ 2 * π := by
  sorry

-- You can add more definitions or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_increasing_interval_l233_23320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l233_23383

/-- Given a 2x2 matrix B and a constant m, proves that if B^(-1) + 2I = mB, 
    then c = 5 and m = 1/161, where c is the bottom-right element of B. -/
theorem matrix_equation_solution (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (m : ℝ) (c : ℝ) (h1 : B 0 0 = 1) (h2 : B 0 1 = 4) (h3 : B 1 0 = 7) (h4 : B 1 1 = c)
  (h5 : B⁻¹ + 2 • (1 : Matrix (Fin 2) (Fin 2) ℝ) = m • B) : 
  c = 5 ∧ m = 1/161 := by
  sorry

#check matrix_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l233_23383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_gon_side_length_approximation_l233_23365

/-- Approximation of π -/
noncomputable def π : ℝ := Real.pi

/-- Approximation of π² -/
noncomputable def π_squared : ℝ := π * π

/-- Side length approximation for regular 11-gon -/
noncomputable def side_length_approx : ℝ := 1/3 + 1/5 + 1/51 + 1/95

/-- Theorem: The approximation is close to the actual side length -/
theorem eleven_gon_side_length_approximation :
  |2 * Real.sin (π / 11) - side_length_approx| < 1e-5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_gon_side_length_approximation_l233_23365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_16_l233_23359

noncomputable def t : ℝ := Real.sqrt (4 / 3)  -- This is a placeholder definition for t

-- Define the series sum
noncomputable def seriesSum (t : ℝ) : ℝ := ∑' n, (n + 1) * t^(3*n + 2)

theorem series_sum_equals_16 (h : t^3 - (1/4)*t - 1 = 0) (ht : t > 0) : 
  seriesSum t = 16 := by
  sorry

#check series_sum_equals_16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_16_l233_23359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l233_23331

/-- The volume of a regular tetrahedron circumscribed around a sphere of radius R -/
noncomputable def tetrahedronVolume (R : ℝ) : ℝ := 8 * R^3 * Real.sqrt 3

/-- Theorem stating that the volume of a regular tetrahedron circumscribed around a sphere of radius R is 8 * R^3 * √3 -/
theorem regular_tetrahedron_volume (R : ℝ) (h : R > 0) :
  tetrahedronVolume R = 8 * R^3 * Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l233_23331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_equality_l233_23384

/-- The nth term of the monomial sequence -/
noncomputable def nth_term (n : ℕ) : ℝ → ℝ := λ x => (-1)^(n-1) * Real.sqrt (n : ℝ) * x^n

/-- The 10th term of the monomial sequence -/
noncomputable def tenth_term : ℝ → ℝ := λ x => -Real.sqrt 10 * x^10

theorem tenth_term_equality :
  nth_term 10 = tenth_term :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_equality_l233_23384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_for_special_function_l233_23371

theorem integral_bounds_for_special_function 
  (f : ℝ → ℝ) 
  (hf_cont : ContinuousOn f (Set.Icc 0 1))
  (hf_range : ∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1)
  (hf_comp : ∀ x ∈ Set.Icc 0 1, f (f x) = 1) :
  ∃ (I : ℝ), I = ∫ x in (Set.Icc 0 1), f x ∧ 3/4 < I ∧ I ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_for_special_function_l233_23371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_l233_23315

theorem binomial_coefficient_equation (x : ℝ) : 
  (Nat.choose 10 (Int.floor x).toNat = Nat.choose 10 (Int.floor (3*x - 2)).toNat) → (x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_l233_23315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_theorem_of_calculus_l233_23393

theorem fundamental_theorem_of_calculus 
  {f : ℝ → ℝ} {F : ℝ → ℝ} {a b : ℝ} 
  (h_cont : ContinuousOn f (Set.Icc a b))
  (h_deriv : ∀ x ∈ Set.Icc a b, HasDerivAt F (f x) x) :
  ∫ x in a..b, f x = F b - F a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_theorem_of_calculus_l233_23393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_sales_theorem_l233_23345

/-- Represents the sales model for a shirt department in a shopping mall -/
structure ShirtSales where
  initial_sales : ℕ  -- Initial average daily sales
  initial_profit : ℕ  -- Initial profit per shirt in yuan
  sales_increase : ℕ  -- Additional shirts sold per yuan of price reduction

/-- Calculates the new average daily sales after a price reduction -/
def new_sales (model : ShirtSales) (price_reduction : ℕ) : ℕ :=
  model.initial_sales + model.sales_increase * price_reduction

/-- Calculates the daily profit after a price reduction -/
def daily_profit (model : ShirtSales) (price_reduction : ℕ) : ℕ :=
  (model.initial_profit - price_reduction) * (new_sales model price_reduction)

theorem shirt_sales_theorem (model : ShirtSales) 
  (h1 : model.initial_sales = 30)
  (h2 : model.initial_profit = 40)
  (h3 : model.sales_increase = 2) :
  (new_sales model 3 = 36) ∧ 
  (∃ x : ℕ, daily_profit model x = 1200 ∧ x = 25) := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_sales_theorem_l233_23345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l233_23336

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_problem (a b c A B C : ℝ) (h_triangle : A + B + C = π)
    (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) (h_sides : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_sine_rule : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  (b * Real.sin (2 * C) = c * Real.sin B → C = π / 3) ∧
  (b * Real.sin (2 * C) = c * Real.sin B → Real.sin (B - π / 3) = 3 / 5 → Real.sin A = (4 * Real.sqrt 3 - 3) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l233_23336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fp_difference_divisible_l233_23300

open BigOperators

def fp (p : ℕ) (x : ℕ) : ℚ :=
  ∑ k in Finset.range (p - 1), 1 / (p * x + k + 1)^2

theorem fp_difference_divisible (p : ℕ) (x y : ℕ) 
    (hp : p.Prime ∧ p > 5) (hx : x > 0) (hy : y > 0) : 
  ∃ (n : ℤ), (fp p x - fp p y : ℚ) = n / (p^3 : ℤ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fp_difference_divisible_l233_23300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enchilada_taco_pricing_l233_23334

/-- The price of enchiladas and tacos given two pricing conditions -/
theorem enchilada_taco_pricing 
  (e t : ℝ)
  (price1 : 4 * e + 5 * t = 5)
  (price2 : 6 * e + 3 * t = 5.4)
  : ‖7 * e + 6 * t - 7.47‖ < 0.005 := by
  sorry

#check enchilada_taco_pricing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enchilada_taco_pricing_l233_23334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l233_23319

noncomputable def a : ℝ × ℝ := (1/2, Real.sqrt 3/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2 + 2

def is_maximum (x : ℝ) : Prop := ∃ (k : ℤ), x = 2 * k * Real.pi + Real.pi / 6

def is_minimum (x : ℝ) : Prop := ∃ (k : ℤ), x = 2 * k * Real.pi - 5 * Real.pi / 6

theorem f_properties :
  (∀ x, f x ≤ 3) ∧
  (∀ x, f x ≥ 1) ∧
  (∀ x, is_maximum x → f x = 3) ∧
  (∀ x, is_minimum x → f x = 1) ∧
  (∀ x ∈ Set.Icc (Real.pi / 6) (7 * Real.pi / 6), 
    ∀ y ∈ Set.Icc x (7 * Real.pi / 6), f y ≤ f x) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), 
    (x < Real.pi / 6 → f x < f (Real.pi / 6)) ∧ 
    (x > 7 * Real.pi / 6 → f x > f (7 * Real.pi / 6))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l233_23319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_usd_to_krw_l233_23357

/-- The exchange rate from U.S. dollars to Korean won -/
noncomputable def exchange_rate (usd : ℝ) (krw : ℝ) : ℝ := krw / usd

/-- Theorem stating the exchange rate given the problem conditions -/
theorem exchange_rate_usd_to_krw :
  let usd : ℝ := 140
  let krw : ℝ := 158760
  exchange_rate usd krw = 1134 := by
  -- Unfold the definitions
  unfold exchange_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_usd_to_krw_l233_23357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l233_23304

noncomputable def f (x : ℝ) : ℝ := 2 / (2 * Real.sin x - 1)

theorem f_range : 
  Set.range f = {y : ℝ | y ≤ -2/3 ∨ y ≥ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l233_23304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_300_approximation_l233_23358

-- Define the approximation for cube root of 0.3
noncomputable def cube_root_0_3 : ℝ := 0.6694

-- Define the approximation for cube root of 300
noncomputable def cube_root_300 : ℝ := 6.694

-- Theorem statement
theorem cube_root_300_approximation (ε : ℝ) (h_ε : ε > 0) :
  ∃ δ > 0, (|cube_root_0_3^3 - 0.3| < δ → |cube_root_300 - Real.rpow 300 (1/3)| < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_300_approximation_l233_23358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_9n_is_9_l233_23391

/-- A positive integer whose digits strictly increase from left to right -/
def StrictlyIncreasingDigits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.repr.get! i).toNat < (n.repr.get! j).toNat

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  n.repr.toList.map Char.toNat |>.sum

/-- Theorem: For any positive integer n with strictly increasing digits,
    the sum of digits of 9n is always 9 -/
theorem sum_of_digits_9n_is_9 (n : ℕ) (h : StrictlyIncreasingDigits n) :
  sumOfDigits (9 * n) = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_9n_is_9_l233_23391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l233_23352

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define function g
noncomputable def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c (2^x)

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : ∀ x, f a b c (x + 1) - f a b c x = 4 * x + 1)
  (h2 : f a b c 0 = 3) :
  (∀ x, f a b c x = 2 * x^2 - x + 3) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 0, g a b c x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 0, g a b c x = 4) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 0, g a b c x ≥ 23/8) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 0, g a b c x = 23/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l233_23352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_perimeter_range_l233_23316

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

-- Define the given condition
noncomputable def given_condition (t : Triangle) : Prop :=
  (Real.sin t.A - Real.sin t.B) / (Real.sqrt 3 * t.a - t.c) = Real.sin t.C / (t.a + t.b)

-- Theorem 1: Prove that B = π/6
theorem angle_B_value (t : Triangle) 
  (h1 : is_acute_triangle t)
  (h2 : given_condition t) : 
  t.B = Real.pi/6 := by sorry

-- Theorem 2: Prove the range of perimeter when a = 2
theorem perimeter_range (t : Triangle) 
  (h1 : is_acute_triangle t)
  (h2 : given_condition t)
  (h3 : t.a = 2) :
  3 + Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c < 2 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_perimeter_range_l233_23316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_equation_solution_l233_23318

-- Define the differential equation
def euler_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x^2 * ((deriv (deriv y)) x) + 2*x * ((deriv y) x) - 6 * (y x) = 0

-- Define the general solution
noncomputable def general_solution (C1 C2 : ℝ) (x : ℝ) : ℝ :=
  C1 / x^3 + C2 * x^2

-- Theorem statement
theorem euler_equation_solution (C1 C2 : ℝ) :
  ∀ x : ℝ, x ≠ 0 → euler_equation (general_solution C1 C2) x :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_equation_solution_l233_23318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l233_23361

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_range : Set.range f = Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l233_23361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l233_23322

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 * x + 2) + 1 / (x - 2)

theorem domain_of_f :
  {x : ℝ | x ≥ -2/3 ∧ x ≠ 2} = {x : ℝ | 3 * x + 2 ≥ 0 ∧ x - 2 ≠ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l233_23322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_on_ellipse_l233_23343

-- Define the ellipse equation
def on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the foci
noncomputable def left_focus : ℝ × ℝ := (-2 * Real.sqrt 3, 0)
noncomputable def right_focus : ℝ × ℝ := (2 * Real.sqrt 3, 0)

-- Define the condition for right angle at P
def right_angle_at_p (x y : ℝ) : Prop :=
  let v1 := (left_focus.1 - x, left_focus.2 - y)
  let v2 := (right_focus.1 - x, right_focus.2 - y)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem statement
theorem four_points_on_ellipse :
  ∃! (s : Finset (ℝ × ℝ)), s.card = 4 ∧ 
  ∀ p ∈ s, on_ellipse p.1 p.2 ∧ right_angle_at_p p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_on_ellipse_l233_23343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_A_to_A_l233_23366

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The x-coordinate of a point after reflection over the y-axis -/
def reflect_x (x : ℝ) : ℝ := -x

theorem reflection_distance_A_to_A' :
  let xA : ℝ := 3
  let yA : ℝ := -2
  distance xA yA (reflect_x xA) yA = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_A_to_A_l233_23366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bagels_bought_l233_23307

/-- Represents the number of bagels bought in a week -/
def num_bagels : ℕ := sorry

/-- Represents the number of muffins bought in a week -/
def num_muffins : ℕ := sorry

/-- The total number of items bought in a week is 7 -/
axiom total_items : num_bagels + num_muffins = 7

/-- The cost of a bagel in cents -/
def bagel_cost : ℕ := 90

/-- The cost of a muffin in cents -/
def muffin_cost : ℕ := 60

/-- The total cost in cents is divisible by 100 (whole number of dollars) -/
axiom total_cost_divisible : 
  ∃ (k : ℕ), bagel_cost * num_bagels + muffin_cost * num_muffins = 100 * k

/-- Theorem: The number of bagels bought is 6 -/
theorem bagels_bought : num_bagels = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bagels_bought_l233_23307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_in_unit_cube_l233_23372

structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

noncomputable def distance_between_skew_lines (c : Cube) : ℝ :=
  (Real.sqrt 6) / 6

theorem distance_between_lines_in_unit_cube :
  ∀ (c : Cube), c.edge_length = 1 →
  distance_between_skew_lines c = (Real.sqrt 6) / 6 := by
  sorry

#check distance_between_lines_in_unit_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_in_unit_cube_l233_23372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_is_12_5_l233_23310

/-- Calculates the price of a turban given the full year salary, service duration, and actual payment --/
def turban_price
  (full_year_salary : ℚ)  -- Full year salary in cash
  (service_duration : ℚ)  -- Service duration in years
  (actual_payment : ℚ)    -- Actual payment in cash
  : ℚ :=
  full_year_salary * service_duration - actual_payment

theorem turban_price_is_12_5
  (h1 : turban_price 90 (3/4) 55 = 12.5)
  : turban_price 90 (3/4) 55 = 12.5 :=
by
  -- Prove that the turban price is 12.5
  sorry

#eval turban_price 90 (3/4) 55  -- Should evaluate to 12.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_is_12_5_l233_23310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l233_23346

/-- The molar mass of hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- The molar mass of chlorine in g/mol -/
noncomputable def molar_mass_Cl : ℝ := 35.45

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of chlorous acid (HClO2) in g/mol -/
noncomputable def molar_mass_HClO2 : ℝ := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

/-- The mass percentage of hydrogen in chlorous acid -/
noncomputable def mass_percentage_H : ℝ := (molar_mass_H / molar_mass_HClO2) * 100

/-- Theorem: The mass percentage of hydrogen in chlorous acid is approximately 1.475% -/
theorem mass_percentage_H_approx :
  abs (mass_percentage_H - 1.475) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l233_23346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_terms_is_four_l233_23362

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The quadratic inequality has solution set [1/3, 4/5] -/
def inequality_solution_set (a₁ d c : ℝ) : Prop :=
  ∀ x, a₁ * x^2 + (d/2 - a₁) * x + c ≥ 0 ↔ 1/3 ≤ x ∧ x ≤ 4/5

theorem min_sum_terms_is_four (a₁ d c : ℝ) :
  inequality_solution_set a₁ d c →
  (∃ n : ℕ+, ∀ m : ℕ+, S_n a₁ d n ≤ S_n a₁ d m) →
  (∃ n : ℕ+, n = 4 ∧ ∀ m : ℕ+, m < n → S_n a₁ d m > S_n a₁ d n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_terms_is_four_l233_23362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l233_23342

-- Define the ⊕ operation
noncomputable def circplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (circplus 1 x) * x - (circplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 6 ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≤ M) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) 2 ∧ f x = M) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l233_23342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_one_l233_23377

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin x - 1 else x^2

-- Theorem statement
theorem integral_f_minus_one_to_one :
  ∫ x in Set.Icc (-1) 1, f x = Real.cos 1 - 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_one_l233_23377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fours_in_500_l233_23390

/-- Count occurrences of a digit in a range of numbers -/
def countDigitOccurrences (start : Nat) (stop : Nat) (digit : Nat) : Nat :=
  sorry

/-- The number of times the digit 4 appears in the numbers from 1 to 500 -/
theorem count_fours_in_500 : countDigitOccurrences 1 500 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fours_in_500_l233_23390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmers_market_cost_l233_23395

/-- The cost of buying oranges and apples at a farmer's market -/
theorem farmers_market_cost (orange_price orange_weight apple_price apple_weight orange_buy apple_buy : ℚ) 
  (h1 : orange_price = 3)
  (h2 : orange_weight = 4)
  (h3 : apple_price = 5)
  (h4 : apple_weight = 6)
  (h5 : orange_buy = 12)
  (h6 : apple_buy = 18)
  : (orange_price / orange_weight * orange_buy) + 
    (apple_price / apple_weight * apple_buy) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmers_market_cost_l233_23395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_three_digits_l233_23370

theorem same_last_three_digits (N : ℕ) (hN : N > 0) :
  (∃ (a b c : ℕ), a ≠ 0 ∧ 
   N % 1000 = 100 * a + 10 * b + c ∧
   (N^2) % 1000 = 100 * a + 10 * b + c) →
  (N % 100 = 24) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_three_digits_l233_23370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_hop_l233_23368

theorem frog_hop (p q d : ℕ) (hp : p > 0) (hq : q > 0) (hd : d > 0) 
  (hpq : d < p + q) (hcoprime : Nat.Coprime p q) : 
  ∃ (a b : ℤ), |a * p - b * q| = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_hop_l233_23368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_travel_time_l233_23301

/-- The time (in hours) it takes for the motorboat to go from A to B -/
def t : ℝ := 6

/-- The speed of the river current (km/h) -/
def r : ℝ := 0  -- We'll use a variable, but define it as 0 for now

/-- The speed of the motorboat relative to the river (km/h) -/
def p : ℝ := 0  -- We'll use a variable, but define it as 0 for now

/-- The speed of the kayak (km/h) -/
def s (r : ℝ) : ℝ := r + 2

/-- The total time of the journey (hours) -/
def total_time : ℝ := 12

theorem motorboat_travel_time (hp : p > r + 2) :
  t = total_time * (s r - p) / r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_travel_time_l233_23301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_arrangement_probability_l233_23306

def num_volumes : ℕ := 4

def total_arrangements (n : ℕ) : ℕ := n.factorial

def favorable_outcomes : ℕ := 2

def probability (favorable : ℕ) (total : ℕ) : ℚ := 
  (favorable : ℚ) / (total : ℚ)

theorem volume_arrangement_probability :
  probability favorable_outcomes (total_arrangements num_volumes) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_arrangement_probability_l233_23306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l233_23353

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_value (α : ℝ) :
  (power_function α (1/2) = 4) → (power_function α 8 = 1/64) := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l233_23353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l233_23351

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc 0 2

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.union (Set.Ioo (1/2) 1) (Set.Ioc 1 (3/2))

-- Theorem statement
theorem domain_shift :
  (∀ x ∈ domain_f_shifted, ∃ y, f (x + 1) = y) →
  (∀ x ∈ domain_f, ∃ y, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l233_23351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_function_properties_l233_23376

noncomputable def tourist_function (x : ℝ) : ℝ := 200 * Real.sin (Real.pi * x / 6 - 5 * Real.pi / 6) + 300

theorem tourist_function_properties :
  ∀ x : ℝ,
  (∀ y : ℝ, tourist_function (x + 12) = tourist_function x) ∧
  (tourist_function 2 = 100) ∧
  (tourist_function 8 = 500) ∧
  (∀ y z : ℝ, 2 ≤ y ∧ y < z ∧ z ≤ 8 → tourist_function y < tourist_function z) →
  (∀ m : ℕ, m ∈ ({6, 7, 8, 9, 10} : Set ℕ) → tourist_function (m : ℝ) ≥ 400) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_function_properties_l233_23376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_is_circle_l233_23339

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- The distance between two points in a 2D plane -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A circle passes through a point if the distance from its center to the point equals its radius -/
def passesThrough (c : Circle) (p : Point) : Prop :=
  distance c.center p = c.radius

theorem locus_of_centers_is_circle (a : ℝ) (fixedPoint : Point) :
  ∃ (centerCircle : Circle), ∀ (c : Circle),
    c.radius = a → passesThrough c fixedPoint →
      passesThrough centerCircle c.center := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_is_circle_l233_23339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_not_even_f_odd_but_not_even_l233_23385

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (9 - x^2) / (abs (6 - x) - 6)

-- Define the domain of f
def domain (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 3 ∧ x ≠ 0

-- Theorem stating that f is odd
theorem f_is_odd : ∀ x : ℝ, domain x → f (-x) = -f x := by sorry

-- Theorem stating that f is not even
theorem f_not_even : ¬(∀ x : ℝ, domain x → f (-x) = f x) := by sorry

-- Main theorem: f is odd but not even
theorem f_odd_but_not_even : (∀ x : ℝ, domain x → f (-x) = -f x) ∧ ¬(∀ x : ℝ, domain x → f (-x) = f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_not_even_f_odd_but_not_even_l233_23385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_line_and_parabola_l233_23369

-- Define the functions
def f (x : ℝ) : ℝ := x - 4

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * x)

-- State the theorem
theorem area_between_line_and_parabola :
  ∃ (a b : ℝ), a < b ∧ 
  (∫ (x : ℝ) in a..b, g x - f x) = 18 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_line_and_parabola_l233_23369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_l233_23367

noncomputable def a : Fin 2 → ℝ := ![1, 2]
noncomputable def b : Fin 2 → ℝ := ![-1, 3]

noncomputable def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

noncomputable def projection (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  let scalar := (dot_product v w) / (magnitude w)^2
  ![scalar * (w 0), scalar * (w 1)]

theorem projection_of_a_onto_b :
  projection a b = ![-1/2, 3/2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_l233_23367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_not_increasing_l233_23313

theorem log_inequality_implies_not_increasing (a b c : ℝ) :
  (0 < a ∧ a ≠ 1) → (0 < b ∧ b ≠ 1) → (0 < c ∧ c ≠ 1) →
  (Real.log 2 / Real.log a < Real.log 2 / Real.log b) → 
  (Real.log 2 / Real.log b < Real.log 2 / Real.log c) →
  ¬(a < b ∧ b < c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_not_increasing_l233_23313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_approx_24_seconds_l233_23356

/-- The time taken for a boy to run around a square field -/
noncomputable def run_time (side_length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let perimeter := 4 * side_length
  let speed_ms := speed_kmh * (1000 / 3600)
  perimeter / speed_ms

/-- Theorem stating that the time taken to run around a square field
    with side length 20 meters at a speed of 12 km/h is approximately 24 seconds -/
theorem run_time_approx_24_seconds :
  ∃ ε > 0, |run_time 20 12 - 24| < ε := by
  sorry

/-- Compute an approximation of the run time -/
def run_time_approx (side_length : Float) (speed_kmh : Float) : Float :=
  let perimeter := 4 * side_length
  let speed_ms := speed_kmh * (1000 / 3600)
  perimeter / speed_ms

#eval run_time_approx 20 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_approx_24_seconds_l233_23356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_zero_l233_23337

-- Define the function to be minimized
noncomputable def f (a b : ℝ) : ℝ :=
  |Real.log ((a + b) / b) / Real.log a| + |Real.log ((b + a) / a) / Real.log b|

-- State the theorem
theorem min_value_is_zero :
  ∀ ε > 0, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ f a b < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_zero_l233_23337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l233_23398

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem monotonic_decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (StrictMonoOn f (Set.Ioo 0 (Real.exp (-1)))) = false ∧
  ∀ a b : ℝ, 0 < a ∧ a < b ∧ b < Real.exp (-1) → f b < f a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l233_23398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_midpoint_trajectory_l233_23328

noncomputable section

-- Define the circle C
def circle_C (φ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos φ, Real.sqrt 2 * (1 + Real.sin φ))

-- Define the line l
def line_l (a θ : ℝ) : ℝ := 1 / (Real.cos θ - 2 * a * Real.sin θ)

-- Define point Q as midpoint of OP
def point_Q (P : ℝ × ℝ) : ℝ × ℝ := (P.1 / 2, P.2 / 2)

theorem tangent_line_and_midpoint_trajectory :
  -- Part 1: When l is tangent to C, a = √2 / 8
  (∃ (φ : ℝ), let (x, y) := circle_C φ; x - 2 * (Real.sqrt 2 / 8) * y - 1 = 0) ∧
  -- Part 2: The polar equation of Q's trajectory is ρ₀ = √2 * sin θ
  (∀ (θ : ℝ), ∃ (φ : ℝ), let (x, y) := point_Q (circle_C φ);
    x^2 + y^2 = 2 * (Real.sin θ)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_midpoint_trajectory_l233_23328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_two_l233_23333

/-- An equilateral triangle with an inscribed circle and a regular hexagon inscribed in that circle -/
structure InscribedFigures where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Assumption that the side length is positive -/
  a_pos : a > 0

/-- The ratio of the area of the equilateral triangle to the area of the inscribed regular hexagon -/
noncomputable def areaRatio (fig : InscribedFigures) : ℝ :=
  let triangleArea := (fig.a^2 * Real.sqrt 3) / 4
  let hexagonArea := (fig.a^2 * Real.sqrt 3) / 8
  triangleArea / hexagonArea

/-- Theorem stating that the ratio of the areas is 2 -/
theorem area_ratio_is_two (fig : InscribedFigures) : areaRatio fig = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_two_l233_23333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l233_23321

/- Arithmetic sequence {a_n} -/
def a : ℕ → ℝ := sorry

/- Geometric sequence {b_n} -/
def b : ℕ → ℝ := sorry

/- Sum of first n terms of arithmetic sequence {a_n} -/
def S : ℕ → ℝ := sorry

/- Common ratio of geometric sequence {b_n} -/
def q : ℝ := sorry

/- Sum of first n terms of sequence {a₂ₙb₂ₙ₋₁} -/
def T : ℕ → ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- {a_n} is arithmetic
  (∀ n : ℕ, b (n + 1) / b n = q) →  -- {b_n} is geometric
  b 1 = 2 →  -- First term of {b_n} is 2
  q > 0 →  -- Common ratio of {b_n} is positive
  b 2 + b 3 = 12 →  -- Given condition
  b 3 = a 4 - 2 * a 1 →  -- Given condition
  S 11 = 11 * b 4 →  -- Given condition
  (∀ n : ℕ, a n = 3 * n - 2) ∧  -- General term formula for {a_n}
  (∀ n : ℕ, b n = 2^n) ∧  -- General term formula for {b_n}
  (∀ n : ℕ, T n = ((3 * n - 2) / 3) * 4^(n + 1) + 8 / 3) -- Sum formula for {a₂ₙb₂ₙ₋₁}
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l233_23321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l233_23340

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_problem (a b : ArithmeticSequence) (n : ℕ) :
  (∀ k, sum_n_terms a k / sum_n_terms b k = (7 * k + 45) / (k + 3)) →
  (∃ m : ℤ, a.a n = m * b.a (2 * n)) →
  n = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l233_23340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l233_23399

/-- The angle of inclination for a line given by its equation -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b) * (180 / Real.pi)

/-- Theorem: The angle of inclination for the line x + √3y + 5 = 0 is 150° -/
theorem line_inclination :
  angle_of_inclination 1 (Real.sqrt 3) 5 = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l233_23399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_three_digit_numbers_count_is_90_l233_23338

def even_three_digit_numbers_count : Nat :=
  let digits : List Nat := [1, 2, 3, 4, 5, 6]
  let valid_hundreds : List Nat := digits.filter (· < 6)
  let valid_tens : List Nat := digits
  let valid_units : List Nat := digits.filter (·.mod 2 = 0)
  valid_hundreds.length * valid_tens.length * valid_units.length

#eval even_three_digit_numbers_count -- Should output 90

theorem even_three_digit_numbers_count_is_90 :
  even_three_digit_numbers_count = 90 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_three_digit_numbers_count_is_90_l233_23338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_l233_23363

theorem gcd_power_minus_one (m n a : ℕ) (ha : a ≥ 2) :
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd m n) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_l233_23363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l233_23379

-- Define a parabola structure
structure Parabola where
  p : ℝ
  q : ℝ

-- Define the function representing the parabola
noncomputable def parabola_function (par : Parabola) (x : ℝ) : ℝ :=
  x^2 + par.p * x + par.q

-- Define the vertex of a parabola
noncomputable def vertex (par : Parabola) : ℝ × ℝ :=
  (-par.p / 2, par.q - par.p^2 / 4)

-- Theorem statement
theorem parabola_properties (par1 par2 : Parabola) :
  -- Both parabolas open upwards
  (∀ x : ℝ, parabola_function par1 x ≥ parabola_function par1 (-(par1.p / 2))) ∧
  (∀ x : ℝ, parabola_function par2 x ≥ parabola_function par2 (-(par2.p / 2))) ∧
  -- Both parabolas have the same shape (second derivative is constant)
  (∀ x : ℝ, (deriv (deriv (parabola_function par1))) x = (deriv (deriv (parabola_function par2))) x) ∧
  -- The vertices of the parabolas are as defined
  vertex par1 = (-par1.p / 2, par1.q - par1.p^2 / 4) ∧
  vertex par2 = (-par2.p / 2, par2.q - par2.p^2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l233_23379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l233_23396

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - m*x + 3*y + 3 = 0

-- Define the line equation
def line_eq (x y m : ℝ) : Prop := m*x + y - m = 0

-- Define symmetry condition
def is_symmetric (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y m ∧ line_eq (m/2) (-3/2) m

-- Theorem statement
theorem circle_symmetry (m : ℝ) :
  is_symmetric m → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l233_23396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l233_23323

open Real MeasureTheory

-- Define the polar curve
noncomputable def r (θ : ℝ) : ℝ := 2 + cos (2 * θ)

-- Define the region of integration
def region_bounds : Set ℝ := {θ | π/4 < θ ∧ θ < 3*π/4}

-- Define the area function for a polar curve
noncomputable def polar_area (f : ℝ → ℝ) (S : Set ℝ) : ℝ :=
  ∫ θ in S, (1/2) * (f θ)^2

-- Theorem statement
theorem area_of_region (h : region_bounds.Nonempty) : 
  4 * polar_area r region_bounds = 9*π/2 - 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l233_23323
