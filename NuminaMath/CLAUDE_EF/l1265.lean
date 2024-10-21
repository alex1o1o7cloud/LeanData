import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1265_126590

/-- A piecewise function f(x) defined as:
    f(x) = (a+3)x-5 for x ≤ 1
    f(x) = 2a/x for x > 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a + 3) * x - 5 else 2 * a / x

/-- The theorem stating that if f(x) is increasing on ℝ,
    then a is in the interval [-2, 0) -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (-2) 0 ∧ a ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1265_126590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l1265_126541

/-- Given a circle C1 with equation (x+2)^2 + (y-1)^2 = 1, 
    prove that a circle C2 symmetric to C1 with respect to the origin 
    has the equation (x-2)^2 + (y+1)^2 = 1 -/
theorem symmetric_circle_equation :
  ∃ (C1 C2 : Set (ℝ × ℝ)),
    C1 = {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 - 1)^2 = 1} ∧
    C2 = {p : ℝ × ℝ | ∀ q ∈ C1, (-q.1, -q.2) = p} →
    C2 = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 1} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l1265_126541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l1265_126583

noncomputable def annual_fixed_cost : ℝ := 60

noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 12 then 0.5 * x^2 + 4 * x else 11 * x + 100 / x - 39

noncomputable def selling_price : ℝ := 10

noncomputable def annual_profit (x : ℝ) : ℝ :=
  selling_price * x - annual_fixed_cost - variable_cost x

theorem max_annual_profit :
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → annual_profit y ≤ annual_profit x) ∧
    x = 12 ∧ annual_profit x = 38/3 := by
  sorry

#check max_annual_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l1265_126583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterized_line_segment_sum_of_squares_l1265_126576

/-- A line segment parameterized by t -/
structure ParametricLine where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The theorem statement -/
theorem parameterized_line_segment_sum_of_squares :
  ∀ (l : ParametricLine),
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
      (l.a * t + l.b, l.c * t + l.d) ∈ Set.Icc (-3, 8) (4, 10)) →
    (l.a * 0 + l.b, l.c * 0 + l.d) = (-3, 8) →
    l.a^2 + l.b^2 + l.c^2 + l.d^2 = 126 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterized_line_segment_sum_of_squares_l1265_126576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_10_l1265_126545

/-- An arithmetic sequence with common difference 1 where a₁, a₃, and a₆ form a geometric sequence -/
def special_sequence (a : ℕ → ℚ) : Prop :=
  (∀ n, a (n + 1) = a n + 1) ∧ 
  (a 3 * a 3 = a 1 * a 6)

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 1 + a n) / 2

/-- Theorem stating that the sum of the first 10 terms of the special sequence is 85 -/
theorem special_sequence_sum_10 (a : ℕ → ℚ) :
  special_sequence a → arithmetic_sum a 10 = 85 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_10_l1265_126545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_is_273_l1265_126500

/-- Calculates the maximum number of students that can be seated in an auditorium 
    with the given constraints. -/
def max_students : ℕ :=
  let num_rows : ℕ := 20
  let first_row_seats : ℕ := 15
  let seats_in_row (i : ℕ) : ℕ := first_row_seats + i - 1
  let students_in_row (i : ℕ) : ℕ := (seats_in_row i + 1) / 2
  (List.range num_rows).map (fun i => students_in_row (i + 1)) |>.sum

/-- Theorem stating that the maximum number of students that can be seated
    in the auditorium with the given constraints is 273. -/
theorem max_students_is_273 : max_students = 273 := by
  sorry

#eval max_students -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_is_273_l1265_126500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_intersection_l1265_126509

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^2 + a * x + b
noncomputable def g (c d x : ℝ) : ℝ := 3 * x^2 + c * x + d

noncomputable def vertex_x (a k : ℝ) : ℝ := -a / (2 * k)

theorem polynomial_intersection (a b c d : ℝ) :
  (g c d (vertex_x a 2) = 0) →
  (f a b (vertex_x c 3) = 0) →
  (f a b (vertex_x a 2) = g c d (vertex_x c 3)) →
  (f a b 50 = -200) →
  (g c d 50 = -200) →
  a + c = -720 :=
by sorry

#check polynomial_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_intersection_l1265_126509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1265_126530

/-- Parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Point where directrix intersects x-axis -/
def P : ℝ × ℝ := (-1, 0)

/-- Line passing through P with slope k -/
def Line (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 + 1)}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem statement -/
theorem parabola_intersection_length (k : ℝ) (A B : ℝ × ℝ) 
  (h_k : k > 0)
  (h_A : A ∈ Parabola ∩ Line k)
  (h_B : B ∈ Parabola ∩ Line k)
  (h_AB : A ≠ B)
  (h_dist : distance F B = 2 * distance F A) :
  distance A B = Real.sqrt 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1265_126530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_inequality_l1265_126593

theorem max_m_inequality (m : ℕ) : 
  (∃ a : ℝ, ∀ x : ℝ, x ∈ Set.Icc (1 : ℝ) (m : ℝ) → |x + a| ≤ Real.log x + 1) ↔ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_inequality_l1265_126593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1265_126526

/-- The time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) (platform_length_m : ℝ) : ℝ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance_m / train_speed_ms

/-- Theorem stating that the time taken for the train to cross the platform is approximately 7.49 seconds -/
theorem train_crossing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_crossing_time 132 110 165 - 7.49| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1265_126526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1265_126513

theorem equation_solution :
  ∃ y : ℝ, (1/2 : ℝ)^(4*y+10) = (8 : ℝ)^(2*y+3) ↔ y = -1.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1265_126513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angle_l1265_126520

/-- An isosceles triangle with a vertex angle of 50 degrees has base angles of 65 degrees each. -/
theorem isosceles_triangle_base_angle (α β γ : ℝ) 
  (h1 : α = β) -- isosceles condition
  (h2 : γ = 50) -- vertex angle is 50 degrees
  (h3 : α + β + γ = 180) : -- sum of angles in a triangle
  α = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angle_l1265_126520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_GAC_l1265_126549

/-- A rectangular prism with vertices ABCDEFGH -/
structure RectangularPrism where
  /-- The width of the prism -/
  AB : ℝ
  /-- The depth of the prism -/
  AD : ℝ
  /-- The height of the prism -/
  AE : ℝ

/-- The angle GAC in a rectangular prism -/
noncomputable def angle_GAC (prism : RectangularPrism) : ℝ :=
  Real.arcsin (Real.sqrt (prism.AB ^ 2 + prism.AD ^ 2) / 
    Real.sqrt (prism.AB ^ 2 + prism.AD ^ 2 + prism.AE ^ 2))

/-- Theorem: In a rectangular prism ABCDEFGH with AB = 2, AD = 3, and AE = 4, sin ∠GAC = √377 / 29 -/
theorem sin_angle_GAC (prism : RectangularPrism) 
    (h_AB : prism.AB = 2)
    (h_AD : prism.AD = 3)
    (h_AE : prism.AE = 4) : 
  Real.sin (angle_GAC prism) = Real.sqrt 377 / 29 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_GAC_l1265_126549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_l1265_126584

/-- The number of distinct natural number factors of 3^4 * 5^3 * 7^2 -/
def num_factors : ℕ := 60

/-- The given number -/
def given_number : ℕ := 3^4 * 5^3 * 7^2

theorem count_factors :
  (Finset.filter (· ∣ given_number) (Finset.range (given_number + 1))).card = num_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_l1265_126584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_arithmetic_progression_l1265_126522

/-- The sequence formed by discarding the last two digits of (1000 + n)^2 -/
def my_sequence (n : ℕ) : ℕ := ((1000 + n)^2) / 100

/-- The difference between consecutive terms in the sequence -/
def difference (n : ℕ) : ℤ := (my_sequence (n + 1) : ℤ) - (my_sequence n : ℤ)

/-- The property of being an arithmetic progression for the first k terms -/
def is_arithmetic_progression_up_to (k : ℕ) : Prop :=
  ∃ d : ℤ, ∀ n < k, difference n = d

theorem sequence_arithmetic_progression :
  is_arithmetic_progression_up_to 10 ∧
  ¬ is_arithmetic_progression_up_to 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_arithmetic_progression_l1265_126522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_occurs_prob_at_least_one_occurs_complement_l1265_126517

/-- The probability that at least one of two independent events occurs -/
theorem prob_at_least_one_occurs (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  1 - (1 - p₁) * (1 - p₂) = p₁ + p₂ - p₁ * p₂ := by
  sorry

/-- The probability that at least one of two independent events occurs is equal to
    one minus the probability that neither event occurs -/
theorem prob_at_least_one_occurs_complement (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  1 - (1 - p₁) * (1 - p₂) = 1 - (1 - p₁) * (1 - p₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_occurs_prob_at_least_one_occurs_complement_l1265_126517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_growth_l1265_126553

/-- S_n is the number of sequences (a_1, a_2, ..., a_n) where a_i ∈ {0,1} 
    and no six consecutive blocks are equal -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: S_n grows at least exponentially with base 3/2 -/
theorem S_growth (n : ℕ) (h : n ≥ 1) : S n ≥ (3/2)^n := by
  sorry

#check S
#check S_growth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_growth_l1265_126553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_max_value_of_m_l1265_126546

-- Define constants
noncomputable def e : ℝ := Real.exp 1

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - a / 2

-- Part I: Range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) ↔ a ∈ Set.Icc 0 (Real.sqrt e) :=
by sorry

-- Part II: Maximum value of m
theorem max_value_of_m :
  ∃ m : ℝ, m > 2.3 ∧ (∀ x : ℝ, x > 0 → Real.exp x ≥ Real.log x + m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_max_value_of_m_l1265_126546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_of_given_line_l1265_126514

/-- The angle of inclination of a line with slope m in radians -/
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

/-- The equation of the line sqrt(3)x - y - 3 = 0 -/
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 3 = 0

theorem angle_of_inclination_of_given_line :
  angle_of_inclination (Real.sqrt 3) = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_of_given_line_l1265_126514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_approx_l1265_126569

/-- Calculates the profit percentage for a set of books sold with discount and sales tax. -/
noncomputable def profit_percentage (cost_A cost_B cost_C sell_A sell_B sell_C discount_rate tax_rate : ℝ) : ℝ :=
  let discounted_A := sell_A * (1 - discount_rate)
  let discounted_B := sell_B * (1 - discount_rate)
  let discounted_C := sell_C * (1 - discount_rate)
  let final_A := discounted_A * (1 + tax_rate)
  let final_B := discounted_B * (1 + tax_rate)
  let final_C := discounted_C * (1 + tax_rate)
  let total_cost := cost_A + cost_B + cost_C
  let total_revenue := final_A + final_B + final_C
  let total_profit := total_revenue - total_cost
  (total_profit / total_cost) * 100

/-- The profit percentage for the given book prices, discount, and tax is approximately 17.6%. -/
theorem book_profit_percentage_approx :
  ∃ ε > 0, |profit_percentage 60 45 30 75 54 39 0.1 0.05 - 17.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_approx_l1265_126569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_in_terms_of_f_l1265_126562

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2*(x - 2)
  else 0  -- undefined outside [-3, 3]

-- Define the function h
noncomputable def h (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -4 - 2*x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2*x - 2
  else 0  -- undefined outside [-3, 3]

-- Theorem statement
theorem h_in_terms_of_f : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
  h x = if -3 ≤ x ∧ x ≤ 0 then 2*(f x) - 2
        else if 0 < x ∧ x ≤ 2 then f x
        else if 2 < x ∧ x ≤ 3 then f x + 2
        else 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_in_terms_of_f_l1265_126562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_approx_l1265_126568

/-- The radius of a cylinder given its height and lateral surface area -/
noncomputable def cylinder_radius (h : ℝ) (A : ℝ) : ℝ :=
  A / (2 * Real.pi * h)

/-- Theorem stating that a cylinder with height 21 m and lateral surface area 1583.3626974092558 m² has a radius of approximately 12 m -/
theorem cylinder_radius_approx :
  let h : ℝ := 21
  let A : ℝ := 1583.3626974092558
  abs (cylinder_radius h A - 12) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_approx_l1265_126568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_fourth_sufficient_not_necessary_l1265_126540

theorem alpha_pi_fourth_sufficient_not_necessary :
  (∀ α : ℝ, α = π / 4 → Real.cos (2 * α) = 0) ∧
  (∃ α : ℝ, α ≠ π / 4 ∧ Real.cos (2 * α) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_fourth_sufficient_not_necessary_l1265_126540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kanul_initial_amount_l1265_126599

theorem kanul_initial_amount 
  (raw_materials_percent : ℝ) 
  (machinery_percent : ℝ) 
  (wages_percent : ℝ) 
  (deposit_percent : ℝ) 
  (remaining_cash_percent : ℝ) 
  (loan_amount : ℝ) 
  (loan_percent : ℝ) :
  raw_materials_percent = 0.15 →
  machinery_percent = 0.10 →
  wages_percent = 0.25 →
  deposit_percent = 0.35 →
  remaining_cash_percent = 0.15 →
  loan_amount = 2000 →
  loan_percent = 0.20 →
  raw_materials_percent + machinery_percent + wages_percent + deposit_percent + remaining_cash_percent = 1 →
  ∃ initial_amount : ℝ, 
    initial_amount = (loan_amount / loan_percent - loan_amount) / remaining_cash_percent ∧
    abs (initial_amount - 53333.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kanul_initial_amount_l1265_126599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_is_40_l1265_126577

/-- Calculates the speed of the slower train given the lengths of two trains,
    their crossing time, and the speed of the faster train. -/
noncomputable def slower_train_speed (length1 length2 : ℝ) (crossing_time : ℝ) (faster_speed : ℝ) : ℝ :=
  let total_length := length1 + length2
  let relative_speed := total_length / crossing_time
  let relative_speed_kmh := relative_speed * 3.6
  relative_speed_kmh - faster_speed

/-- Theorem stating that under the given conditions, the speed of the slower train is 40 km/h. -/
theorem slower_train_speed_is_40 :
  slower_train_speed 190 160 12.59899208063355 60 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_is_40_l1265_126577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1265_126543

noncomputable def x : ℝ := Real.arctan 2

theorem trigonometric_identities :
  (Real.tan x = 2) →
  ((2/3) * (Real.sin x)^2 + (1/4) * (Real.cos x)^2 = 7/12) ∧
  (2 * (Real.sin x)^2 - Real.sin x * Real.cos x + (Real.cos x)^2 = 7/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1265_126543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_legs_exceed_twice_heads_l1265_126592

/-- Represents a group of dogs and people -/
structure MyGroup where
  dogs : ℕ
  people : ℕ

/-- The number of heads in the group -/
def heads (g : MyGroup) : ℕ := g.dogs + g.people

/-- The number of legs in the group -/
def legs (g : MyGroup) : ℕ := 4 * g.dogs + 2 * g.people

/-- Theorem stating the relationship between legs and heads in the group -/
theorem legs_exceed_twice_heads (g : MyGroup) (h : g.dogs = 14) :
  legs g = 2 * heads g + 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_legs_exceed_twice_heads_l1265_126592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_problem_l1265_126548

noncomputable def square_perimeter_to_side (p : ℝ) : ℝ := p / 4

noncomputable def square_side_to_area (s : ℝ) : ℝ := s * s

noncomputable def square_area_to_perimeter (a : ℝ) : ℝ := 4 * Real.sqrt a

theorem square_perimeter_problem (p1 p3 : ℝ) (h1 : p1 = 40) (h3 : p3 = 24) :
  let s1 := square_perimeter_to_side p1
  let s3 := square_perimeter_to_side p3
  let a1 := square_side_to_area s1
  let a3 := square_side_to_area s3
  let a2 := a1 + a3
  let p2 := square_area_to_perimeter a2
  abs (p2 - 46.64) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_problem_l1265_126548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1265_126595

-- Define the function f(x)
noncomputable def f (x : ℝ) := Real.sqrt (x - 5) + Real.sqrt (24 - 3*x)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 5 8 ∧ 
  (∀ x ∈ Set.Icc 5 8, f x ≤ f c) ∧
  f c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1265_126595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_high_trend_permutations_l1265_126563

/-- A high-trend permutation of (1,2,3,4,5,6) -/
def HighTrendPermutation : Type :=
  { p : Fin 6 → Fin 6 //
    Function.Bijective p ∧
    p 2 = 3 ∧
    p 3 = 4 ∧
    p 0 + p 1 < p 4 + p 5 }

instance : Fintype HighTrendPermutation := by
  sorry

/-- The number of high-trend permutations is 4 -/
theorem num_high_trend_permutations :
  Fintype.card HighTrendPermutation = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_high_trend_permutations_l1265_126563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_partitions_eq_11_l1265_126501

/-- A partition of n into at most k parts is a list of at most k non-negative integers
    that sum to n, arranged in non-increasing order. -/
def Partition (n : ℕ) (k : ℕ) : Type :=
  { l : List ℕ // l.sum = n ∧ l.length ≤ k ∧ l.Sorted (· ≥ ·) }

/-- The number of partitions of 6 into at most 5 parts -/
def num_partitions : ℕ := sorry

theorem num_partitions_eq_11 : num_partitions = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_partitions_eq_11_l1265_126501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_coordinates_l1265_126555

/-- Given points A(0,1) and B(-1,2) in a Cartesian coordinate system, 
    the vector AB has coordinates (-1, 1). -/
theorem vector_AB_coordinates : 
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (-1, 2)
  (B.1 - A.1, B.2 - A.2) = (-1, 1) := by
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_coordinates_l1265_126555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_air_conditioner_consumption_l1265_126536

/-- Represents the consumption of an air conditioner -/
structure AirConditioner where
  consumption_rate : ℝ
  runtime_hours : ℝ
  usage_days : ℕ

/-- Calculates the total consumption of an air conditioner -/
noncomputable def total_consumption (ac : AirConditioner) : ℝ :=
  (ac.consumption_rate / ac.runtime_hours) * (ac.usage_days : ℝ) * 6

/-- Theorem stating the total consumption of the air conditioner -/
theorem air_conditioner_consumption :
  ∀ (ac : AirConditioner),
  ac.consumption_rate = 7.2 ∧
  ac.runtime_hours = 8 ∧
  ac.usage_days = 5 →
  total_consumption ac = 27 := by
  intro ac h
  simp [total_consumption]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_air_conditioner_consumption_l1265_126536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_shuttle_speed_l1265_126551

/-- The speed of a space shuttle orbiting Earth in kilometers per hour -/
noncomputable def speed_km_per_hour : ℝ := 7200

/-- The number of seconds in an hour -/
noncomputable def seconds_per_hour : ℝ := 3600

/-- The speed of the space shuttle in kilometers per second -/
noncomputable def speed_km_per_second : ℝ := speed_km_per_hour / seconds_per_hour

theorem space_shuttle_speed : speed_km_per_second = 2 := by
  -- Unfold the definitions
  unfold speed_km_per_second speed_km_per_hour seconds_per_hour
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_shuttle_speed_l1265_126551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_inequalities_l1265_126527

def is_inequality (expr : String) : Bool :=
  expr.contains '≤' || expr.contains '≥' || expr.contains '≠' || expr.contains '<' || expr.contains '>'

def expressions : List String := [
  "x-y=2", "x≤y", "x+y", "x²-3y", "x≥0", "(1/2)x≠3"
]

theorem count_inequalities :
  (expressions.filter is_inequality).length = 3 := by
  simp [expressions, is_inequality]
  rfl

#eval expressions.filter is_inequality
#eval (expressions.filter is_inequality).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_inequalities_l1265_126527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parallel_theorem_l1265_126512

-- Define the parabolas C₁ and C₂
def C₁ : Set (ℝ × ℝ) := {p | ∃ a : ℝ, p.2 = a * p.1^2}
def C₂ : Set (ℝ × ℝ) := {p | ∃ b c d : ℝ, p.2 = b * (p.1 - c)^2 + d}

-- Define the property of parallel axes of symmetry
def parallel_axes (C₁ C₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Define the intersection points
def intersection_points (A₀ B₀ : ℝ × ℝ) (C₁ C₂ : Set (ℝ × ℝ)) : Prop :=
  A₀ ∈ C₁ ∧ A₀ ∈ C₂ ∧ B₀ ∈ C₁ ∧ B₀ ∈ C₂

-- Define the property of points being on the parabolas
def points_on_parabolas (n : ℕ) (A B : Fin (2*n+1) → ℝ × ℝ) (C₁ C₂ : Set (ℝ × ℝ)) : Prop :=
  (∀ i, A i ∈ C₁) ∧ (∀ i, B i ∈ C₂)

-- Define parallel line segments
def parallel_segments (n : ℕ) (A B : Fin (2*n+1) → ℝ × ℝ) : Prop :=
  ∀ i : Fin (2*n), (A (i+1) - A i).1 * (B (i+1) - B i).2 = (A (i+1) - A i).2 * (B (i+1) - B i).1

-- The main theorem
theorem parabola_parallel_theorem (n : ℕ) (A B : Fin (2*n+1) → ℝ × ℝ) :
  parallel_axes C₁ C₂ →
  intersection_points (A 0) (B 0) C₁ C₂ →
  points_on_parabolas n A B C₁ C₂ →
  parallel_segments n A B →
  (A (2*n) - A 0).1 * (B (2*n) - B 0).2 = (A (2*n) - A 0).2 * (B (2*n) - B 0).1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parallel_theorem_l1265_126512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_a_2013_l1265_126537

noncomputable def a : ℕ → ℝ
| 0 => 100
| n + 1 => a n + 1 / a n

theorem closest_integer_to_a_2013 :
  ∃ (m : ℤ), m = 118 ∧ ∀ (k : ℤ), k ≠ m → |k - (a 2012 : ℝ)| > |m - (a 2012 : ℝ)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_a_2013_l1265_126537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_irrational_l1265_126556

-- Define the coefficient sequence
noncomputable def c : ℕ → Bool := sorry

-- Define the power series f(x)
noncomputable def f (x : ℝ) : ℝ := ∑' (i : ℕ), (if c i then 1 else 0) * x^i

-- Define the condition that each coefficient is 0 or 1
def coeff_binary : Prop := ∀ i : ℕ, c i = true ∨ c i = false

-- State the theorem
theorem f_half_irrational
  (h_coeff : coeff_binary)
  (h_value : f (2/3) = 3/2) :
  Irrational (f (1/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_irrational_l1265_126556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_eight_l1265_126557

noncomputable section

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4/3 * x^3 + 2*x^2 - 3*x - 1

-- Define the extreme value point a
noncomputable def a : ℝ := 1/2

-- Assume a is an extreme value point of g
axiom a_is_extreme : ∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (a - ε) (a + ε), g x ≤ g a

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then a^x else Real.log x / Real.log a

-- State the theorem
theorem f_sum_equals_eight : f (1/4) + f (Real.log (1/6) / Real.log 2) = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_eight_l1265_126557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_proof_no_point_m_existence_l1265_126575

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

noncomputable def perimeter_triangle (p q f : Point) : ℝ :=
  sorry  -- Definition of triangle perimeter

def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem ellipse_eccentricity_proof (e : Ellipse) (f₁ f₂ p q : Point) (l : Line) :
  on_ellipse f₁ e → on_ellipse f₂ e →  -- F₁ and F₂ are on the ellipse
  on_line f₂ l → on_line p l → on_line q l →  -- F₂, P, Q are on line l
  on_ellipse p e → on_ellipse q e →  -- P and Q are on the ellipse
  perimeter_triangle p q f₁ = 2 * Real.sqrt 3 * (2 * e.b) →  -- Perimeter condition
  eccentricity e = Real.sqrt 6 / 3 :=
by sorry

theorem no_point_m_existence (e : Ellipse) (o p q : Point) (l : Line) :
  on_ellipse p e → on_ellipse q e →  -- P and Q are on the ellipse
  on_line p l → on_line q l →  -- P and Q are on line l
  l.slope = 1 →  -- Slope of l is 1
  ¬∃ m : Point, on_ellipse m e ∧ m.x = 2 * p.x + q.x ∧ m.y = 2 * p.y + q.y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_proof_no_point_m_existence_l1265_126575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_choice_maximizes_probability_l1265_126542

/-- The optimal choice for player C to maximize winning probability -/
noncomputable def optimal_choice : ℝ := 13/24

/-- Player A's choice interval -/
def A_interval : Set ℝ := Set.Icc 0 1

/-- Player B's choice interval -/
def B_interval : Set ℝ := Set.Icc (1/2) (2/3)

/-- Winning probability function for player C -/
noncomputable def winning_probability (x : ℝ) : ℝ :=
  x * (2/3 - x) + (x - 1/2) * (1 - x)

theorem optimal_choice_maximizes_probability :
  ∀ x ∈ A_interval, winning_probability optimal_choice ≥ winning_probability x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_choice_maximizes_probability_l1265_126542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l1265_126529

/-- The volume of a regular triangular pyramid with height h and lateral face area Q -/
noncomputable def pyramidVolume (h Q : ℝ) : ℝ :=
  (Real.sqrt 3 / 72) * h * (-4 * h^2 + Real.sqrt (16 * h^4 + 192 * Q^2))

/-- Theorem: The volume of a regular triangular pyramid with height h and lateral face area Q
    is equal to (√3/72) * h * (-4h² + √(16h⁴ + 192Q²)) and is positive -/
theorem regular_triangular_pyramid_volume (h Q : ℝ) (h_pos : h > 0) (Q_pos : Q > 0) :
  ∃ V : ℝ, V = pyramidVolume h Q ∧ V > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l1265_126529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l1265_126511

theorem base_number_proof (x : ℝ) (k : ℕ) 
  (h1 : x ^ k = 2) 
  (h2 : x ^ (4 * k + 2) = 784) : 
  x = 7 ∨ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l1265_126511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l1265_126561

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_first_term
  (a d : ℝ)
  (h1 : arithmeticSum a d 30 = 150)
  (h2 : arithmeticSum (a + 30 * d) d 70 = 4900) :
  a = -13.85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l1265_126561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1265_126521

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  tangentToXAxis : Bool

/-- Calculates the length of the major axis of an ellipse -/
noncomputable def majorAxisLength (e : Ellipse) : ℝ :=
  distance e.focus1 e.focus2

theorem ellipse_major_axis_length 
  (e : Ellipse) 
  (h1 : e.focus1 = ⟨5, 10⟩) 
  (h2 : e.focus2 = ⟨35, 30⟩) 
  (h3 : e.tangentToXAxis = true) : 
  ∃ (a : ℝ), a = 5 * Real.sqrt 13 ∧ 2 * a = majorAxisLength e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1265_126521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_approximation_equilibrium_price_condition_l1265_126581

noncomputable section

-- Define the price range
def price_range (x : ℝ) : Prop := 1 < x ∧ x < 14

-- Define the supply function
noncomputable def supply (a x : ℝ) : ℝ := a * x + (7/2) * a^2 - a

-- Define the demand function
noncomputable def demand (x : ℝ) : ℝ := -(1/224) * x^2 - (1/112) * x + 1

-- Define the quantity sold
noncomputable def quantity_sold (a x : ℝ) : ℝ :=
  if demand x > supply a x then supply a x else demand x

-- Define the monthly sales revenue
noncomputable def monthly_revenue (a x : ℝ) : ℝ := quantity_sold a x * x

-- Statement for part (1)
theorem revenue_approximation (a x : ℝ) :
  a = 1/7 → x = 7 → price_range x → ∃ ε > 0, |monthly_revenue a x - 50313| < ε :=
by sorry

-- Statement for part (2)
theorem equilibrium_price_condition (a : ℝ) :
  (∀ x, price_range x → supply a x = demand x → x ≥ 6) ↔ (0 < a ∧ a ≤ 1/7) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_approximation_equilibrium_price_condition_l1265_126581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_lattice_point_l1265_126504

open Real Set

/-- Unit circle centered at the origin -/
def unitCircle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Intersection of ray OQ with the unit circle -/
noncomputable def rayIntersection (q : ℝ × ℝ) : ℝ × ℝ :=
  if q = (0, 0) then (0, 0)
  else (q.1 / Real.sqrt (q.1^2 + q.2^2), q.2 / Real.sqrt (q.1^2 + q.2^2))

/-- Main theorem -/
theorem exists_close_lattice_point (p : ℝ × ℝ) (k : ℕ) (h : p ∈ unitCircle) :
  ∃ q : ℤ × ℤ, (|q.1| = k ∨ |q.2| = k) ∧
    dist p (rayIntersection (↑q.1, ↑q.2)) < 1 / (2 * ↑k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_lattice_point_l1265_126504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l1265_126519

/-- The line y = ax + 1 intersects the curve x² + y² + bx - y = 1 at two points,
    and these two points are symmetrical with respect to the line x + y = 0. -/
def symmetric_intersection (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (y₁ = a * x₁ + 1) ∧
    (y₂ = a * x₂ + 1) ∧
    (x₁^2 + y₁^2 + b * x₁ - y₁ = 1) ∧
    (x₂^2 + y₂^2 + b * x₂ - y₂ = 1) ∧
    (x₁ + y₁ = -(x₂ + y₂))

/-- Given the symmetric intersection condition, prove that a + b = 2 -/
theorem sum_of_a_and_b (a b : ℝ) (h : symmetric_intersection a b) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l1265_126519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l1265_126591

/-- Two circles of radius r touch each other and are externally tangent to a third circle of radius R.
    Given AB = 12 and R = 8, this theorem proves that r = 24. -/
theorem circle_radius_problem (R AB : ℝ) (hR : R = 8) (hAB : AB = 12) : 
  ∃ r : ℝ, r = 24 ∧ 
    (R / (R + r) = AB / (2 * r)) := by
  -- We'll use r = 24 as our solution
  let r := 24
  
  -- Show that this r satisfies the equation
  have h_eq : R / (R + r) = AB / (2 * r) := by
    rw [hR, hAB] -- Replace R with 8 and AB with 12
    norm_num -- Simplify numerical expressions
  
  -- Prove existence
  use r
  constructor
  · rfl -- r = 24 by definition
  · exact h_eq

-- The main result
#check circle_radius_problem


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l1265_126591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_between_21_and_22_l1265_126588

structure Triangle where
  A : ℝ × ℝ
  D : ℝ × ℝ
  B : ℝ × ℝ

def isIsoscelesRight (t : Triangle) : Prop :=
  (t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2 = (t.D.1 - t.D.1)^2 + (t.D.2 - t.D.2)^2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem sum_of_distances_between_21_and_22 (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : t.A = (20, 0)) 
  (h3 : t.D = (6, 6)) :
  21 < distance t.A t.D + distance t.B t.D ∧ 
  distance t.A t.D + distance t.B t.D < 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_between_21_and_22_l1265_126588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_linear_transform_l1265_126560

-- Define a sample type
def Sample := List ℝ

-- Define the variance function
noncomputable def variance (s : Sample) : ℝ := sorry

-- Define variables for our sample elements
variable (a₁ a₂ a₃ : ℝ)

-- Define our specific samples
def originalSample : Sample := [a₁, a₂, a₃]
def transformedSample (a₁ a₂ a₃ : ℝ) : Sample := 
  [3 * a₁ + 1, 3 * a₂ + 1, 3 * a₃ + 1]

-- State the theorem
theorem variance_linear_transform (a : ℝ) :
  (variance (originalSample a₁ a₂ a₃) = a) →
  (variance (transformedSample a₁ a₂ a₃) = 9 * a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_linear_transform_l1265_126560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1265_126515

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define the midpoint of chord AB
def chord_midpoint : ℝ × ℝ := (3, 1)

-- Define the equation of line AB
def line_AB_equation (x y : ℝ) : Prop := x + y - 4 = 0

-- Theorem statement
theorem chord_equation :
  ∀ x y : ℝ,
  circle_equation x y →
  line_AB_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1265_126515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_water_calculation_l1265_126579

/-- The amount of water in the pool after Timmy, Tommy, Tina, and Trudy fill it -/
def pool_water (tina_pail : ℝ) (tommy_pail : ℝ) (timmy_pail : ℝ) (trudy_pail : ℝ) 
  (timmy_trips : ℕ) (trudy_trips : ℕ) (tommy_trips : ℕ) (tina_trips : ℕ) : ℝ :=
  timmy_pail * timmy_trips + trudy_pail * trudy_trips + 
  tommy_pail * tommy_trips + tina_pail * tina_trips

theorem pool_water_calculation : 
  ∀ (tina_pail : ℝ) (tommy_pail : ℝ) (timmy_pail : ℝ) (trudy_pail : ℝ),
  tina_pail = 4 →
  tommy_pail = tina_pail + 2 →
  timmy_pail = 2 * tommy_pail →
  trudy_pail = 1.5 * timmy_pail →
  pool_water tina_pail tommy_pail timmy_pail trudy_pail 4 4 6 6 = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_water_calculation_l1265_126579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_find_k_equal_distances_l1265_126516

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := k * x - y + 1 - 2 * k = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (2, -1)

-- Define the intersection points with axes
noncomputable def intersection_x (k : ℝ) : ℝ := (1 + 2 * k) / k
noncomputable def intersection_y (k : ℝ) : ℝ := -2 * k - 1

-- Theorem 1: The line passes through the fixed point for all k
theorem line_passes_through_fixed_point (k : ℝ) :
  line_equation k (fixed_point.1) (fixed_point.2) := by
  sorry

-- Theorem 2: The value of k that satisfies |OA| = |OB| is -1
theorem find_k_equal_distances :
  ∃ k : ℝ, k < 0 ∧ intersection_x k > 0 ∧ intersection_y k > 0 ∧
  intersection_x k = intersection_y k ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_find_k_equal_distances_l1265_126516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1265_126587

noncomputable def cubic_equation (t : ℝ) (s : ℂ) (x : ℂ) : Prop :=
  x^3 + t * x + s = 0

def roots_form_equilateral_triangle (roots : Finset ℂ) : Prop :=
  roots.card = 3 ∧
  ∀ z₁ z₂, z₁ ∈ roots → z₂ ∈ roots → z₁ ≠ z₂ → Complex.abs (z₁ - z₂) = Real.sqrt 3

theorem cubic_equation_roots (t : ℝ) (s : ℂ) :
  (∃ roots : Finset ℂ, (∀ x, x ∈ roots → cubic_equation t s x) ∧
    roots_form_equilateral_triangle roots) ∧
  Complex.arg s = π / 6 →
  t = 0 ∧ s = Complex.mk (Real.sqrt 3 / 2) (1 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1265_126587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_given_equal_perimeter_triangle_l1265_126572

noncomputable def triangle_side (A : ℝ) : ℝ := Real.sqrt ((4 * A) / Real.sqrt 3)

noncomputable def hexagon_side (s : ℝ) : ℝ := s / 2

noncomputable def hexagon_area (t : ℝ) : ℝ := (3 * Real.sqrt 3 * t^2) / 2

theorem hexagon_area_given_equal_perimeter_triangle (A : ℝ) (h : A = 9) :
  hexagon_area (hexagon_side (triangle_side A)) = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_given_equal_perimeter_triangle_l1265_126572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vectors_x_value_l1265_126518

/-- Two vectors in ℝ² -/
def a (x : ℝ) : ℝ × ℝ := (x, 1)
def b (x : ℝ) : ℝ × ℝ := (4, x)

/-- Opposite direction condition -/
def opposite_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ v = k • w

/-- Main theorem -/
theorem opposite_vectors_x_value :
  ∀ x : ℝ, opposite_direction (a x) (b x) → x = -2 :=
by
  intro x h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vectors_x_value_l1265_126518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fedya_arrived_third_l1265_126528

/-- Represents the order of arrival of people at the meeting -/
def ArrivalOrder := Fin 5 → String

/-- The set of people who attended the meeting -/
def Attendees : Finset String := {"Roman", "Fedya", "Liza", "Katya", "Andrey"}

/-- Predicate to check if a person arrived before another -/
def ArrivedBefore (order : ArrivalOrder) (person1 person2 : String) : Prop :=
  ∃ (i j : Fin 5), order i = person1 ∧ order j = person2 ∧ i < j

/-- Predicate to check if a person arrived immediately before another -/
def ArrivedImmediatelyBefore (order : ArrivalOrder) (person1 person2 : String) : Prop :=
  ∃ (i : Fin 5), order i = person1 ∧ order i.succ = person2

/-- Theorem stating that Fedya arrived third given the conditions -/
theorem fedya_arrived_third (order : ArrivalOrder) 
  (h1 : ArrivedBefore order "Liza" "Roman")
  (h2 : ArrivedImmediatelyBefore order "Katya" "Fedya")
  (h3 : ArrivedBefore order "Fedya" "Roman")
  (h4 : ArrivedBefore order "Katya" "Liza")
  (h5 : order 0 ≠ "Katya")
  (h6 : Function.Injective order)
  (h7 : ∀ x, x ∈ Attendees ↔ ∃ i, order i = x) :
  order 2 = "Fedya" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fedya_arrived_third_l1265_126528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l1265_126531

-- Define the circle C
noncomputable def circle_C (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

-- Define the line l
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi/3) = 3 * Real.sqrt 3

-- Define the ray
def ray (x y : ℝ) : Prop :=
  Real.sqrt 3 * x = y ∧ x ≥ 0

-- Define point P
noncomputable def point_P : ℝ × ℝ :=
  (1, Real.sqrt 3)

-- Define point Q
noncomputable def point_Q : ℝ × ℝ :=
  (3/2, 3*Real.sqrt 3/2)

theorem length_PQ_is_two :
  let (px, py) := point_P
  let (qx, qy) := point_Q
  Real.sqrt ((qx - px)^2 + (qy - py)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l1265_126531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_outside_unit_circle_l1265_126544

/-- A polynomial with positive increasing real coefficients -/
def IncreasingPolynomial (n : ℕ) : Type :=
  { f : Polynomial ℝ // 
    (∀ i, i ≤ n → f.coeff i > 0) ∧ 
    (∀ i j, i < j ∧ j ≤ n → f.coeff i < f.coeff j) }

/-- All roots of an increasing polynomial have modulus greater than 1 -/
theorem roots_outside_unit_circle (n : ℕ) (f : IncreasingPolynomial n) :
  ∀ r : ℂ, (f.val.map Complex.ofReal).eval r = 0 → Complex.abs r > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_outside_unit_circle_l1265_126544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_second_group_l1265_126523

theorem average_age_second_group (total_students : ℕ) (avg_age_all : ℝ) 
  (first_group_size : ℕ) (avg_age_first_group : ℝ) (last_student_age : ℝ) :
  total_students = 15 →
  avg_age_all = 15 →
  first_group_size = 7 →
  avg_age_first_group = 14 →
  last_student_age = 15 →
  (let second_group_size := total_students - first_group_size - 1
   let total_age := (total_students : ℝ) * avg_age_all
   let first_group_total_age := (first_group_size : ℝ) * avg_age_first_group
   let remaining_age := total_age - first_group_total_age - last_student_age
   let avg_age_second_group := remaining_age / (second_group_size : ℝ)
   avg_age_second_group = 16) := by
  intro h1 h2 h3 h4 h5
  sorry

#check average_age_second_group

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_second_group_l1265_126523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_exp_minus_linear_l1265_126525

/-- Given a function f(x) = e^x - ax - 1 that is monotonically increasing for all real x, 
    prove that a ≤ 0 -/
theorem monotone_exp_minus_linear (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => Real.exp x - a * x - 1)) →
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_exp_minus_linear_l1265_126525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_l1265_126594

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

noncomputable def g (A ω φ θ : ℝ) (x : ℝ) : ℝ := f A ω φ (x - θ)

theorem periodic_function_properties
  (A ω φ : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : abs φ < π / 2)
  (h_period : ∀ x, f A ω φ (x + π) = f A ω φ x)
  (h_max : f A ω φ (π / 6) = 2 ∧ ∀ x, f A ω φ x ≤ 2) :
  (∀ x, f A ω φ x = 2 * Real.sin (2 * x + π / 6)) ∧
  (∀ x, x ∈ Set.Icc (-π / 2) 0 → f A ω φ x ∈ Set.Icc (-2) 1) ∧
  (∀ θ, θ ∈ Set.Icc (π / 12) (π / 3) ↔
    (∀ x y, x < y → x ∈ Set.Icc 0 (π / 4) → y ∈ Set.Icc 0 (π / 4) →
      g A ω φ θ x < g A ω φ θ y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_l1265_126594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_AB_constant_l1265_126564

noncomputable section

/-- An ellipse with center at the origin, focus on the y-axis, major axis length 4, and eccentricity √2/2 -/
def Ellipse_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 / 4 = 1}

/-- A point P on the ellipse in the first quadrant with x-coordinate 1 -/
noncomputable def Point_P : ℝ × ℝ :=
  (1, Real.sqrt 2)

/-- The slope of a line through Point_P -/
def Slope (k : ℝ) : ℝ × ℝ → Prop :=
  fun p => p.2 - Point_P.2 = k * (p.1 - Point_P.1)

/-- Two points A and B on the ellipse, each on a line through P with slope k and -1/k respectively -/
noncomputable def Points_A_B (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x_A := (k^2 - 2 * Real.sqrt 2 * k - 2) / (2 + k^2)
  let y_A := Point_P.2 + k * (x_A - Point_P.1)
  let x_B := (k^2 + 2 * Real.sqrt 2 * k - 2) / (2 + k^2)
  let y_B := Point_P.2 - (1/k) * (x_B - Point_P.1)
  ((x_A, y_A), (x_B, y_B))

theorem slope_AB_constant (k : ℝ) (hk : k ≠ 0) :
  let (A, B) := Points_A_B k
  (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_AB_constant_l1265_126564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1265_126510

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

-- State the theorem
theorem range_of_f :
  (∀ y ∈ Set.Icc (-1 : ℝ) 2,
    ∃ x ∈ Set.Icc 0 Real.pi, f x = y) ∧
  (∀ x ∈ Set.Icc 0 Real.pi, -1 ≤ f x ∧ f x ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1265_126510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1265_126505

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_solution (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period (λ x ↦ f (2*x + 1)) 5)
  (h_value : f 1 = 5) :
  f 2009 + f 2010 = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1265_126505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1265_126554

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x + Real.pi / 4)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ (x : ℝ), f ω (x + Real.pi) = f ω x) 
  (h_smallest_period : ∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f ω (x + T) = f ω x) → T ≥ Real.pi) :
  (ω = 1) ∧ 
  (∀ (k : ℤ), StrictMonoOn (f ω) (Set.Icc (-3*Real.pi/8 + k*Real.pi) (Real.pi/8 + k*Real.pi))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1265_126554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1265_126507

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

theorem monotonic_decreasing_interval (k : ℤ) :
  StrictMonoOn (fun x => -f x) (Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1265_126507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_coloring_l1265_126558

/-- A coloring of an n × n grid using two colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a 2×2 subgrid at position (i, j) is monochromatic. -/
def isMonochromatic (c : Coloring n) (i j : Fin n) : Prop :=
  ∃ (color : Bool), ∀ (di dj : Fin 2), c ⟨(i + di) % n, by sorry⟩ ⟨(j + dj) % n, by sorry⟩ = color

/-- A valid coloring has no monochromatic 2×2 subgrids. -/
def isValidColoring (c : Coloring n) : Prop :=
  ∀ (i j : Fin n), ¬isMonochromatic c i j

/-- The existence of a valid coloring for a given n. -/
def existsValidColoring (n : ℕ) : Prop :=
  ∃ (c : Coloring n), isValidColoring c

theorem largest_valid_coloring : 
  (existsValidColoring 4) ∧ (¬existsValidColoring 5) := by
  sorry

#check largest_valid_coloring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_coloring_l1265_126558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1265_126532

theorem quartic_equation_solutions :
  let S : Set ℂ := {(-2 : ℂ), (-Complex.I * Real.sqrt 2 : ℂ), (Complex.I * Real.sqrt 2 : ℂ), (2 : ℂ)}
  ∀ z : ℂ, z^4 - 6*z^2 + 8 = 0 ↔ z ∈ S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1265_126532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_xy_value_l1265_126503

-- Define the equation
def satisfies_equation (x y : ℕ+) : Prop :=
  (1 : ℚ) / x.val + (1 : ℚ) / (3 * y.val) = (1 : ℚ) / 6

-- Define the theorem
theorem smallest_xy_value :
  ∃ (x y : ℕ+), satisfies_equation x y ∧
    (∀ (x' y' : ℕ+), satisfies_equation x' y' → x.val * y.val ≤ x'.val * y'.val) ∧
    x.val * y.val = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_xy_value_l1265_126503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_outliers_l1265_126552

def data_set : List ℝ := [10, 24, 35, 35, 35, 42, 42, 45, 58, 62]
def Q1 : ℝ := 35
def Q2 : ℝ := 38.5
def Q3 : ℝ := 45

def IQR : ℝ := Q3 - Q1

noncomputable def lower_threshold : ℝ := Q1 - 1.5 * IQR
noncomputable def upper_threshold : ℝ := Q3 + 1.5 * IQR

noncomputable def mean : ℝ := (data_set.sum) / (data_set.length)

noncomputable def std_dev : ℝ := Real.sqrt ((data_set.map (λ x => (x - mean) ^ 2)).sum / data_set.length)

noncomputable def is_outlier (x : ℝ) : Bool :=
  x < lower_threshold || x > upper_threshold || x < mean - 2 * std_dev || x > mean + 2 * std_dev

theorem number_of_outliers :
  (data_set.filter is_outlier).length = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_outliers_l1265_126552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_reduction_l1265_126534

-- Define the initial conditions
noncomputable def initial_volume : ℝ := 10
noncomputable def initial_concentration : ℝ := 0.20
noncomputable def final_volume : ℝ := 40

-- Define the function to calculate the final concentration
noncomputable def final_concentration (init_vol : ℝ) (init_conc : ℝ) (final_vol : ℝ) : ℝ :=
  (init_vol * init_conc) / final_vol

-- Define the function to calculate the percentage reduction
noncomputable def percentage_reduction (init_conc : ℝ) (final_conc : ℝ) : ℝ :=
  ((init_conc - final_conc) / init_conc) * 100

-- Theorem statement
theorem alcohol_concentration_reduction :
  percentage_reduction initial_concentration 
    (final_concentration initial_volume initial_concentration final_volume) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_reduction_l1265_126534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_2_a_values_for_max_value_1_l1265_126550

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 3

-- Part 1
theorem range_of_f_when_a_is_2 :
  ∃ (y_min y_max : ℝ), y_min = -21/4 ∧ y_max = 15 ∧
  ∀ x, x ∈ Set.Icc (-2 : ℝ) 3 → 
    y_min ≤ f 2 x ∧ f 2 x ≤ y_max ∧
    (∃ x₁ x₂, x₁ ∈ Set.Icc (-2 : ℝ) 3 ∧ x₂ ∈ Set.Icc (-2 : ℝ) 3 ∧ 
              f 2 x₁ = y_min ∧ f 2 x₂ = y_max) :=
by sorry

-- Part 2
theorem a_values_for_max_value_1 :
  ∃ (a₁ a₂ : ℝ), a₁ = -1/3 ∧ a₂ = -1 ∧
  ∀ a : ℝ, (∀ x, x ∈ Set.Icc (-1 : ℝ) 3 → f a x ≤ 1) ∧
           (∃ x, x ∈ Set.Icc (-1 : ℝ) 3 ∧ f a x = 1) →
           a = a₁ ∨ a = a₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_2_a_values_for_max_value_1_l1265_126550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_circle_l1265_126547

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop := λ x y => x^2/a^2 + y^2/b^2 = 1
  center_origin : True
  foci_on_x_axis : True
  eccentricity : c/a = 1/2
  point_on_ellipse : eq 1 (3/2)

/-- Line passing through left focus of the ellipse -/
structure Line (e : Ellipse) where
  k : ℝ
  eq : ℝ → ℝ → Prop := λ x y => y = k*(x + e.c)
  passes_through_focus : True

/-- Triangle formed by the line and the ellipse -/
structure Triangle (e : Ellipse) (l : Line e) where
  area : ℝ
  eq : area = 6*Real.sqrt 2/7

/-- Circle centered at origin -/
structure Circle where
  r : ℝ
  eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 = r^2
  center_origin : True

/-- Main theorem statement -/
theorem ellipse_line_circle 
  (e : Ellipse) 
  (l : Line e) 
  (t : Triangle e l) 
  (c : Circle) : 
  (c.eq = λ x y => x^2 + y^2 = 1/2) ∧ 
  (∃ (x y : ℝ), l.eq x y ∧ c.eq x y ∧ 
    ∀ (x' y' : ℝ), l.eq x' y' → c.eq x' y' → (x, y) = (x', y')) := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_circle_l1265_126547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_flags_is_60_l1265_126585

/-- Represents the set of available colors for the flag strips -/
inductive FlagColor
| Red
| White
| Blue
| Green
| Yellow

/-- A flag consists of three horizontal strips, each with a distinct color -/
structure FlagStripes where
  top : FlagColor
  middle : FlagColor
  bottom : FlagColor
  distinct : top ≠ middle ∧ top ≠ bottom ∧ middle ≠ bottom

/-- The number of distinct flags possible -/
def num_distinct_flags : ℕ := sorry

/-- Theorem stating that the number of distinct flags is 60 -/
theorem num_distinct_flags_is_60 : num_distinct_flags = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_flags_is_60_l1265_126585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_count_proof_l1265_126538

/-- The number of ducks in a pond satisfying specific conditions -/
def duck_count : ℕ :=
  let total_ducks : ℕ := 40
  let muscovy_ratio : ℚ := 1/2
  let female_muscovy_ratio : ℚ := 3/10
  let female_muscovy_count : ℕ := 6
  total_ducks

theorem duck_count_proof : 
  (1/2 : ℚ) * (3/10 : ℚ) * (duck_count : ℚ) = 6 := by
  simp [duck_count]
  norm_num

#eval duck_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_count_proof_l1265_126538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_sister_point_pairs_l1265_126589

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else (x + 1) / Real.exp 1

-- Define what it means for two points to be a sister point pair
def is_sister_point_pair (A B : ℝ × ℝ) : Prop :=
  A ≠ B ∧
  f A.1 = A.2 ∧
  f B.1 = B.2 ∧
  B.1 = -A.1 ∧
  B.2 = -A.2

-- State the theorem
theorem f_has_two_sister_point_pairs :
  ∃ (A₁ B₁ A₂ B₂ : ℝ × ℝ),
    is_sister_point_pair A₁ B₁ ∧
    is_sister_point_pair A₂ B₂ ∧
    A₁ ≠ A₂ ∧
    (∀ (C D : ℝ × ℝ), is_sister_point_pair C D → (C = A₁ ∧ D = B₁) ∨ (C = B₁ ∧ D = A₁) ∨ (C = A₂ ∧ D = B₂) ∨ (C = B₂ ∧ D = A₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_sister_point_pairs_l1265_126589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_is_thales_circle_l1265_126578

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point outside the circle
def PointOutside (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

-- Define a point on the circle
def OnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define if a point is on a line
def OnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the set of midpoints
def MidpointSet (c : Circle) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  { m : ℝ × ℝ | ∃ (l : Line), OnLine l p ∧ (∃ a b : ℝ × ℝ, 
    OnCircle c a ∧ OnCircle c b ∧ OnLine l a ∧ OnLine l b ∧ 
    m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) }

-- Define the circle with diameter OP
noncomputable def ThalesCircle (c : Circle) (p : ℝ × ℝ) : Circle :=
  { center := ((c.center.1 + p.1) / 2, (c.center.2 + p.2) / 2),
    radius := Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) / 2 }

-- Define membership in a circle
def InCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 ≤ c.radius^2

-- The theorem to be proved
theorem midpoint_set_is_thales_circle (c : Circle) (p : ℝ × ℝ) 
  (h : PointOutside c p) : 
  MidpointSet c p = { m : ℝ × ℝ | InCircle (ThalesCircle c p) m ∧ InCircle c m } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_is_thales_circle_l1265_126578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_strictly_increasing_two_zeros_l1265_126597

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - (x + 1) / (x - 1)

-- Define the domain of f
def domain (x : ℝ) : Prop := (0 < x ∧ x < 1) ∨ (x > 1)

-- Theorem 1: For any natural number n, f(n) + f(1/n) = 0
theorem sum_reciprocals (n : ℕ) (hn : n > 1) : f n + f (1 / (n : ℝ)) = 0 := by
  sorry

-- Theorem 2: f is strictly increasing on its domain
theorem strictly_increasing (x₁ x₂ : ℝ) (hx₁ : domain x₁) (hx₂ : domain x₂) (h : x₁ < x₂) : 
  f x₁ < f x₂ := by
  sorry

-- Theorem 3: f has exactly two zeros, and their product is 1
theorem two_zeros : ∃! (x₁ x₂ : ℝ), domain x₁ ∧ domain x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ * x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_strictly_increasing_two_zeros_l1265_126597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_squared_equals_ten_l1265_126508

-- Define the sum of sin^2 for angles from 0° to 180° in 10° increments
noncomputable def sin_sum_squared : ℝ :=
  (Finset.range 19).sum (λ i => Real.sin (↑i * 10 * Real.pi / 180) ^ 2)

-- Theorem statement
theorem sin_sum_squared_equals_ten : sin_sum_squared = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_squared_equals_ten_l1265_126508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1265_126574

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

-- State the theorem
theorem f_monotonicity :
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → f x > f y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1265_126574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_5_50_l1265_126598

/-- Calculates the angle between clock hands at a given time -/
noncomputable def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  |60 * (hour % 12 : ℝ) - 11 * (minute : ℝ)| / 2

/-- The angle between clock hands at 5:50 is 125° -/
theorem angle_at_5_50 : clockAngle 5 50 = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_5_50_l1265_126598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_sets_l1265_126559

def is_valid_set (S : Finset ℕ) : Prop :=
  S.Nonempty ∧ S.card ≥ 2 ∧
  ∀ m n, m ∈ S → n ∈ S → m > n → (n * n) / (m - n) ∈ S

theorem characterize_valid_sets :
  ∀ S : Finset ℕ, is_valid_set S →
    ∃ s : ℕ, s > 0 ∧ S = {s, 2 * s} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_sets_l1265_126559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l1265_126582

-- Define the triangle ABC and points D and E
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

-- Define the ratios
def bd_dc_ratio : ℚ × ℚ := (2, 3)
def ae_ec_ratio : ℚ × ℚ := (4, 1)

-- Define point P as the intersection of BE and AD
noncomputable def P (t : Triangle) : ℝ × ℝ := sorry

-- Express P as a linear combination of A, B, and C
noncomputable def P_vector (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_point_coordinates (t : Triangle) :
  P_vector t = ((11 : ℝ)/41, (21 : ℝ)/41, (9 : ℝ)/41) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l1265_126582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_walk_l1265_126586

/-- A polygon with sides of length 5 meters and interior angles of 160° -/
structure Polygon where
  sideLength : ℝ
  interiorAngle : ℝ

/-- The number of sides in the polygon -/
noncomputable def numberOfSides (p : Polygon) : ℝ := 360 / (180 - p.interiorAngle)

/-- The perimeter of the polygon -/
noncomputable def perimeter (p : Polygon) : ℝ := p.sideLength * numberOfSides p

theorem xiaoming_walk (p : Polygon) 
  (h1 : p.sideLength = 5)
  (h2 : p.interiorAngle = 160) :
  numberOfSides p = 18 ∧ perimeter p = 90 := by
  sorry

#check xiaoming_walk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_walk_l1265_126586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_deductive_reasoning_iron_conducts_electricity_l1265_126539

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)
variable (Iron : U)

-- Define DeductiveReasoning as a type to represent the concept
def DeductiveReasoning (α : Type) : Prop :=
  ∀ (P Q : α → Prop) (a : α),
    (∀ x, P x → Q x) → P a → Q a

-- State the theorem
theorem is_deductive_reasoning
  (h1 : ∀ x, Metal x → ConductsElectricity x)
  (h2 : Metal Iron)
  : DeductiveReasoning U :=
by
  -- Define DeductiveReasoning
  intro P Q a h_PQ h_Pa
  -- Apply the definition of DeductiveReasoning
  exact h_PQ a h_Pa

-- Prove that the given argument is an instance of deductive reasoning
theorem iron_conducts_electricity
  (h1 : ∀ x, Metal x → ConductsElectricity x)
  (h2 : Metal Iron)
  : ConductsElectricity Iron :=
by
  -- Apply the deductive reasoning
  exact h1 Iron h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_deductive_reasoning_iron_conducts_electricity_l1265_126539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_regular_hexagon_l1265_126596

/-- The central angle of a regular hexagon is 60 degrees. -/
theorem central_angle_regular_hexagon : ℝ := by
  let circle_degrees : ℝ := 360
  let hexagon_sides : ℕ := 6
  let central_angle : ℝ := circle_degrees / hexagon_sides
  have : central_angle = 60 := by
    -- Proof goes here
    sorry
  exact central_angle


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_regular_hexagon_l1265_126596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1265_126566

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

-- State the theorem
theorem sum_of_max_min_f : 
  (⨆ (x : ℝ), f x) + (⨅ (x : ℝ), f x) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1265_126566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_equivalent_sum_l1265_126506

/-- The number of faces on a standard die -/
def standardDiceFaces : ℕ := 6

/-- The target sum we want to match the probability of -/
def targetSum : ℕ := 500

/-- The sum of n dice rolls -/
def sum (n : ℕ) (rolls : Fin n → Fin standardDiceFaces) : ℕ := 
  (Finset.univ.sum (λ i => (rolls i).val) : ℕ)

/-- The probability of an event occurring -/
noncomputable def Prob {α : Type*} (event : α → Prop) : ℝ := sorry

/-- 
Given n standard 6-sided dice, where the probability of obtaining a sum of 500 
is greater than zero and equal to the probability of obtaining a sum of S, 
the smallest possible value of S is 88.
-/
theorem smallest_equivalent_sum (n : ℕ) (S : ℕ) 
  (h1 : n * standardDiceFaces ≥ targetSum)
  (h2 : ∃ (rolls : Fin n → Fin standardDiceFaces), sum n rolls = targetSum)
  (h3 : Prob (λ rolls => sum n rolls = targetSum) = Prob (λ rolls => sum n rolls = S))
  (h4 : ∀ S' < S, Prob (λ rolls => sum n rolls = targetSum) ≠ Prob (λ rolls => sum n rolls = S')) :
  S = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_equivalent_sum_l1265_126506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volumes_l1265_126570

/-- Represents a truncated quadrangular pyramid -/
structure TruncatedPyramid where
  lowerBaseSide : ℝ
  upperBaseSide : ℝ
  height : ℝ

/-- Calculates the volume of a part of the truncated pyramid -/
noncomputable def volumeOfPart (a b h : ℝ) : ℝ :=
  (h / 3) * (a^2 + b^2 + a*b)

/-- Theorem stating the volumes of the two parts of the truncated pyramid -/
theorem truncated_pyramid_volumes (p : TruncatedPyramid) 
  (h_lower : p.lowerBaseSide = 2)
  (h_upper : p.upperBaseSide = 1)
  (h_height : p.height = 3) :
  let middleBaseSide := (4:ℝ)/3
  let upperHeight := 1
  let lowerHeight := 2
  (volumeOfPart p.upperBaseSide middleBaseSide upperHeight = 37/27) ∧
  (volumeOfPart p.lowerBaseSide middleBaseSide lowerHeight = 152/27) := by
  sorry

#check truncated_pyramid_volumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volumes_l1265_126570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l1265_126571

noncomputable def complex_number : ℂ := 1 / (1 + Complex.I)

theorem complex_number_in_fourth_quadrant :
  let z := complex_number
  (z.re > 0) ∧ (z.im < 0) :=
by
  -- Introduce z
  let z := complex_number
  
  -- Expand the definition of complex_number
  have h1 : z = 1 / (1 + Complex.I) := rfl
  
  -- Multiply numerator and denominator by the conjugate
  have h2 : z = (1 - Complex.I) / ((1 + Complex.I) * (1 - Complex.I)) := by sorry
  
  -- Simplify
  have h3 : z = (1/2 : ℝ) - (1/2 : ℝ) * Complex.I := by sorry
  
  -- Check real part is positive
  have real_pos : z.re > 0 := by sorry
  
  -- Check imaginary part is negative
  have imag_neg : z.im < 0 := by sorry
  
  -- Combine the results
  exact ⟨real_pos, imag_neg⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l1265_126571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_l1265_126580

/-- Given a cone with volume 1, if a plane parallel to the base cuts the cone such that
the ratio of the radii of the top and bottom bases of the resulting truncated cone is 1/2,
then the volume of the truncated cone is 7/8. -/
theorem truncated_cone_volume (cone : Real → Real → Real) (plane : Real → Real → Real → Real) :
  (∀ r h, cone r h = (1/3) * Real.pi * r^2 * h) →
  (∃ a b c d, ∀ x y z, plane x y z = a*x + b*y + c*z + d) →
  (∃ r h, cone r h = 1) →
  (∃ k, k > 0 ∧ ∀ x y z, plane x y z = cone x y - k) →
  (∃ r₁ r₂ h₁ h₂, 
    r₂ / r₁ = 1/2 ∧ 
    cone r₁ h₁ - cone r₂ h₂ = 7/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_l1265_126580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_three_point_probability_l1265_126502

def probability_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem basketball_three_point_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  let n : ℕ := 10
  let k : ℕ := 3
  probability_exactly_k_successes n k p = (n.choose k : ℝ) * p^k * (1 - p)^(n - k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_three_point_probability_l1265_126502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_lower_bound_l1265_126573

/-- Sum of digits of an integer -/
def sum_of_digits (n : ℤ) : ℕ :=
  sorry

/-- Given an integer N, if (N - 46) has a sum of digits equal to 352, 
    then the sum of digits of N is at least 362. -/
theorem sum_of_digits_lower_bound (N : ℤ) : 
  (sum_of_digits (N - 46) = 352) → sum_of_digits N ≥ 362 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_lower_bound_l1265_126573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l1265_126565

/-- Predicate to check if a point is the focus of a parabola -/
def is_focus (f : ℝ × ℝ) (parabola : ℝ × ℝ → Prop) : Prop :=
  ∀ p, parabola p → 
    (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 + f.1)^2

/-- Predicate to check if a line is the directrix of a parabola -/
def is_directrix (d : ℝ → Prop) (parabola : ℝ × ℝ → Prop) : Prop :=
  ∀ p, parabola p → 
    ∃ q, d q ∧ (p.1 - q)^2 + p.2^2 = (p.1 + q)^2

/-- A parabola with equation y² = 2x has its focus at (1/2, 0) and directrix x = -1/2 -/
theorem parabola_focus_and_directrix :
  ∀ x y : ℝ, y^2 = 2*x →
  (∃ f : ℝ × ℝ, f = (1/2, 0) ∧ is_focus f (λ p ↦ p.2^2 = 2*p.1)) ∧
  (∃ d : ℝ → Prop, d = (λ x ↦ x = -1/2) ∧ is_directrix d (λ p ↦ p.2^2 = 2*p.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l1265_126565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1265_126533

/-- Given a parabola y = ax^2 + bx - 5 that intersects the x-axis at (-1, 0) and (5, 0),
    prove that its equation is y = x^2 - 4x - 5 -/
theorem parabola_equation (a b : ℝ) : 
  (a * (-1)^2 + b * (-1) - 5 = 0) →
  (a * 5^2 + b * 5 - 5 = 0) →
  (∀ x, a * x^2 + b * x - 5 = x^2 - 4*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1265_126533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solution_proof_l1265_126524

theorem integer_solution_proof (x y d : ℕ) : 
  x^2 + y^2 = 468 → d + (x * y) / d = 42 → 
  ((x = 12 ∧ y = 18) ∨ (x = 18 ∧ y = 12)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solution_proof_l1265_126524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_area_theorem_l1265_126567

theorem grid_area_theorem (total_rows : ℕ) (total_cols : ℕ) (shaded_area : ℝ) 
  (h1 : total_rows = 5)
  (h2 : total_cols = 8)
  (h3 : shaded_area = 37) : 
  (total_rows * total_cols : ℝ) - 18.5 * (shaded_area / 18.5) = 43 := by
  sorry

#check grid_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_area_theorem_l1265_126567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_normal_symmetry_l1265_126535

-- Define the standard normal distribution function
noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x^2) / 2)

-- Define Φ(x) as the cumulative distribution function
noncomputable def Φ (x : ℝ) : ℝ := ∫ y in Set.Iio x, f y

-- State the theorem
theorem standard_normal_symmetry : Φ 0 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_normal_symmetry_l1265_126535
