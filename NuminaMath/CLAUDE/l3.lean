import Mathlib

namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3_381

theorem geometric_progression_problem (b₁ q : ℚ) : 
  (b₁ * q^3 - b₁ * q = -45/32) → 
  (b₁ * q^5 - b₁ * q^3 = -45/512) → 
  ((b₁ = 6 ∧ q = 1/4) ∨ (b₁ = -6 ∧ q = -1/4)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3_381


namespace NUMINAMATH_CALUDE_square_root_of_49_l3_327

theorem square_root_of_49 : 
  {x : ℝ | x^2 = 49} = {7, -7} := by sorry

end NUMINAMATH_CALUDE_square_root_of_49_l3_327


namespace NUMINAMATH_CALUDE_triangle_rotation_l3_382

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices O, P, and Q -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point) : ℝ := sorry

/-- Rotates a point 90 degrees counter-clockwise around the origin -/
def rotate90 (p : Point) : Point :=
  { x := -p.y, y := p.x }

/-- The main theorem -/
theorem triangle_rotation (t : Triangle) : 
  t.O = ⟨0, 0⟩ → 
  t.P = ⟨7, 0⟩ → 
  t.Q.x > 0 → 
  t.Q.y > 0 → 
  angle t.P t.Q = π / 2 → 
  angle t.P t.Q - angle t.O t.Q = π / 4 → 
  rotate90 t.Q = ⟨-7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2⟩ := by sorry

end NUMINAMATH_CALUDE_triangle_rotation_l3_382


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3_393

/-- Given a geometric sequence {a_n} with common ratio 2 and sum of first four terms equal to 1,
    prove that the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 1 + a 2 + a 3 + a 4 = 1) →  -- sum of first four terms is 1
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3_393


namespace NUMINAMATH_CALUDE_math_books_count_l3_326

/-- Proves that the number of math books bought is 60 given the specified conditions -/
theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℕ) (total_price : ℕ) :
  total_books = 90 →
  math_book_price = 4 →
  history_book_price = 5 →
  total_price = 390 →
  ∃ (math_books : ℕ), math_books = 60 ∧ 
    math_books + (total_books - math_books) = total_books ∧
    math_book_price * math_books + history_book_price * (total_books - math_books) = total_price :=
by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l3_326


namespace NUMINAMATH_CALUDE_ellipse_equation_l3_324

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the general form of the ellipse
def ellipse (x y A B : ℝ) : Prop := (x^2 / A) + (y^2 / B) = 1

-- State the theorem
theorem ellipse_equation 
  (x y A B : ℝ) 
  (h1 : A > 0) 
  (h2 : B > 0) 
  (h3 : ∃ (xf yf xv yv : ℝ), 
    line xf yf ∧ line xv yv ∧ 
    ellipse xf yf A B ∧ ellipse xv yv A B ∧ 
    ((xf = 0 ∧ xv ≠ 0) ∨ (yf = 0 ∧ yv ≠ 0))) :
  ((A = 5 ∧ B = 4) ∨ (A = 1 ∧ B = 5)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3_324


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l3_315

/-- The expected total rainfall over a week given daily probabilities --/
theorem expected_weekly_rainfall (p_sun p_light p_heavy : ℝ) 
  (rain_light rain_heavy : ℝ) (days : ℕ) :
  p_sun + p_light + p_heavy = 1 →
  p_sun = 0.5 →
  p_light = 0.2 →
  p_heavy = 0.3 →
  rain_light = 2 →
  rain_heavy = 5 →
  days = 7 →
  (p_sun * 0 + p_light * rain_light + p_heavy * rain_heavy) * days = 13.3 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l3_315


namespace NUMINAMATH_CALUDE_sum_of_primes_less_than_20_is_77_l3_354

def is_prime (n : ℕ) : Prop := sorry

def sum_of_primes_less_than_20 : ℕ := sorry

theorem sum_of_primes_less_than_20_is_77 : 
  sum_of_primes_less_than_20 = 77 := by sorry

end NUMINAMATH_CALUDE_sum_of_primes_less_than_20_is_77_l3_354


namespace NUMINAMATH_CALUDE_limit_hours_proof_l3_345

/-- The limit of hours per week for the regular rate -/
def limit_hours : ℕ := sorry

/-- The regular hourly rate in dollars -/
def regular_rate : ℚ := 16

/-- The overtime rate as a percentage increase over the regular rate -/
def overtime_rate_increase : ℚ := 75 / 100

/-- The total hours worked in a week -/
def total_hours : ℕ := 44

/-- The total compensation earned in dollars -/
def total_compensation : ℚ := 752

/-- Calculates the overtime rate based on the regular rate and overtime rate increase -/
def overtime_rate : ℚ := regular_rate * (1 + overtime_rate_increase)

theorem limit_hours_proof :
  regular_rate * limit_hours + 
  overtime_rate * (total_hours - limit_hours) = 
  total_compensation ∧ 
  limit_hours = 40 := by sorry

end NUMINAMATH_CALUDE_limit_hours_proof_l3_345


namespace NUMINAMATH_CALUDE_meal_contribution_proof_l3_377

/-- Calculates the individual contribution for a shared meal bill -/
def calculate_individual_contribution (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) : ℚ :=
  (total_price - coupon_value) / num_people

/-- Proves that the individual contribution for the given scenario is $21 -/
theorem meal_contribution_proof (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) 
  (h1 : total_price = 67)
  (h2 : coupon_value = 4)
  (h3 : num_people = 3) :
  calculate_individual_contribution total_price coupon_value num_people = 21 := by
  sorry

#eval calculate_individual_contribution 67 4 3

end NUMINAMATH_CALUDE_meal_contribution_proof_l3_377


namespace NUMINAMATH_CALUDE_f_monotonicity_and_maximum_l3_312

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2

theorem f_monotonicity_and_maximum (k : ℝ) :
  (k = 1 →
    (∀ x y, x < y ∧ y < 0 → f k x < f k y) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < Real.log 2 → f k x > f k y) ∧
    (∀ x y, Real.log 2 < x ∧ x < y → f k x < f k y)) ∧
  (1/2 < k ∧ k ≤ 1 →
    ∀ x, x ∈ Set.Icc 0 k → f k x ≤ (k - 1) * Real.exp k - k^3) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_maximum_l3_312


namespace NUMINAMATH_CALUDE_sum_is_integer_l3_351

theorem sum_is_integer (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  ∃ n : ℤ, (x + y + z : ℝ) = n := by
sorry

end NUMINAMATH_CALUDE_sum_is_integer_l3_351


namespace NUMINAMATH_CALUDE_function_properties_l3_376

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a^x - a + 1

def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1/2) - 1

def F (a m : ℝ) (x : ℝ) : ℝ := g a (2*x) - m * g a (x - 1)

def h (m : ℝ) : ℝ :=
  if m ≤ 1 then 1 - 2*m
  else if m < 2 then -m^2
  else 4 - 4*m

theorem function_properties (a : ℝ) (ha : a > 0 ∧ a ≠ 1) (hf : f a (1/2) = 2) :
  a = 1/2 ∧
  (∀ x, g a x = (1/2)^x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, F a m x ≥ h m) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3_376


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3_396

/-- An arithmetic progression with first three terms x - 3, x + 3, and 3x + 5 has x = 2 -/
theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ : ℝ := x - 3
  let a₂ : ℝ := x + 3
  let a₃ : ℝ := 3*x + 5
  (a₂ - a₁ = a₃ - a₂) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3_396


namespace NUMINAMATH_CALUDE_remainder_8734_mod_9_l3_355

theorem remainder_8734_mod_9 : 8734 ≡ 4 [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_remainder_8734_mod_9_l3_355


namespace NUMINAMATH_CALUDE_power_function_through_point_l3_301

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 2 = 4 →
  f 9 = 81 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3_301


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3_357

theorem square_sum_given_product_and_sum (m n : ℝ) 
  (h1 : m * n = 12) 
  (h2 : m + n = 8) : 
  m^2 + n^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3_357


namespace NUMINAMATH_CALUDE_three_digit_reverse_difference_theorem_l3_378

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_do_not_repeat (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

def same_digits (m n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (m = 100 * a + 10 * b + c ∨ m = 100 * a + 10 * c + b ∨
     m = 100 * b + 10 * a + c ∨ m = 100 * b + 10 * c + a ∨
     m = 100 * c + 10 * a + b ∨ m = 100 * c + 10 * b + a) ∧
    (n = 100 * a + 10 * b + c ∨ n = 100 * a + 10 * c + b ∨
     n = 100 * b + 10 * a + c ∨ n = 100 * b + 10 * c + a ∨
     n = 100 * c + 10 * a + b ∨ n = 100 * c + 10 * b + a)

theorem three_digit_reverse_difference_theorem :
  ∀ x : ℕ,
    is_three_digit x ∧
    digits_do_not_repeat x ∧
    is_three_digit (x - reverse_number x) ∧
    same_digits x (x - reverse_number x) →
    x = 954 ∨ x = 459 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_reverse_difference_theorem_l3_378


namespace NUMINAMATH_CALUDE_total_lemonade_poured_l3_334

def first_intermission : ℝ := 0.25 + 0.125
def second_intermission : ℝ := 0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666
def third_intermission : ℝ := 0.25 + 0.125
def fourth_intermission : ℝ := 0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666

theorem total_lemonade_poured :
  first_intermission + second_intermission + third_intermission + fourth_intermission = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_total_lemonade_poured_l3_334


namespace NUMINAMATH_CALUDE_some_base_value_l3_373

theorem some_base_value (x y some_base : ℝ) 
  (h1 : x * y = 1) 
  (h2 : (some_base^((x + y)^2)) / (some_base^((x - y)^2)) = 2401) : 
  some_base = 7 := by sorry

end NUMINAMATH_CALUDE_some_base_value_l3_373


namespace NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l3_319

/-- Represents the hill run by Jack and Jill -/
structure HillRun where
  length : ℝ
  jack_uphill_speed : ℝ
  jack_downhill_speed : ℝ
  jill_uphill_speed : ℝ
  jill_downhill_speed : ℝ
  jack_pause_time : ℝ
  jack_pause_location : ℝ

/-- Calculates the meeting point of Jack and Jill -/
def meeting_point (h : HillRun) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem jack_and_jill_meeting_point (h : HillRun) 
  (h_length : h.length = 6)
  (h_jack_up : h.jack_uphill_speed = 12)
  (h_jack_down : h.jack_downhill_speed = 18)
  (h_jill_up : h.jill_uphill_speed = 15)
  (h_jill_down : h.jill_downhill_speed = 21)
  (h_pause_time : h.jack_pause_time = 0.25)
  (h_pause_loc : h.jack_pause_location = 3) :
  meeting_point h = 63 / 22 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l3_319


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l3_383

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 2 ∧
  (1/2) * t.b * t.c * Real.sin t.A = (Real.sqrt 2 / 2) * (t.c * Real.sin t.C + t.b * Real.sin t.B - t.a * Real.sin t.A)

-- Theorem for the value of angle A
theorem angle_A_value (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 := by
  sorry

-- Theorem for the maximum area of triangle ABC
theorem max_area (t : Triangle) (h : triangle_conditions t) : 
  (∀ t' : Triangle, triangle_conditions t' → (1/2) * t'.b * t'.c * Real.sin t'.A ≤ Real.sqrt 3 / 2) ∧
  (∃ t' : Triangle, triangle_conditions t' ∧ (1/2) * t'.b * t'.c * Real.sin t'.A = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_max_area_l3_383


namespace NUMINAMATH_CALUDE_total_area_of_three_triangles_l3_328

theorem total_area_of_three_triangles (base height : ℝ) (h1 : base = 40) (h2 : height = 20) :
  3 * (1/2 * base * height) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_three_triangles_l3_328


namespace NUMINAMATH_CALUDE_equation_solution_l3_322

theorem equation_solution : ∀ x : ℚ, (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3_322


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l3_330

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | 3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3} = 
  {(0, 0), (6, 6), (-6, -6)} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l3_330


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3_306

/-- Given a geometric progression with the first three terms, find the fourth term -/
theorem fourth_term_of_geometric_progression (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3^(1/4 : ℝ)) 
    (h₂ : a₂ = 3^(1/6 : ℝ)) (h₃ : a₃ = 3^(1/12 : ℝ)) : 
  ∃ (a₄ : ℝ), a₄ = (a₃ * a₂) / a₁ ∧ a₄ = 1 := by
sorry


end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3_306


namespace NUMINAMATH_CALUDE_prism_18_edges_8_faces_l3_386

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ (p : Prism), p.edges = 18 → num_faces p = 8 := by
  sorry

#check prism_18_edges_8_faces

end NUMINAMATH_CALUDE_prism_18_edges_8_faces_l3_386


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3_346

theorem arithmetic_calculations :
  (128 + 52 / 13 = 132) ∧
  (132 / 11 * 29 - 178 = 170) ∧
  (45 * (320 / (4 * 5)) = 720) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3_346


namespace NUMINAMATH_CALUDE_magnitude_of_b_magnitude_of_c_and_area_l3_379

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 15 ∧ Real.sin t.A = 1/4

-- Theorem 1
theorem magnitude_of_b (t : Triangle) (h : triangle_conditions t) 
  (hcosB : Real.cos t.B = Real.sqrt 5 / 3) :
  t.b = 8 * Real.sqrt 15 / 3 :=
sorry

-- Theorem 2
theorem magnitude_of_c_and_area (t : Triangle) (h : triangle_conditions t) 
  (hb : t.b = 4 * t.a) :
  t.c = 15 ∧ (1/2 * t.b * t.c * Real.sin t.A = 15/2 * Real.sqrt 15) :=
sorry

end NUMINAMATH_CALUDE_magnitude_of_b_magnitude_of_c_and_area_l3_379


namespace NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l3_390

theorem pascal_triangle_row_15_fifth_number :
  let row := List.map (fun k => Nat.choose 15 k) (List.range 16)
  row[0] = 1 ∧ row[1] = 15 →
  row[4] = Nat.choose 15 4 ∧ Nat.choose 15 4 = 1365 :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l3_390


namespace NUMINAMATH_CALUDE_term_2007_is_6019_l3_368

/-- An arithmetic sequence with first term 1, second term 4, and third term 7 -/
def arithmetic_sequence (n : ℕ) : ℕ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 2007th term of the sequence is 6019 -/
theorem term_2007_is_6019 : arithmetic_sequence 2007 = 6019 := by
  sorry

end NUMINAMATH_CALUDE_term_2007_is_6019_l3_368


namespace NUMINAMATH_CALUDE_bottles_theorem_l3_311

/-- The number of ways to take out 24 bottles, where each time either 3 or 4 bottles are taken -/
def ways_to_take_bottles : ℕ :=
  -- Number of ways to take out 4 bottles 6 times
  1 +
  -- Number of ways to take out 3 bottles 8 times
  1 +
  -- Number of ways to take out 3 bottles 4 times and 4 bottles 3 times
  (Nat.choose 7 3)

/-- Theorem stating that the number of ways to take out the bottles is 37 -/
theorem bottles_theorem : ways_to_take_bottles = 37 := by
  sorry

#eval ways_to_take_bottles

end NUMINAMATH_CALUDE_bottles_theorem_l3_311


namespace NUMINAMATH_CALUDE_polygon_deformable_to_triangle_l3_323

/-- A planar polygon represented by its vertices -/
structure PlanarPolygon where
  vertices : List (ℝ × ℝ)
  n : ℕ
  h_n : vertices.length = n

/-- A function that checks if a polygon can be deformed into a triangle -/
def can_deform_to_triangle (p : PlanarPolygon) : Prop :=
  ∃ (v1 v2 v3 : ℝ × ℝ), v1 ∈ p.vertices ∧ v2 ∈ p.vertices ∧ v3 ∈ p.vertices ∧
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- The main theorem stating that any planar polygon with more than 4 vertices
    can be deformed into a triangle -/
theorem polygon_deformable_to_triangle (p : PlanarPolygon) (h : p.n > 4) :
  can_deform_to_triangle p := by
  sorry

end NUMINAMATH_CALUDE_polygon_deformable_to_triangle_l3_323


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3_335

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3_335


namespace NUMINAMATH_CALUDE_both_charts_rough_determination_l3_365

/-- Represents a chart type -/
inductive ChartType
  | ThreeD_Column
  | TwoD_Bar

/-- Represents the ability to determine relationships between categorical variables -/
inductive RelationshipDetermination
  | Accurate
  | Rough
  | Unable

/-- Function that determines the relationship determination capability of a chart type -/
def chart_relationship_determination : ChartType → RelationshipDetermination
  | ChartType.ThreeD_Column => RelationshipDetermination.Rough
  | ChartType.TwoD_Bar => RelationshipDetermination.Rough

/-- Theorem stating that both 3D column charts and 2D bar charts can roughly determine relationships -/
theorem both_charts_rough_determination :
  (chart_relationship_determination ChartType.ThreeD_Column = RelationshipDetermination.Rough) ∧
  (chart_relationship_determination ChartType.TwoD_Bar = RelationshipDetermination.Rough) :=
by
  sorry

#check both_charts_rough_determination

end NUMINAMATH_CALUDE_both_charts_rough_determination_l3_365


namespace NUMINAMATH_CALUDE_fish_value_calculation_l3_398

/-- Calculates the total value of non-spoiled fish after sales, spoilage, and new stock arrival --/
def fish_value (initial_trout initial_bass : ℕ) 
               (sold_trout sold_bass : ℕ) 
               (trout_price bass_price : ℚ) 
               (spoil_trout_ratio spoil_bass_ratio : ℚ)
               (new_trout new_bass : ℕ) : ℚ :=
  let remaining_trout := initial_trout - sold_trout
  let remaining_bass := initial_bass - sold_bass
  let spoiled_trout := ⌊remaining_trout * spoil_trout_ratio⌋
  let spoiled_bass := ⌊remaining_bass * spoil_bass_ratio⌋
  let final_trout := remaining_trout - spoiled_trout + new_trout
  let final_bass := remaining_bass - spoiled_bass + new_bass
  final_trout * trout_price + final_bass * bass_price

/-- The theorem statement --/
theorem fish_value_calculation :
  fish_value 120 80 30 20 5 10 (1/4) (1/3) 150 50 = 1990 := by
  sorry

end NUMINAMATH_CALUDE_fish_value_calculation_l3_398


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l3_385

theorem excluded_students_average_mark
  (N : ℕ)  -- Total number of students
  (A : ℚ)  -- Average mark of all students
  (E : ℕ)  -- Number of excluded students
  (AR : ℚ) -- Average mark of remaining students
  (h1 : N = 25)
  (h2 : A = 80)
  (h3 : E = 5)
  (h4 : AR = 90)
  : ∃ AE : ℚ, AE = 40 ∧ N * A - E * AE = (N - E) * AR :=
sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l3_385


namespace NUMINAMATH_CALUDE_prob_two_females_is_three_tenths_l3_380

-- Define the total number of contestants
def total_contestants : ℕ := 5

-- Define the number of female contestants
def female_contestants : ℕ := 3

-- Define the number of contestants to be chosen
def chosen_contestants : ℕ := 2

-- Define the probability of choosing 2 female contestants
def prob_two_females : ℚ := (female_contestants.choose chosen_contestants) / (total_contestants.choose chosen_contestants)

-- Theorem statement
theorem prob_two_females_is_three_tenths : prob_two_females = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_females_is_three_tenths_l3_380


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_nine_l3_358

/-- Given nonzero digits a, b, and c that form 6 distinct three-digit numbers,
    if the sum of these numbers is 5994, then each number is divisible by 9. -/
theorem three_digit_numbers_divisible_by_nine 
  (a b c : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hsum : 100 * (a + b + c) + 10 * (a + b + c) + (a + b + c) = 5994) :
  let numbers := [100*a + 10*b + c, 100*a + 10*c + b, 
                  100*b + 10*a + c, 100*b + 10*c + a, 
                  100*c + 10*a + b, 100*c + 10*b + a]
  ∀ n ∈ numbers, n % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_nine_l3_358


namespace NUMINAMATH_CALUDE_line_through_points_l3_397

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points :
  let p1 : Point := ⟨2, 10⟩
  let p2 : Point := ⟨6, 26⟩
  let p3 : Point := ⟨10, 42⟩
  let p4 : Point := ⟨45, 182⟩
  collinear p1 p2 p3 → collinear p1 p2 p4 :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3_397


namespace NUMINAMATH_CALUDE_percentage_relation_l3_339

theorem percentage_relation (x y : ℕ) (N : ℚ) (hx : Prime x) (hy : Prime y) (hxy : x ≠ y) 
  (h : 70 = (x : ℚ) / 100 * N) : 
  (y : ℚ) / 100 * N = (y * 70 : ℚ) / x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3_339


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l3_374

theorem consecutive_integers_sum_of_powers (n : ℤ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 = 2450 → 
  (n - 1)^5 + n^5 + (n + 1)^5 = 52070424 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l3_374


namespace NUMINAMATH_CALUDE_andrews_eggs_l3_361

theorem andrews_eggs (initial_eggs bought_eggs final_eggs : ℕ) : 
  bought_eggs = 62 → final_eggs = 70 → initial_eggs + bought_eggs = final_eggs → initial_eggs = 8 := by
  sorry

end NUMINAMATH_CALUDE_andrews_eggs_l3_361


namespace NUMINAMATH_CALUDE_equation_solution_l3_340

theorem equation_solution :
  ∃ (X Y : ℚ), 
    (∀ x : ℚ, x ≠ 5 ∧ x ≠ 6 → 
      (Y * x + 8) / (x^2 - 11*x + 30) = X / (x - 5) + 7 / (x - 6)) →
    X + Y = -22/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3_340


namespace NUMINAMATH_CALUDE_shaded_region_area_l3_375

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The shaded region formed by the intersection of three semicircles -/
def ShadedRegion (s1 s2 s3 : Semicircle) : Set Point := sorry

/-- The area of a set of points -/
def area (s : Set Point) : ℝ := sorry

/-- The midpoint of an arc -/
def arcMidpoint (s : Semicircle) : Point := sorry

theorem shaded_region_area 
  (s1 s2 s3 : Semicircle)
  (h1 : s1.radius = 2 ∧ s2.radius = 2 ∧ s3.radius = 2)
  (h2 : arcMidpoint s1 = s3.center)
  (h3 : arcMidpoint s2 = s3.center)
  (h4 : s3.center = arcMidpoint s3) :
  area (ShadedRegion s1 s2 s3) = 8 := by sorry

end NUMINAMATH_CALUDE_shaded_region_area_l3_375


namespace NUMINAMATH_CALUDE_equation_solutions_l3_305

theorem equation_solutions :
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3_305


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l3_336

theorem quadratic_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x ∈ Set.Icc 0 3, a * x^2 - 2 * a * x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 3, a * x^2 - 2 * a * x = 3) →
  a = -3 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l3_336


namespace NUMINAMATH_CALUDE_amithab_average_expenditure_l3_314

/-- Given Amithab's monthly expenses, prove the average expenditure for February to July. -/
theorem amithab_average_expenditure
  (jan_expense : ℕ)
  (jan_to_jun_avg : ℕ)
  (jul_expense : ℕ)
  (h1 : jan_expense = 1200)
  (h2 : jan_to_jun_avg = 4200)
  (h3 : jul_expense = 1500) :
  (6 * jan_to_jun_avg - jan_expense + jul_expense) / 6 = 4250 :=
by sorry

end NUMINAMATH_CALUDE_amithab_average_expenditure_l3_314


namespace NUMINAMATH_CALUDE_hare_leaps_per_dog_leap_is_two_l3_331

/-- The number of hare leaps equal to one dog leap -/
def hare_leaps_per_dog_leap : ℕ := 2

/-- The number of dog leaps for a given number of hare leaps -/
def dog_leaps (hare_leaps : ℕ) : ℕ := (5 * hare_leaps : ℕ)

/-- The ratio of dog speed to hare speed -/
def speed_ratio : ℕ := 10

theorem hare_leaps_per_dog_leap_is_two :
  hare_leaps_per_dog_leap = 2 ∧
  (∀ h : ℕ, dog_leaps h = 5 * h) ∧
  speed_ratio = 10 := by
  sorry

end NUMINAMATH_CALUDE_hare_leaps_per_dog_leap_is_two_l3_331


namespace NUMINAMATH_CALUDE_sector_properties_l3_359

/-- Given a sector with radius 2 cm and central angle 2 radians, prove that its arc length is 4 cm and its area is 4 cm². -/
theorem sector_properties :
  let r : ℝ := 2  -- radius in cm
  let α : ℝ := 2  -- central angle in radians
  let arc_length : ℝ := r * α
  let sector_area : ℝ := (1/2) * r^2 * α
  (arc_length = 4 ∧ sector_area = 4) :=
by sorry

end NUMINAMATH_CALUDE_sector_properties_l3_359


namespace NUMINAMATH_CALUDE_train_length_l3_348

/-- The length of a train given its crossing times over two platforms -/
theorem train_length (platform1_length platform2_length : ℝ)
                     (crossing_time1 crossing_time2 : ℝ)
                     (h1 : platform1_length = 200)
                     (h2 : platform2_length = 300)
                     (h3 : crossing_time1 = 15)
                     (h4 : crossing_time2 = 20) :
  ∃ (train_length : ℝ),
    train_length = 100 ∧
    (train_length + platform1_length) / crossing_time1 =
    (train_length + platform2_length) / crossing_time2 :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l3_348


namespace NUMINAMATH_CALUDE_gregs_mom_cookies_l3_364

theorem gregs_mom_cookies (greg_halves brad_halves left_halves : ℕ) 
  (h1 : greg_halves = 4)
  (h2 : brad_halves = 6)
  (h3 : left_halves = 18) :
  (greg_halves + brad_halves + left_halves) / 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gregs_mom_cookies_l3_364


namespace NUMINAMATH_CALUDE_perfect_squares_from_products_l3_350

theorem perfect_squares_from_products (a b c d : ℕ) 
  (h1 : ∃ x : ℕ, a * b * c = x ^ 2)
  (h2 : ∃ x : ℕ, a * c * d = x ^ 2)
  (h3 : ∃ x : ℕ, b * c * d = x ^ 2)
  (h4 : ∃ x : ℕ, a * b * d = x ^ 2) :
  (∃ w : ℕ, a = w ^ 2) ∧ 
  (∃ x : ℕ, b = x ^ 2) ∧ 
  (∃ y : ℕ, c = y ^ 2) ∧ 
  (∃ z : ℕ, d = z ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_from_products_l3_350


namespace NUMINAMATH_CALUDE_a1_iff_a2017_positive_l3_372

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ
  q : ℝ

/-- The theorem stating that for an arithmetic-geometric sequence with q = 0,
    a₁ > 0 if and only if a₂₀₁₇ > 0 -/
theorem a1_iff_a2017_positive (seq : ArithmeticGeometricSequence) 
    (h_q : seq.q = 0) :
    seq.a 1 > 0 ↔ seq.a 2017 > 0 := by
  sorry

end NUMINAMATH_CALUDE_a1_iff_a2017_positive_l3_372


namespace NUMINAMATH_CALUDE_parallelogram_missing_vertex_l3_371

/-- A parallelogram in a 2D coordinate system -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- The area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Theorem: Given a parallelogram with three known vertices and a known area,
    prove that the fourth vertex has specific coordinates -/
theorem parallelogram_missing_vertex 
  (p : Parallelogram)
  (h1 : p.v1 = (4, 4))
  (h2 : p.v3 = (5, 9))
  (h3 : p.v4 = (8, 9))
  (h4 : area p = 5) :
  p.v2 = (3, 4) := by sorry

end NUMINAMATH_CALUDE_parallelogram_missing_vertex_l3_371


namespace NUMINAMATH_CALUDE_exactly_four_false_l3_352

/-- Represents a statement about the number of false statements -/
inductive Statement
  | one
  | two
  | three
  | four
  | five

/-- Returns true if the statement is consistent with the given number of false statements -/
def isConsistent (s : Statement) (numFalse : Nat) : Bool :=
  match s with
  | .one => numFalse = 1
  | .two => numFalse = 2
  | .three => numFalse = 3
  | .four => numFalse = 4
  | .five => numFalse = 5

/-- The list of all statements on the card -/
def allStatements : List Statement := [.one, .two, .three, .four, .five]

/-- Counts the number of false statements given a predicate -/
def countFalse (pred : Statement → Bool) : Nat :=
  allStatements.filter (fun s => !pred s) |>.length

theorem exactly_four_false :
  ∃ (pred : Statement → Bool),
    (∀ s, pred s ↔ isConsistent s (countFalse pred)) ∧
    countFalse pred = 4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_false_l3_352


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l3_304

def S : Set Nat := {1, 2, 3, 4, 5}

def A : Set Nat := {x ∈ S | x % 2 = 0}

def B : Set Nat := {x ∈ S | x % 2 = 1}

theorem events_mutually_exclusive_and_complementary :
  (A ∩ B = ∅) ∧ (A ∪ B = S) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l3_304


namespace NUMINAMATH_CALUDE_baker_cakes_l3_307

/-- The initial number of cakes Baker made -/
def initial_cakes : ℕ := 169

/-- The number of cakes Baker's friend bought -/
def bought_cakes : ℕ := 137

/-- The number of cakes Baker has left -/
def remaining_cakes : ℕ := 32

/-- Theorem stating that the initial number of cakes is equal to the sum of bought cakes and remaining cakes -/
theorem baker_cakes : initial_cakes = bought_cakes + remaining_cakes := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l3_307


namespace NUMINAMATH_CALUDE_jenny_recycling_l3_392

/-- Represents the recycling problem with given weights and prices -/
structure RecyclingProblem where
  bottle_weight : Nat
  can_weight : Nat
  jar_weight : Nat
  max_weight : Nat
  cans_collected : Nat
  bottle_price : Nat
  can_price : Nat
  jar_price : Nat

/-- Calculates the number of jars that can be carried given the remaining weight -/
def max_jars (p : RecyclingProblem) (remaining_weight : Nat) : Nat :=
  remaining_weight / p.jar_weight

/-- Calculates the total money earned from recycling -/
def total_money (p : RecyclingProblem) (cans : Nat) (jars : Nat) (bottles : Nat) : Nat :=
  cans * p.can_price + jars * p.jar_price + bottles * p.bottle_price

/-- States the theorem about Jenny's recycling problem -/
theorem jenny_recycling (p : RecyclingProblem) 
  (h1 : p.bottle_weight = 6)
  (h2 : p.can_weight = 2)
  (h3 : p.jar_weight = 8)
  (h4 : p.max_weight = 100)
  (h5 : p.cans_collected = 20)
  (h6 : p.bottle_price = 10)
  (h7 : p.can_price = 3)
  (h8 : p.jar_price = 12) :
  let remaining_weight := p.max_weight - (p.cans_collected * p.can_weight)
  let jars := max_jars p remaining_weight
  let bottles := 0
  (cans, jars, bottles) = (20, 7, 0) ∧ 
  total_money p p.cans_collected jars bottles = 144 := by
  sorry

end NUMINAMATH_CALUDE_jenny_recycling_l3_392


namespace NUMINAMATH_CALUDE_f_decreasing_iff_b_leq_neg_two_l3_347

/-- A piecewise function f parameterized by b -/
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x =>
  if x < 0 then x^2 + (2+b)*x - 1 else (2*b-1)*x + b - 2

/-- f is decreasing on ℝ if and only if b ≤ -2 -/
theorem f_decreasing_iff_b_leq_neg_two (b : ℝ) :
  (∀ x y : ℝ, x < y → f b x > f b y) ↔ b ≤ -2 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_iff_b_leq_neg_two_l3_347


namespace NUMINAMATH_CALUDE_special_polynomial_sum_l3_384

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y z : ℝ, p x = x^3 + y*x^2 + z*x + (7 - 6*y - 6*z)) ∧ 
  p 1 = 7 ∧ p 2 = 14 ∧ p 3 = 21

theorem special_polynomial_sum (p : ℝ → ℝ) (h : special_polynomial p) : 
  p 0 + p 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_sum_l3_384


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3_362

theorem water_tank_capacity (initial_fraction : Rat) (added_volume : ℝ) (final_fraction : Rat) :
  initial_fraction = 1/3 →
  added_volume = 5 →
  final_fraction = 2/5 →
  ∃ (capacity : ℝ), capacity = 75 ∧ 
    initial_fraction * capacity + added_volume = final_fraction * capacity :=
by sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3_362


namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l3_394

theorem cubic_equation_unique_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l3_394


namespace NUMINAMATH_CALUDE_two_zeros_twelve_divisors_l3_391

def endsWithTwoZeros (n : ℕ) : Prop := n % 100 = 0

def countDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem two_zeros_twelve_divisors :
  ∀ n : ℕ, endsWithTwoZeros n ∧ countDivisors n = 12 ↔ n = 200 ∨ n = 500 := by
  sorry

end NUMINAMATH_CALUDE_two_zeros_twelve_divisors_l3_391


namespace NUMINAMATH_CALUDE_suitcase_waiting_time_l3_387

/-- The number of suitcases loaded onto the plane -/
def total_suitcases : ℕ := 200

/-- The number of suitcases belonging to the businesspeople -/
def business_suitcases : ℕ := 10

/-- The time interval between placing suitcases on the conveyor belt (in seconds) -/
def placement_interval : ℕ := 2

/-- The probability of the businesspeople waiting exactly two minutes for their last suitcase -/
def exact_two_minutes_probability : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases : ℚ)

/-- The expected time (in seconds) the businesspeople will wait for their last suitcase -/
def expected_waiting_time : ℚ := 4020 / 11

theorem suitcase_waiting_time :
  (exact_two_minutes_probability = (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases : ℚ)) ∧
  (expected_waiting_time = 4020 / 11) := by
  sorry

end NUMINAMATH_CALUDE_suitcase_waiting_time_l3_387


namespace NUMINAMATH_CALUDE_area_smallest_rectangle_radius_6_l3_367

/-- The area of the smallest rectangle containing a circle of given radius -/
def smallest_rectangle_area (radius : ℝ) : ℝ :=
  (2 * radius) * (3 * radius)

/-- Theorem: The area of the smallest rectangle containing a circle of radius 6 is 216 -/
theorem area_smallest_rectangle_radius_6 :
  smallest_rectangle_area 6 = 216 := by
  sorry

end NUMINAMATH_CALUDE_area_smallest_rectangle_radius_6_l3_367


namespace NUMINAMATH_CALUDE_boat_speed_distance_relationship_l3_353

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  stillWater : ℝ
  downstream : ℝ
  upstream : ℝ

/-- Represents distances traveled by the boat -/
structure BoatDistance where
  downstream : ℝ
  upstream : ℝ

/-- Theorem stating the relationship between boat speed, current speed, and distances traveled -/
theorem boat_speed_distance_relationship 
  (speed : BoatSpeed) 
  (distance : BoatDistance) 
  (currentSpeed : ℝ) :
  speed.stillWater = 12 →
  speed.downstream = speed.stillWater + currentSpeed →
  speed.upstream = speed.stillWater - currentSpeed →
  distance.downstream = speed.downstream * 3 →
  distance.upstream = speed.upstream * 15 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_distance_relationship_l3_353


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l3_363

/-- Calculates the total cost of typing and revising a manuscript. -/
def manuscript_cost (total_pages : ℕ) (first_time_cost : ℕ) (revision_cost : ℕ) 
  (revised_once : ℕ) (revised_twice : ℕ) (revised_thrice : ℕ) : ℕ :=
  total_pages * first_time_cost + 
  revised_once * revision_cost + 
  revised_twice * revision_cost * 2 + 
  revised_thrice * revision_cost * 3

theorem manuscript_cost_theorem : 
  manuscript_cost 500 5 4 200 150 50 = 5100 := by
  sorry

#eval manuscript_cost 500 5 4 200 150 50

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l3_363


namespace NUMINAMATH_CALUDE_fraction_equality_l3_388

theorem fraction_equality : (3023 - 2990)^2 / 121 = 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3_388


namespace NUMINAMATH_CALUDE_equal_costs_at_60_guests_unique_equal_cost_guests_l3_320

/-- Represents the venues for the prom --/
inductive Venue
| caesars_palace
| venus_hall

/-- Calculates the total cost for a given venue and number of guests --/
def total_cost (v : Venue) (guests : ℕ) : ℚ :=
  match v with
  | Venue.caesars_palace => 800 + 34 * guests
  | Venue.venus_hall => 500 + 39 * guests

/-- Proves that the total costs are equal when there are 60 guests --/
theorem equal_costs_at_60_guests :
  total_cost Venue.caesars_palace 60 = total_cost Venue.venus_hall 60 := by
  sorry

/-- Proves that 60 is the unique number of guests for which costs are equal --/
theorem unique_equal_cost_guests :
  ∀ g : ℕ, total_cost Venue.caesars_palace g = total_cost Venue.venus_hall g ↔ g = 60 := by
  sorry

end NUMINAMATH_CALUDE_equal_costs_at_60_guests_unique_equal_cost_guests_l3_320


namespace NUMINAMATH_CALUDE_square_of_rational_difference_l3_313

theorem square_of_rational_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by sorry

end NUMINAMATH_CALUDE_square_of_rational_difference_l3_313


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3_316

theorem expand_and_simplify (x : ℝ) : (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3_316


namespace NUMINAMATH_CALUDE_total_earnings_proof_l3_303

def total_earnings (jermaine_earnings terrence_earnings emilee_earnings : ℕ) : ℕ :=
  jermaine_earnings + terrence_earnings + emilee_earnings

theorem total_earnings_proof (terrence_earnings emilee_earnings : ℕ) 
  (h1 : terrence_earnings = 30)
  (h2 : emilee_earnings = 25) :
  total_earnings (terrence_earnings + 5) terrence_earnings emilee_earnings = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_total_earnings_proof_l3_303


namespace NUMINAMATH_CALUDE_min_theta_value_l3_349

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem min_theta_value (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f ω (x + p) + |f ω (x + p)| = f ω x + |f ω x| ∧
    ∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f ω (x + q) + |f ω (x + q)| = f ω x + |f ω x|) → p ≤ q) →
  (∃ (θ : ℝ), θ > 0 ∧ ∀ (x : ℝ), f ω x ≥ f ω θ) →
  (∃ (θ_min : ℝ), θ_min > 0 ∧ 
    (∀ (x : ℝ), f ω x ≥ f ω θ_min) ∧
    (∀ (θ : ℝ), θ > 0 → (∀ (x : ℝ), f ω x ≥ f ω θ) → θ_min ≤ θ) ∧
    θ_min = 5 * Real.pi / 8) :=
sorry

end NUMINAMATH_CALUDE_min_theta_value_l3_349


namespace NUMINAMATH_CALUDE_periodic_odd_function_sum_l3_356

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_sum (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 3)
  (h_odd : is_odd f)
  (h_f_neg_one : f (-1) = 2) :
  f 2011 + f 2012 = 0 := by
sorry

end NUMINAMATH_CALUDE_periodic_odd_function_sum_l3_356


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3_341

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 2}
def N : Finset Nat := {3, 4}

theorem complement_union_theorem :
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3_341


namespace NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l3_309

-- Define the variables
variable (x y z : ℝ)
-- Define constants of proportionality
variable (k j : ℝ)

-- Define the relationships
def x_varies_as_y_squared : Prop := ∃ k > 0, x = k * y^2
def y_varies_as_cube_root_z_squared : Prop := ∃ j > 0, y = j * (z^2)^(1/3)

-- State the theorem
theorem x_varies_as_four_thirds_power_of_z 
  (h1 : x_varies_as_y_squared x y) 
  (h2 : y_varies_as_cube_root_z_squared y z) : 
  ∃ m > 0, x = m * z^(4/3) := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l3_309


namespace NUMINAMATH_CALUDE_smallest_cube_box_volume_for_cone_l3_325

/-- The volume of the smallest cube-shaped box that can accommodate a cone vertically -/
theorem smallest_cube_box_volume_for_cone (cone_height : ℝ) (cone_base_diameter : ℝ) 
  (h_height : cone_height = 15) 
  (h_diameter : cone_base_diameter = 8) : ℝ := by
  sorry

#check smallest_cube_box_volume_for_cone

end NUMINAMATH_CALUDE_smallest_cube_box_volume_for_cone_l3_325


namespace NUMINAMATH_CALUDE_rose_price_is_three_l3_399

-- Define the sales data
def tulips_day1 : ℕ := 30
def roses_day1 : ℕ := 20
def tulips_day2 : ℕ := 2 * tulips_day1
def roses_day2 : ℕ := 2 * roses_day1
def tulips_day3 : ℕ := (tulips_day2 * 10) / 100
def roses_day3 : ℕ := 16

-- Define the total sales
def total_tulips : ℕ := tulips_day1 + tulips_day2 + tulips_day3
def total_roses : ℕ := roses_day1 + roses_day2 + roses_day3

-- Define the price of a tulip
def tulip_price : ℚ := 2

-- Define the total earnings
def total_earnings : ℚ := 420

-- Theorem to prove
theorem rose_price_is_three :
  ∃ (rose_price : ℚ), 
    rose_price * total_roses + tulip_price * total_tulips = total_earnings ∧
    rose_price = 3 := by
  sorry


end NUMINAMATH_CALUDE_rose_price_is_three_l3_399


namespace NUMINAMATH_CALUDE_system_solutions_l3_344

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  (x + y)^4 = 6*x^2*y^2 - 215 ∧ x*y*(x^2 + y^2) = -78

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(3, -2), (-2, 3), (-3, 2), (2, -3)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solutions := by sorry

end NUMINAMATH_CALUDE_system_solutions_l3_344


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3_317

theorem complex_fraction_simplification : 
  (((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324)) / 
   ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324))) = 221 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3_317


namespace NUMINAMATH_CALUDE_race_time_l3_395

/-- In a 1000-meter race, runner A beats runner B by 48 meters or 6 seconds -/
def Race (t : ℝ) : Prop :=
  -- A's distance in t seconds
  1000 = t * (1000 / t) ∧
  -- B's distance in t seconds
  952 = t * (952 / (t + 6)) ∧
  -- A and B have the same speed
  1000 / t = 952 / (t + 6)

/-- The time taken by runner A to complete the race is 125 seconds -/
theorem race_time : ∃ t : ℝ, Race t ∧ t = 125 := by sorry

end NUMINAMATH_CALUDE_race_time_l3_395


namespace NUMINAMATH_CALUDE_sum_21_terms_arithmetic_sequence_l3_302

/-- Arithmetic sequence with first term 3 and common difference 10 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  3 + (n - 1) * 10

/-- Sum of the first n terms of the arithmetic sequence -/
def sumArithmeticSequence (n : ℕ) : ℤ :=
  n * (3 + arithmeticSequence n) / 2

theorem sum_21_terms_arithmetic_sequence :
  sumArithmeticSequence 21 = 2163 := by
  sorry

end NUMINAMATH_CALUDE_sum_21_terms_arithmetic_sequence_l3_302


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3_343

theorem arithmetic_sequence_product (a : ℤ) : 
  (∃ x : ℤ, x * (x + 1) * (x + 2) * (x + 3) = 360) → 
  (a * (a + 1) * (a + 2) * (a + 3) = 360 → (a = 3 ∨ a = -6)) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3_343


namespace NUMINAMATH_CALUDE_sum_fraction_denominator_form_main_result_l3_389

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_fraction (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (double_factorial (2 * (i + 1))) / (double_factorial (2 * (i + 1) + 1)))

theorem sum_fraction_denominator_form (n : ℕ) :
  ∃ (a b : ℕ), b % 2 = 1 ∧ (sum_fraction n).den = 2^a * b := by sorry

theorem main_result : ∃ (a b : ℕ), b % 2 = 1 ∧
  (sum_fraction 2010).den = 2^a * b ∧ (a * b) / 10 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_fraction_denominator_form_main_result_l3_389


namespace NUMINAMATH_CALUDE_jenson_shirts_per_day_l3_321

/-- The number of shirts Jenson makes per day -/
def shirts_per_day : ℕ := sorry

/-- The number of pairs of pants Kingsley makes per day -/
def pants_per_day : ℕ := 5

/-- The amount of fabric used for one shirt (in yards) -/
def fabric_per_shirt : ℕ := 2

/-- The amount of fabric used for one pair of pants (in yards) -/
def fabric_per_pants : ℕ := 5

/-- The total amount of fabric needed every 3 days (in yards) -/
def total_fabric_3days : ℕ := 93

theorem jenson_shirts_per_day :
  shirts_per_day = 3 ∧
  shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pants = total_fabric_3days / 3 :=
sorry

end NUMINAMATH_CALUDE_jenson_shirts_per_day_l3_321


namespace NUMINAMATH_CALUDE_marks_radiator_cost_l3_360

/-- The total cost of replacing a car radiator -/
def total_cost (work_duration : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_duration * hourly_rate + part_cost

/-- Proof that Mark's total cost for replacing his car radiator is $300 -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marks_radiator_cost_l3_360


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3_369

def a : Fin 2 → ℝ := ![3, 2]

def v1 : Fin 2 → ℝ := ![3, -2]
def v2 : Fin 2 → ℝ := ![2, 3]
def v3 : Fin 2 → ℝ := ![-4, 6]
def v4 : Fin 2 → ℝ := ![-3, 2]

def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

theorem perpendicular_vectors :
  dot_product a v1 ≠ 0 ∧
  dot_product a v2 ≠ 0 ∧
  dot_product a v3 = 0 ∧
  dot_product a v4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3_369


namespace NUMINAMATH_CALUDE_vasya_numbers_l3_318

theorem vasya_numbers : ∃ (x y : ℝ), x + y = x * y ∧ x + y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l3_318


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3_342

theorem unique_solution_exists (x y z : ℝ) : 
  (x / 6) * 12 = 11 ∧ 
  4 * (x - y) + 5 = 11 ∧ 
  Real.sqrt z = (3 * x + y / 2) ^ 2 →
  x = 5.5 ∧ y = 4 ∧ z = 117132.0625 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3_342


namespace NUMINAMATH_CALUDE_jackson_flight_distance_l3_300

theorem jackson_flight_distance (beka_miles jackson_miles : ℕ) 
  (h1 : beka_miles = 873)
  (h2 : beka_miles = jackson_miles + 310) : 
  jackson_miles = 563 := by
sorry

end NUMINAMATH_CALUDE_jackson_flight_distance_l3_300


namespace NUMINAMATH_CALUDE_segment_combination_uniqueness_l3_332

theorem segment_combination_uniqueness :
  ∃! (x y : ℕ), 7 * x + 12 * y = 100 :=
by sorry

end NUMINAMATH_CALUDE_segment_combination_uniqueness_l3_332


namespace NUMINAMATH_CALUDE_roots_and_coefficients_l3_329

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots
def is_root (a b c x : ℝ) : Prop := quadratic_equation a b c x

-- Theorem statement
theorem roots_and_coefficients (a b c X₁ X₂ : ℝ) 
  (ha : a ≠ 0) 
  (hX₁ : is_root a b c X₁) 
  (hX₂ : is_root a b c X₂) 
  (hX₁₂ : X₁ ≠ X₂) : 
  (X₁ + X₂ = -b / a) ∧ (X₁ * X₂ = c / a) := by
  sorry

end NUMINAMATH_CALUDE_roots_and_coefficients_l3_329


namespace NUMINAMATH_CALUDE_probability_mixed_selection_l3_310

/- Define the number of male and female students -/
def num_male : ℕ := 3
def num_female : ℕ := 4

/- Define the total number of students -/
def total_students : ℕ := num_male + num_female

/- Define the number of volunteers to be selected -/
def num_volunteers : ℕ := 3

/- Theorem stating the probability of selecting both male and female students -/
theorem probability_mixed_selection :
  (1 : ℚ) - (Nat.choose num_male num_volunteers + Nat.choose num_female num_volunteers : ℚ) / 
  (Nat.choose total_students num_volunteers : ℚ) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_mixed_selection_l3_310


namespace NUMINAMATH_CALUDE_shaded_area_five_circles_plus_one_l3_366

/-- The area of the shaded region formed by five circles of radius 5 units
    intersecting at the origin, with an additional circle creating 10 similar sectors. -/
theorem shaded_area_five_circles_plus_one (r : ℝ) (n : ℕ) : 
  r = 5 → n = 10 → (n : ℝ) * (π * r^2 / 4 - r^2 / 2) = 62.5 * π - 125 := by
  sorry

#check shaded_area_five_circles_plus_one

end NUMINAMATH_CALUDE_shaded_area_five_circles_plus_one_l3_366


namespace NUMINAMATH_CALUDE_factor_x8_minus_81_l3_337

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x + Real.sqrt 3) * (x - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_x8_minus_81_l3_337


namespace NUMINAMATH_CALUDE_factor_expression_l3_370

theorem factor_expression (x : ℝ) : 75 * x^19 + 225 * x^38 = 75 * x^19 * (1 + 3 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3_370


namespace NUMINAMATH_CALUDE_trainees_seating_theorem_l3_333

/-- Represents the number of trainees and plates -/
def n : ℕ := 67

/-- Represents the number of correct seatings after rotating i positions -/
def correct_seatings (i : ℕ) : ℕ := sorry

theorem trainees_seating_theorem :
  ∃ i : ℕ, i > 0 ∧ i < n ∧ correct_seatings i ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_trainees_seating_theorem_l3_333


namespace NUMINAMATH_CALUDE_union_equals_real_complement_intersect_B_l3_338

-- Define sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 5}

-- Theorem for A ∪ B = ℝ
theorem union_equals_real : A ∪ B = Set.univ := by sorry

-- Theorem for (∁ₐA) ∩ B = {x | x < 2}
theorem complement_intersect_B : 
  (Set.univ \ A) ∩ B = {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_union_equals_real_complement_intersect_B_l3_338


namespace NUMINAMATH_CALUDE_max_angle_A1MC1_is_pi_over_2_l3_308

/-- Represents a right square prism -/
structure RightSquarePrism where
  base_side : ℝ
  height : ℝ
  height_eq_half_base : height = base_side / 2

/-- Represents a point on an edge of the prism -/
structure EdgePoint where
  x : ℝ
  valid : 0 ≤ x ∧ x ≤ 1

/-- Calculates the angle A₁MC₁ given a point M on edge AB -/
def angle_A1MC1 (prism : RightSquarePrism) (M : EdgePoint) : ℝ := sorry

/-- Theorem: The maximum value of angle A₁MC₁ in a right square prism is π/2 -/
theorem max_angle_A1MC1_is_pi_over_2 (prism : RightSquarePrism) :
  ∃ M : EdgePoint, ∀ N : EdgePoint, angle_A1MC1 prism M ≥ angle_A1MC1 prism N ∧ 
  angle_A1MC1 prism M = π / 2 :=
sorry

end NUMINAMATH_CALUDE_max_angle_A1MC1_is_pi_over_2_l3_308
