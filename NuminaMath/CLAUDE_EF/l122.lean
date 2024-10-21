import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_existence_condition_l122_12246

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then x + 2 else x^2

-- Theorem for monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ a ∈ Set.Ici 2 := by
  sorry

-- Theorem for the existence of x₂
theorem existence_condition (a : ℝ) :
  (∀ x₁ : ℝ, x₁ < a → ∃ x₂ : ℝ, x₂ ≥ a ∧ f a x₁ + f a x₂ = 0) ↔ a ∈ Set.Iic (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_existence_condition_l122_12246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_sufficient_condition_l122_12267

-- Define the types for planes and lines
structure Plane : Type
structure Line : Type

-- Define the parallelism relation
axiom parallel : Plane → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom parallel_lines : Line → Line → Prop

-- Define the "lies in" relation for lines and planes
axiom lies_in : Line → Plane → Prop

-- Define the intersection relation for lines
axiom intersect : Line → Line → Prop

-- State the theorem
theorem parallel_planes_sufficient_condition 
  (α β : Plane) (m n l₁ l₂ : Line) : 
  α ≠ β → 
  m ≠ n → 
  lies_in m α → 
  lies_in n α → 
  lies_in l₁ β → 
  lies_in l₂ β → 
  intersect l₁ l₂ → 
  parallel_lines m l₁ → 
  parallel_lines n l₂ → 
  parallel α β :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_sufficient_condition_l122_12267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_property_l122_12282

theorem largest_divisor_property : ∀ k : ℕ, 
  (∀ x y : ℤ, (k : ℤ) ∣ (x * y + 1) → (k : ℤ) ∣ (x + y)) → k ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_property_l122_12282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_l122_12293

/-- The time (in hours) it takes to fill a tank with a pump and a leak -/
noncomputable def fillTime (pumpFillTime leakEmptyTime : ℝ) : ℝ :=
  1 / (1 / pumpFillTime - 1 / leakEmptyTime)

/-- Theorem: Given a pump that can fill a tank in 4 hours and a leak that can empty
    the tank in 5 hours, it takes 20 hours to fill the tank with both the pump and leak active -/
theorem fill_time_with_leak :
  fillTime 4 5 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_l122_12293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_for_obtuse_l122_12294

theorem cos_half_angle_for_obtuse (α : ℝ) 
  (h1 : Real.sin α = (4 / 9) * Real.sqrt 2) 
  (h2 : π / 2 < α ∧ α < π) : 
  Real.cos (α / 2) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_for_obtuse_l122_12294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_exists_l122_12271

noncomputable def line (x : ℝ) : ℝ × ℝ := (x, 3 * x + 2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / norm_squared) • w

theorem constant_projection_exists :
  ∃ (w : ℝ × ℝ), ∀ (x : ℝ), projection (line x) w = (-3/5, 1/5) := by
  sorry

#check constant_projection_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_exists_l122_12271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_draws_is_twelve_l122_12248

/-- The expected number of draws until two consecutive red balls are drawn -/
noncomputable def expected_draws : ℝ := 12

/-- The probability of drawing a red ball -/
noncomputable def prob_red : ℝ := 1/3

/-- The probability of drawing a non-red ball -/
noncomputable def prob_not_red : ℝ := 2/3

/-- The set of possible ball colors -/
inductive BallColor
| Red
| Yellow
| Blue

/-- A function that simulates drawing a ball -/
def draw_ball : BallColor := sorry

/-- A function that checks if two consecutive draws are red -/
def two_consecutive_red (draw1 draw2 : BallColor) : Prop :=
  draw1 = BallColor.Red ∧ draw2 = BallColor.Red

/-- The main theorem stating that the expected number of draws is 12 -/
theorem expected_draws_is_twelve :
  expected_draws = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_draws_is_twelve_l122_12248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l122_12242

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x else x^2

theorem f_composition_negative_two : f (f (-2)) = 8 := by
  -- Evaluate f(-2)
  have h1 : f (-2) = 4 := by
    simp [f]
    norm_num
  
  -- Evaluate f(4)
  have h2 : f 4 = 8 := by
    simp [f]
    norm_num
  
  -- Combine the steps
  calc f (f (-2)) = f 4 := by rw [h1]
                  _ = 8 := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l122_12242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l122_12279

-- Define the function
def f (x : ℝ) : ℝ := x - x^2

-- State the theorem
theorem area_under_curve : 
  (∫ x in Set.Icc 0 1, f x) = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l122_12279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_g_l122_12224

-- Define the function g as noncomputable
noncomputable def g (t : ℝ) : ℝ := 2 * t / (1 - 2 * t)

-- State the theorem
theorem inverse_function_g (w z : ℝ) (hw : w ≠ 1/2) (hz : z = g w) : 
  w = z / (2 * (1 + z)) := by
  -- The proof is omitted for now
  sorry

-- Example usage (optional)
example (w z : ℝ) (hw : w ≠ 1/2) (hz : z = g w) : 
  w = z / (2 * (1 + z)) := by
  exact inverse_function_g w z hw hz

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_g_l122_12224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l122_12217

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 6 = 0

/-- Circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- A point P on the circle C -/
def point_on_C (P : ℝ × ℝ) : Prop := circle_C P.1 P.2

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (P A B : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum area of triangle PAB -/
theorem max_area_triangle_PAB :
  ∃ (A B P : ℝ × ℝ),
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    point_on_C P ∧
    (∀ (Q : ℝ × ℝ), point_on_C Q →
      area_triangle P A B ≤ (27 * Real.sqrt 3) / 4) ∧
    (∃ (R : ℝ × ℝ), point_on_C R ∧
      area_triangle R A B = (27 * Real.sqrt 3) / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l122_12217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l122_12205

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l122_12205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parabola_vertices_l122_12239

/-- The distance between the vertices of the parabolas y = x^2 - 4x + 5 and y = x^2 + 2x + 4 is √13 -/
theorem distance_between_parabola_vertices :
  let f (x : ℝ) := x^2 - 4*x + 5
  let g (x : ℝ) := x^2 + 2*x + 4
  let vertex_f : ℝ × ℝ := (2, f 2)
  let vertex_g : ℝ × ℝ := (-1, g (-1))
  Real.sqrt ((vertex_f.1 - vertex_g.1)^2 + (vertex_f.2 - vertex_g.2)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parabola_vertices_l122_12239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weight_theorem_l122_12263

-- Define the weight of James's bag
noncomputable def james_bag_weight : ℝ := 18

-- Define the weight ratio of Oliver's bag to James's bag
noncomputable def oliver_bag_ratio : ℝ := 1/6

-- Define the number of Oliver's bags
def oliver_bag_count : ℕ := 2

-- Define the weight ratio of Elizabeth's bag to James's bag
noncomputable def elizabeth_bag_ratio : ℝ := 3/4

-- Theorem statement
theorem combined_weight_theorem : 
  oliver_bag_count * (oliver_bag_ratio * james_bag_weight) + 
  elizabeth_bag_ratio * james_bag_weight = 19.5 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weight_theorem_l122_12263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_in_sequence_ratios_l122_12280

def sequenceQ (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => if n % 2 = 0 then sequenceQ (n / 2) else sequenceQ (n / 2) + sequenceQ (n / 2 + 1)

theorem rational_in_sequence_ratios :
  ∀ (r : ℚ), r > 0 → ∃ (n : ℕ), n ≥ 1 ∧ r = sequenceQ (n - 1) / sequenceQ n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_in_sequence_ratios_l122_12280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_n_correct_l122_12238

/-- An arithmetic sequence with common difference less than -1 -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  d_lt_neg_one : d < -1

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

/-- The value of n when S_n reaches its smallest positive value -/
def smallest_positive_n : ℕ := 20

theorem smallest_positive_n_correct (seq : ArithmeticSequence) :
  let n := smallest_positive_n
  (S seq n > 0) ∧ (∀ m, m > n → S seq m ≤ S seq n) := by
  sorry

#eval smallest_positive_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_n_correct_l122_12238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l122_12225

theorem greatest_integer_fraction : 
  ⌊(2^50 + 5^50 : ℝ) / (2^47 + 5^47 : ℝ)⌋ = 124 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l122_12225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extremum_points_condition_l122_12219

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + x + 1

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Theorem statement
theorem two_extremum_points_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0) ↔ a < -1 ∨ a > 1 :=
by
  sorry

#check two_extremum_points_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extremum_points_condition_l122_12219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_sqrt_l122_12277

theorem count_integers_between_sqrt : 
  (Finset.filter (fun x => 3 * 3 < x ∧ x < 5 * 5) (Finset.range 25)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_between_sqrt_l122_12277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l122_12286

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem tangent_line_sum : 
  (∀ x : ℝ, (deriv f) 1 * x + (f 1 - (deriv f) 1) = (1/2) * x + 2) → 
  f 1 + (deriv f) 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l122_12286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_and_third_term_l122_12261

/-- Geometric sequence with positive terms -/
noncomputable def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ a n > 0

/-- Sum of first n terms of a geometric sequence -/
noncomputable def GeometricSum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a 1) * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio_and_third_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_seq : GeometricSequence a q)
  (h_incr : ∀ n, a (n + 1) > a n)
  (h_arithmetic_mean : (5/3) * (a 3) = (a 2 + a 4) / 2)
  (h_sum : GeometricSum a q 5 = 484) :
  q = 3 ∧ a 3 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_and_third_term_l122_12261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l122_12221

noncomputable def f (x : ℝ) := 4 - x^2 - 2 * Real.log (x^2 + 1) / Real.log 5

theorem inequality_solution_set :
  {x : ℝ | x > 0 ∧ f (Real.log x) + 3 * f (Real.log (1/x)) + 8 ≤ 0} =
  {x : ℝ | 0 < x ∧ x ≤ Real.exp (-2) ∨ x ≥ Real.exp 2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l122_12221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l122_12204

noncomputable def binomial_expansion (x : ℝ) : ℝ := 2 * x + 1 / (3 * x)

theorem coefficient_of_x_squared (n : ℕ) :
  (∀ x : ℝ, x ≠ 0 → (binomial_expansion x)^n = 729) →
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → (binomial_expansion x)^n = c * x^2 + (binomial_expansion x)^n - c * x^2 ∧ c = 160 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l122_12204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_problem_l122_12275

theorem sqrt_sum_problem : Real.sqrt 9 + ((-1 : ℝ) ^ (1/3 : ℝ)) - Real.sqrt 0 + Real.sqrt (1/4) = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_problem_l122_12275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_stream_speed_l122_12252

/-- The speed of a stream given a swimmer's still water speed and the ratio of upstream to downstream time -/
noncomputable def stream_speed (still_water_speed : ℝ) (upstream_downstream_ratio : ℝ) : ℝ :=
  still_water_speed * (upstream_downstream_ratio - 1) / (upstream_downstream_ratio + 1)

/-- Theorem stating that for a swimmer with 4.5 km/h still water speed and 2:1 upstream to downstream time ratio, the stream speed is 1.5 km/h -/
theorem swimmer_stream_speed :
  stream_speed 4.5 2 = 1.5 := by
  -- Unfold the definition of stream_speed
  unfold stream_speed
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_stream_speed_l122_12252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_minimal_perimeter_l122_12249

/-- A triangle with base b, height h, and vertex at (x, h) -/
structure Triangle where
  b : ℝ
  h : ℝ
  x : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.h

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  Real.sqrt (t.x^2 + t.h^2) + Real.sqrt ((t.b - t.x)^2 + t.h^2) + t.b

/-- An isosceles triangle with base b and height h -/
noncomputable def isoscelesTriangle (b h : ℝ) : Triangle :=
  { b := b, h := h, x := b/2 }

theorem isosceles_minimal_perimeter (b h : ℝ) (hb : b > 0) (hh : h > 0) :
  ∀ t : Triangle, t.b = b → area t = area (isoscelesTriangle b h) →
    perimeter (isoscelesTriangle b h) ≤ perimeter t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_minimal_perimeter_l122_12249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_square_l122_12201

/-- The area of a rectangle containing a square with perimeter 16 cm, 
    where the probability of a random point in the rectangle not being in the square 
    is 0.4666666666666667, is approximately 30 square centimeters. -/
theorem rectangle_area_with_square (rectangle_area : ℝ) (square_perimeter : ℝ) 
    (probability_not_in_square : ℝ) : 
  square_perimeter = 16 →
  probability_not_in_square = 0.4666666666666667 →
  ∃ ε > 0, |rectangle_area - 30| < ε := by
  sorry

#check rectangle_area_with_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_square_l122_12201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_tangent_value_l122_12216

theorem complex_tangent_value (θ : ℝ) (z : ℂ) (k : ℤ) :
  z = (3 + 4 * Complex.I) * (Complex.cos θ + Complex.I * Complex.sin θ) →
  z.im = 0 →
  θ ≠ k * π + π / 2 →
  Complex.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_tangent_value_l122_12216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_proof_l122_12211

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := 5^(-(1/2 : ℝ))

theorem ordering_proof : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_proof_l122_12211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_philosophy_value_theorem_l122_12212

/-- Represents the study of philosophy -/
def PhilosophyStudy : Type := Unit

/-- Represents the art of guiding people to live better lives -/
def ArtOfGuidingLives : Type := Unit

/-- UNESCO has designated a World Philosophy Day -/
axiom world_philosophy_day : Prop

/-- We should value the study of philosophy -/
axiom value_philosophy_study : Prop

/-- Philosophy is the art of guiding people to live better lives -/
axiom philosophy_is_art_of_guiding : PhilosophyStudy → ArtOfGuidingLives

/-- Instance for Inhabited ArtOfGuidingLives -/
instance : Inhabited ArtOfGuidingLives := ⟨Unit.unit⟩

/-- The reason we should value the study of philosophy -/
def reason_to_value_philosophy : Prop :=
  ∃ (p : PhilosophyStudy), philosophy_is_art_of_guiding p = default

theorem philosophy_value_theorem :
  world_philosophy_day →
  value_philosophy_study →
  reason_to_value_philosophy :=
by
  sorry

#check philosophy_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_philosophy_value_theorem_l122_12212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_distributive_laws_l122_12257

noncomputable def avg (a b : ℝ) : ℝ := (a + b) / 2

theorem avg_distributive_laws (x y z : ℝ) : 
  (x + avg y z = avg (x + y) (x + z)) ∧ 
  (avg x (avg y z) = avg (avg x y) (avg x z)) ∧
  (avg x (y + z) ≠ avg x y + avg x z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_distributive_laws_l122_12257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_2014_heze_simulation_l122_12298

noncomputable def area (t : Set (ℝ × ℝ)) : ℝ := sorry
def congruent (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem problem_2014_heze_simulation :
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x * y = 1 → x = 1 / y) ∧
  (∃ t1 t2 : Set (ℝ × ℝ), area t1 = area t2 ∧ ¬ congruent t1 t2) ∧
  (∀ m : ℝ, (∀ x : ℝ, x^2 - 2*x + m ≠ 0) → m > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_2014_heze_simulation_l122_12298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_second_discount_l122_12268

/-- Given an original price, two successive discounts, and a final price,
    calculate the second discount percentage. -/
noncomputable def second_discount_percentage (original_price first_discount final_price : ℝ) : ℝ :=
  100 * (1 - final_price / (original_price * (1 - first_discount / 100)))

/-- The second discount percentage for the saree problem -/
theorem saree_second_discount :
  second_discount_percentage 390 14 285.09 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_second_discount_l122_12268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_exists_l122_12243

/-- The speed of a boat in still water (mph) -/
def boat_speed : ℝ := 12

/-- The distance traveled by the boat (miles) -/
def distance : ℝ := 60

/-- The time difference between upstream and downstream trips (hours) -/
def time_difference : ℝ := 2

/-- Theorem stating the existence of a stream speed satisfying the given conditions -/
theorem stream_speed_exists : ∃ v : ℝ, 
  v > 0 ∧ 
  distance / (boat_speed + v) + time_difference = distance / (boat_speed - v) ∧
  abs (v - 2.31) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_exists_l122_12243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l122_12209

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - (1/2)*x) / (x^2 - x + 1)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc (2/5 : ℝ) 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l122_12209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l122_12236

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 5 = 0

def chord_length (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : ℝ := 6

def passes_through (l : ℝ → ℝ → Prop) : Prop := l 2 4

theorem line_equation (l : ℝ → ℝ → Prop) 
  (h1 : passes_through l)
  (h2 : chord_length l circle_equation = 6) :
  (∀ x y, l x y ↔ x - 2 = 0) ∨ 
  (∀ x y, l x y ↔ 3*x - 4*y + 10 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l122_12236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_fifth_terms_l122_12244

def geometric_sequence (a₀ : ℚ) (r : ℚ) : ℕ → ℚ :=
  λ n ↦ a₀ * r^n

theorem sum_of_fourth_and_fifth_terms :
  let seq := geometric_sequence 4096 (1/4)
  (seq 3 + seq 4 = 80) ∧
  (seq 0 = 4096) ∧
  (seq 1 = 1024) ∧
  (seq 2 = 256) ∧
  (seq 5 = 4) ∧
  (seq 6 = 1) ∧
  (seq 7 = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_fifth_terms_l122_12244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_B_l122_12250

-- Define set A
def A : Finset ℕ := {1, 2, 3}

-- Define set B
def B : Finset ℕ := Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product A)

-- Theorem statement
theorem number_of_subsets_of_B : Finset.card (Finset.powerset B) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_B_l122_12250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rectangle_in_prime_array_l122_12233

theorem no_rectangle_in_prime_array (p : ℕ) (hp : Nat.Prime p) :
  ∃ (selected : Finset ((Fin (p^2)) × (Fin (p^2)))),
    (selected.card = p^3) ∧
    (∀ (a b c d : (Fin (p^2)) × (Fin (p^2))),
      a ∈ selected → b ∈ selected → c ∈ selected → d ∈ selected →
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
      ¬(((a.1 = b.1 ∧ c.1 = d.1) ∨ (a.1 = c.1 ∧ b.1 = d.1)) ∧
        ((a.2 = b.2 ∧ c.2 = d.2) ∨ (a.2 = c.2 ∧ b.2 = d.2)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rectangle_in_prime_array_l122_12233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_positive_l122_12253

theorem function_always_positive (a : ℝ) (h : a ∈ Set.Icc (-1 : ℝ) 1) :
  (∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (∀ x : ℝ, x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_positive_l122_12253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l122_12222

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Checks if a point lies on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The ellipse with given properties has the specified equation -/
theorem ellipse_equation :
  ∀ (e : Ellipse),
    e.a > 0 ∧ e.b > 0 →
    isOnEllipse ⟨10, 3⟩ e →
    distance ⟨10, 3⟩ ⟨0, 0⟩ + distance ⟨10, 3⟩ ⟨0, 4⟩ = 2 * e.a →
    distance ⟨0, 0⟩ ⟨0, 4⟩ = 4 →
    e = Ellipse.mk ((Real.sqrt 109 + Real.sqrt 101) / 2)
                   (Real.sqrt (((Real.sqrt 109 + Real.sqrt 101) / 2)^2 - 4))
                   0
                   2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l122_12222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_sine_l122_12276

theorem triangle_special_angle_sine (p : ℝ) :
  0 < p → p < π / 2 →
  Real.sin (π - 2 * p) = 1 / 2 :=
by
  intros hp_pos hp_lt_pi_half
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_sine_l122_12276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_gua_theorem_l122_12245

/-- A polynomial with complex coefficients -/
def ComplexPolynomial := List ℂ

/-- The number of non-real roots of a polynomial -/
noncomputable def nonRealRoots (p : ComplexPolynomial) : ℕ := sorry

/-- Check if a polynomial has 2m consecutive zero coefficients -/
def hasMissingTerms (p : ComplexPolynomial) (m : ℕ) : Prop := sorry

/-- Check if a polynomial has 2m+1 consecutive zero coefficients enclosed by terms of different signs -/
def hasMissingTermsDiffSigns (p : ComplexPolynomial) (m : ℕ) : Prop := sorry

/-- Check if a polynomial has 2m+1 consecutive zero coefficients enclosed by terms of the same sign -/
def hasMissingTermsSameSign (p : ComplexPolynomial) (m : ℕ) : Prop := sorry

theorem de_gua_theorem (p : ComplexPolynomial) (m : ℕ) :
  (hasMissingTerms p m → nonRealRoots p ≥ 2*m) ∧
  (hasMissingTermsDiffSigns p m → nonRealRoots p ≥ 2*m) ∧
  (hasMissingTermsSameSign p m → nonRealRoots p ≥ 2*m + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_gua_theorem_l122_12245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latest_start_time_for_roasting_l122_12260

/-- Represents the time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the roasting time for a single turkey -/
def roastingTimePerTurkey (weight : Nat) (minutesPerPound : Nat) : Nat :=
  weight * minutesPerPound

/-- Calculates the total roasting time for all turkeys -/
def totalRoastingTime (numberOfTurkeys : Nat) (weightPerTurkey : Nat) (minutesPerPound : Nat) : Nat :=
  numberOfTurkeys * roastingTimePerTurkey weightPerTurkey minutesPerPound

/-- Converts minutes to hours and minutes -/
def minutesToTime (minutes : Nat) : Time :=
  { hours := minutes / 60, minutes := minutes % 60 }

/-- Subtracts a given number of hours from a time -/
def subtractHours (t : Time) (hours : Nat) : Time :=
  { hours := t.hours - hours, minutes := t.minutes }

theorem latest_start_time_for_roasting 
  (numberOfTurkeys : Nat) 
  (weightPerTurkey : Nat) 
  (minutesPerPound : Nat) 
  (dinnerTime : Time) :
  numberOfTurkeys = 2 →
  weightPerTurkey = 16 →
  minutesPerPound = 15 →
  dinnerTime = { hours := 18, minutes := 0 } →
  subtractHours dinnerTime (totalRoastingTime numberOfTurkeys weightPerTurkey minutesPerPound / 60) = { hours := 10, minutes := 0 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_latest_start_time_for_roasting_l122_12260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_angle_in_specific_pyramid_l122_12214

/-- Regular triangular pyramid with specific proportions -/
structure RegularTriangularPyramid where
  /-- Length of the lateral edge -/
  a : ℝ
  /-- Distance between opposite edges -/
  d : ℝ
  /-- Condition: distance between opposite edges is 3/64 of lateral edge -/
  h : d = (3 / 64) * a

/-- The angle between the apothem and the adjacent lateral face -/
noncomputable def apothemAngle (p : RegularTriangularPyramid) : ℝ :=
  Real.arcsin ((3 * Real.sqrt 39) / 25)

/-- Theorem: The angle between the apothem and the adjacent lateral face
    in the given regular triangular pyramid is arcsin(3√39/25) -/
theorem apothem_angle_in_specific_pyramid (p : RegularTriangularPyramid) :
  apothemAngle p = Real.arcsin ((3 * Real.sqrt 39) / 25) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_angle_in_specific_pyramid_l122_12214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_y_conditions_l122_12289

theorem real_y_conditions (x y : ℝ) : 
  (9 * y^2 + 6 * x * y + x + 12 = 0) ↔ (x ≤ -3 ∨ x ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_y_conditions_l122_12289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_polygon_l122_12283

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of vertices (true means red) -/
def Coloring (n : ℕ) := Fin n → Bool

/-- An axis of symmetry for a regular polygon -/
structure SymmetryAxis (n : ℕ) where
  axis : ℝ × ℝ → ℝ × ℝ

/-- Find the symmetric vertex index -/
def symmetricVertex (n : ℕ) (a : SymmetryAxis n) (i : Fin n) : Fin n :=
  ⟨(i.val + n / 2) % n, by 
    sorry  -- Proof that the result is within bounds
  ⟩

/-- Check if a coloring satisfies the symmetry condition for a given axis -/
def satisfiesSymmetryCondition (n : ℕ) (p : RegularPolygon n) (c : Coloring n) (a : SymmetryAxis n) : Prop :=
  ∀ i : Fin n, c i = true → c (symmetricVertex n a i) = false

/-- The main theorem -/
theorem smallest_symmetric_polygon :
  (∀ n : ℕ, n ≥ 14 →
    ∀ p : RegularPolygon n,
    ∀ c : Coloring n,
    (∃ (count : ℕ), count = 5 ∧ (∃ (red_vertices : Fin count → Fin n), ∀ i : Fin count, c (red_vertices i) = true)) →
    ∃ a : SymmetryAxis n, satisfiesSymmetryCondition n p c a) ∧
  (∀ n : ℕ, n < 14 →
    ∃ p : RegularPolygon n,
    ∃ c : Coloring n,
    (∃ (count : ℕ), count = 5 ∧ (∃ (red_vertices : Fin count → Fin n), ∀ i : Fin count, c (red_vertices i) = true)) ∧
    ∀ a : SymmetryAxis n, ¬satisfiesSymmetryCondition n p c a) :=
by
  sorry  -- The proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_polygon_l122_12283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oppositeAnglesEqualImpliesParallelogram3D_l122_12258

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a 3D quadrilateral
structure Quadrilateral3D where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

-- Define the concept of opposite angles being equal
def oppositeAnglesEqual (q : Quadrilateral3D) : Prop :=
  sorry -- We need to define angle measurement in 3D space, which is complex

-- Define a parallelogram in 3D space
def isParallelogram3D (q : Quadrilateral3D) : Prop :=
  (q.A.x - q.B.x = q.D.x - q.C.x) ∧
  (q.A.y - q.B.y = q.D.y - q.C.y) ∧
  (q.A.z - q.B.z = q.D.z - q.C.z) ∧
  (q.B.x - q.C.x = q.A.x - q.D.x) ∧
  (q.B.y - q.C.y = q.A.y - q.D.y) ∧
  (q.B.z - q.C.z = q.A.z - q.D.z)

-- Theorem statement
theorem oppositeAnglesEqualImpliesParallelogram3D (q : Quadrilateral3D) :
  oppositeAnglesEqual q → isParallelogram3D q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oppositeAnglesEqualImpliesParallelogram3D_l122_12258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l122_12284

/-- The compound interest function -/
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- The problem statement -/
theorem investment_problem (ε : ℝ) (h_ε : ε > 0) :
  ∃ (P : ℝ), P ≥ 0 ∧ P ≤ 39410 ∧ P ≥ 39409 ∧ 
  |compound_interest P 0.08 12 3 - 50000| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l122_12284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l122_12272

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

noncomputable def g (x : ℝ) : ℝ := x + 1/x^2

theorem max_value_of_f (p q : ℝ) :
  (∃ (x₀ : ℝ), x₀ ∈ Set.Icc 1 2 ∧
    (∀ (x : ℝ), x ∈ Set.Icc 1 2 → f p q x ≥ f p q x₀) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 1 2 → g x ≥ g x₀)) →
  (∃ (x_max : ℝ), x_max ∈ Set.Icc 1 2 ∧
    (∀ (x : ℝ), x ∈ Set.Icc 1 2 → f p q x ≤ f p q x_max) ∧
    f p q x_max = 4 - (5/2) * Real.rpow 2 (1/3) + Real.rpow 4 (1/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l122_12272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_takeoff_run_length_l122_12266

/-- Calculates the length of an airplane's takeoff run given uniform acceleration. -/
theorem airplane_takeoff_run_length
  (takeoff_time : ℝ)
  (takeoff_speed_kmh : ℝ)
  (h1 : takeoff_time = 15)
  (h2 : takeoff_speed_kmh = 100)
  (h3 : takeoff_speed_kmh > 0)
  : ∃ (run_length : ℝ), abs (run_length - 208) < 1 ∧ run_length > 0 := by
  sorry

#check airplane_takeoff_run_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_takeoff_run_length_l122_12266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l122_12274

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_ratio (t : Triangle) 
  (h1 : t.b + Real.cos t.C = 0)
  (h2 : Real.sin t.A = 2 * Real.sin (t.A + t.C)) :
  t.a / t.c = 2 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l122_12274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_iff_equal_distances_l122_12297

-- Define the quadrilateral ABCD and point P
variable (A B C D P : EuclideanSpace ℝ (Fin 2))

-- Define the property of being a convex quadrilateral
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the property of a quadrilateral being cyclic
def is_cyclic_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define that P is inside ABCD
def point_inside_quadrilateral (P A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define angle equality
def angle_eq (ABC DEF : EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Theorem statement
theorem cyclic_quadrilateral_iff_equal_distances 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_inside : point_inside_quadrilateral P A B C D)
  (h_angle1 : angle_eq (P, B, C) (D, B, A))
  (h_angle2 : angle_eq (P, D, C) (B, D, A)) :
  is_cyclic_quadrilateral A B C D ↔ dist A P = dist C P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_iff_equal_distances_l122_12297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l122_12247

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (α (n + 1) + 1 / b α n)

theorem b_4_less_than_b_7 (α : ℕ → ℕ) : b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l122_12247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_fifty_l122_12240

/-- The average speed of a car on route one, given specific conditions. -/
def average_speed_route_one : ℝ → Prop := fun x =>
  let route_one_distance : ℝ := 75
  let route_two_distance : ℝ := 90
  let speed_ratio : ℝ := 1.8
  let time_difference : ℝ := 0.5
  (route_one_distance / x) = (route_two_distance / (speed_ratio * x)) + time_difference

/-- Theorem stating that the average speed on route one is 50 km/h. -/
theorem average_speed_is_fifty : average_speed_route_one 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_fifty_l122_12240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l122_12231

theorem binomial_expansion_theorem (n : ℕ) (x : ℝ) :
  (Nat.choose n 2 * 6 = 36) →
  ∃ t : ℝ, t = 36 * x^2 ∧ t ∈ (Set.range (λ i : ℕ ↦ Nat.choose n i * x^(n-i) * (-Real.sqrt 6)^i)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l122_12231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_properties_l122_12251

def number : Nat := 1111111613
def result : Nat := 1111111631

-- Function to calculate the product of digits
def productOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) * productOfDigits (n / 10)

-- Function to check if a number contains zero
def containsZero (n : Nat) : Bool :=
  if n = 0 then true
  else if n < 10 then false
  else if n % 10 = 0 then true
  else containsZero (n / 10)

-- Function to count digits in a number
def countDigits (n : Nat) : Nat :=
  if n < 10 then 1 else 1 + countDigits (n / 10)

theorem number_properties :
  (countDigits number = 10) ∧
  (¬containsZero number) ∧
  (number + productOfDigits number = result) ∧
  (productOfDigits number = productOfDigits result) := by
  sorry

#eval number
#eval result
#eval productOfDigits number
#eval productOfDigits result
#eval countDigits number
#eval containsZero number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_properties_l122_12251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_pairing_theorem_l122_12262

/-- Represents the number of pairs of socks -/
def n : ℕ := sorry

/-- Probability that all pairs are successful -/
def prob_all_successful (n : ℕ) : ℚ :=
  (2^n * n.factorial) / (2*n).factorial

/-- Expected number of successful pairs -/
def expected_successful_pairs (n : ℕ) : ℚ :=
  n / (2*n - 1)

/-- Main theorem about sock pairing -/
theorem sock_pairing_theorem (h : n > 0) :
  prob_all_successful n = (2^n * n.factorial) / (2*n).factorial ∧
  expected_successful_pairs n = n / (2*n - 1) ∧
  expected_successful_pairs n > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_pairing_theorem_l122_12262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neva_speed_difference_l122_12207

/-- Represents Neva's speeds at different ages --/
structure NevaSpeed where
  young_distance : ℝ  -- distance cycled when young (in miles)
  young_time : ℝ      -- time taken when young (in minutes)
  old_distance : ℝ    -- distance walked when older (in miles)
  old_time : ℝ        -- time taken when older (in minutes)

/-- Calculate the difference in time per mile between Neva's current walking speed and her younger cycling speed --/
noncomputable def time_difference (speeds : NevaSpeed) : ℝ :=
  (speeds.old_time / speeds.old_distance) - (speeds.young_time / speeds.young_distance)

/-- Theorem stating the difference in time per mile --/
theorem neva_speed_difference :
  let speeds : NevaSpeed := {
    young_distance := 20,
    young_time := 2 * 60 + 45,  -- 2 hours and 45 minutes in minutes
    old_distance := 8,
    old_time := 3 * 60          -- 3 hours in minutes
  }
  time_difference speeds = 14.25 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_neva_speed_difference_l122_12207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_points_l122_12203

-- Define the function representing the difference between y = x and y = 2ln x
noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log x

-- State the theorem
theorem min_distance_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (A.1 = A.2 ∧ A.2 = 2 * Real.log A.1) ∧
    (B.1 = B.2 ∧ B.2 = 2 * Real.log B.1) ∧
    (∀ (P Q : ℝ × ℝ), 
      (P.1 = P.2 ∧ P.2 = 2 * Real.log P.1) →
      (Q.1 = Q.2 ∧ Q.2 = 2 * Real.log Q.1) →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 - 2 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_points_l122_12203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_common_denominator_l122_12281

theorem simplest_common_denominator (x y : ℝ) : 
  let f₁ := 1 / (x + y)
  let f₂ := 1 / (x - y)
  let f₃ := 1 / (x^2 - y^2)
  ∃ (a b c : ℝ), 
     a * (x + y) * (x - y) = f₁ ∧
     b * (x + y) * (x - y) = f₂ ∧
     c * (x + y) * (x - y) = f₃ ∧
     ∀ (d : ℝ), (∃ (e f g : ℝ), e * d = f₁ ∧ f * d = f₂ ∧ g * d = f₃) →
                d = (x + y) * (x - y) ∨ d = -(x + y) * (x - y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_common_denominator_l122_12281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l122_12291

/-- Sales equation relating daily sales volume y to sales price x -/
noncomputable def sales_equation (a : ℝ) (x : ℝ) : ℝ := a / (x - 4) + 100 * (8 - x)

/-- Daily profit as a function of sales price x -/
noncomputable def profit (a : ℝ) (x : ℝ) : ℝ := (x - 4) * (sales_equation a x)

theorem max_profit_theorem (a : ℝ) :
  (∀ x, 4 < x → x < 8 → sales_equation a x ≥ 0) →
  sales_equation a 6 = 220 →
  (∃ x_max, 4 < x_max ∧ x_max < 8 ∧
    (∀ x, 4 < x → x < 8 → profit a x ≤ profit a x_max) ∧
    x_max = 6 ∧ profit a x_max = 440) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l122_12291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_ratio_product_l122_12295

/-- Given a triangle ABC with points A', B', C' on sides BC, AC, AB respectively,
    and AA', BB', CC' concurrent at point O, prove that if the sum of the ratios
    AO/OA', BO/OB', and CO/OC' is 92, then their product is 94. -/
theorem triangle_concurrent_lines_ratio_product (A B C A' B' C' O : ℝ × ℝ) : 
  let triangle_area := λ p q r : ℝ × ℝ ↦ abs ((p.1 - r.1) * (q.2 - r.2) - (q.1 - r.1) * (p.2 - r.2)) / 2
  let on_side := λ p q r : ℝ × ℝ ↦ 
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • q + t • r
  let collinear := λ p q r : ℝ × ℝ ↦ 
    (q.1 - p.1) * (r.2 - p.2) = (r.1 - p.1) * (q.2 - p.2)
  A' ≠ B ∧ A' ≠ C ∧ on_side A' B C ∧
  B' ≠ A ∧ B' ≠ C ∧ on_side B' A C ∧
  C' ≠ A ∧ C' ≠ B ∧ on_side C' A B ∧
  collinear A A' O ∧ collinear B B' O ∧ collinear C C' O ∧
  (triangle_area B O C / triangle_area B A' C + 
   triangle_area C O A / triangle_area C B' A + 
   triangle_area A O B / triangle_area A C' B) = 92 →
  (triangle_area B O C / triangle_area B A' C) *
  (triangle_area C O A / triangle_area C B' A) *
  (triangle_area A O B / triangle_area A C' B) = 94 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_ratio_product_l122_12295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_a_range_l122_12223

/-- A quadratic function satisfying certain properties -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  a : ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  symmetry : ∀ x, f (2 + x) = f (2 - x)
  inequality : f 1 < f 0 ∧ f 0 ≤ f a

/-- The range of 'a' for a special quadratic function -/
theorem special_quadratic_a_range (sq : SpecialQuadratic) :
  sq.a ≤ 0 ∨ sq.a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_a_range_l122_12223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_8_eq_5048_l122_12296

/-- A monic polynomial of degree 7 satisfying specific conditions -/
noncomputable def r : ℝ → ℝ := sorry

/-- r is a monic polynomial of degree 7 -/
axiom r_monic_degree_7 : ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, 
  ∀ x, r x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀

/-- r satisfies the given conditions for k = 1, 2, 3, 4, 5, 6, 7 -/
axiom r_conditions : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 → r k = k

/-- Theorem: r(8) = 5048 -/
theorem r_8_eq_5048 : r 8 = 5048 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_8_eq_5048_l122_12296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_keith_pears_l122_12269

/-- Given that Jason picked 46 pears, Mike picked 12 pears, and the total number of pears picked was 105, prove that Keith picked 47 pears. -/
theorem keith_pears (jason_pears mike_pears total_pears keith_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : mike_pears = 12)
  (h3 : total_pears = 105)
  (h4 : total_pears = jason_pears + mike_pears + keith_pears) :
  keith_pears = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_keith_pears_l122_12269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_theorem_l122_12227

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem function_and_inequality_theorem :
  (∀ x : ℝ, f (x + 1) = x^2 - (1/3) * f 3) →
  (∀ x : ℝ, f x = x^2 - 2*x) ∧
  (∀ x ∈ Set.Ioo (-2 : ℝ) (-1/2 : ℝ),
    ∀ a : ℝ, (f a + 4*a < (a + 2) * f (x^2)) ↔ a ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_theorem_l122_12227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l122_12265

def U : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {1, 3}
def B : Finset ℕ := {3, 5}

def complement (s : Finset ℕ) : Finset ℕ := U \ s

theorem complement_of_union :
  complement (A ∪ B) = {0, 2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l122_12265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_roots_zero_l122_12237

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The value of a cubic polynomial at a given point -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The sum of the roots of a cubic polynomial -/
noncomputable def CubicPolynomial.sumRoots (p : CubicPolynomial) : ℝ := -p.b / p.a

theorem sum_roots_zero (Q : CubicPolynomial) 
    (h : ∀ x : ℝ, Q.eval (x^4 + 2*x) ≥ Q.eval (x^3 + 2)) : 
  Q.sumRoots = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_roots_zero_l122_12237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_group_exists_l122_12290

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with n players -/
structure Tournament (n : ℕ) where
  /-- The result of each match, indexed by pairs of distinct players -/
  results : Fin n → Fin n → MatchResult
  /-- No draws -/
  no_draws : ∀ i j, i ≠ j → (results i j = MatchResult.Win ↔ results j i = MatchResult.Loss)
  /-- Each player plays against every other player -/
  complete : ∀ i j, i ≠ j → (results i j = MatchResult.Win ∨ results i j = MatchResult.Loss)

/-- A group of four players is ordered if one player beats the other three and one loses to the other three -/
def is_ordered (t : Tournament 8) (a b c d : Fin 8) : Prop :=
  (∀ x, x ∈ ({b, c, d} : Set (Fin 8)) → t.results a x = MatchResult.Win ∧ t.results x b = MatchResult.Win) ∨
  (∀ x, x ∈ ({a, c, d} : Set (Fin 8)) → t.results b x = MatchResult.Win ∧ t.results x a = MatchResult.Win) ∨
  (∀ x, x ∈ ({a, b, d} : Set (Fin 8)) → t.results c x = MatchResult.Win ∧ t.results x a = MatchResult.Win) ∨
  (∀ x, x ∈ ({a, b, c} : Set (Fin 8)) → t.results d x = MatchResult.Win ∧ t.results x a = MatchResult.Win)

/-- In any tournament with 8 players, there always exists an ordered group of four players -/
theorem ordered_group_exists (t : Tournament 8) :
  ∃ a b c d : Fin 8, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ is_ordered t a b c d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_group_exists_l122_12290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_ab_l122_12256

theorem max_value_of_ab (a b : ℝ) (h : a^2 + 2*b^2 = 1) : 
  (∀ x y : ℝ, x^2 + 2*y^2 = 1 → x*y ≤ a*b) ∧ a*b = Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_ab_l122_12256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ariana_total_owed_l122_12292

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 12) ^ (12 * time)

/-- Calculates the total amount for a bill with flat late fee -/
def flatLateFee (principal : ℝ) (feePerMonth : ℝ) (months : ℕ) : ℝ :=
  principal + feePerMonth * (months : ℝ)

/-- Calculates the total amount for a bill with doubling late fee -/
def doublingLateFee (principal : ℝ) (initialFee : ℝ) (months : ℕ) : ℝ :=
  principal + initialFee * ((2 : ℝ) ^ months - 1)

/-- Represents Ariana's bill payment situation -/
structure BillPayment where
  bill1Amount : ℝ
  bill1Rate : ℝ
  bill1Months : ℝ
  bill2Amount : ℝ
  bill2Fee : ℝ
  bill2Months : ℕ
  bill3Amount : ℝ
  bill3InitialFee : ℝ
  bill3Months : ℕ

/-- Calculates the total amount owed for all bills -/
noncomputable def totalAmountOwed (payment : BillPayment) : ℝ :=
  compoundInterest payment.bill1Amount payment.bill1Rate payment.bill1Months +
  flatLateFee payment.bill2Amount payment.bill2Fee payment.bill2Months +
  doublingLateFee payment.bill3Amount payment.bill3InitialFee payment.bill3Months

/-- Theorem stating that Ariana owes $1779.031 -/
theorem ariana_total_owed :
  let payment : BillPayment := {
    bill1Amount := 200,
    bill1Rate := 0.1,
    bill1Months := 0.25,
    bill2Amount := 130,
    bill2Fee := 50,
    bill2Months := 8,
    bill3Amount := 444,
    bill3InitialFee := 40,
    bill3Months := 4
  }
  totalAmountOwed payment = 1779.031 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ariana_total_owed_l122_12292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l122_12287

theorem sequence_properties (a : ℚ) (p : ℚ) (h1 : p ≠ 0) (h2 : p ≠ -1) :
  let a_n : ℕ → ℚ := λ n => if n = 1 then a else (a / p) * ((p + 1) / p) ^ (n - 2)
  ∀ n : ℕ, (Finset.range n).sum (λ i => a_n (i + 1)) = p * a_n (n + 1) →
  (∀ k : ℕ, ∃ d : ℚ, a_n (k + 1) + a_n (k + 3) = 2 * a_n (k + 2) → 
    ((p = -1/3 ∧ d = 9*a*(2^(k-1))) ∨ 
     (p = -2/3 ∧ d = (9*a/8)*((1/2)^(k-1))))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l122_12287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_range_l122_12226

-- Define the line equation
def line (a b x y : ℝ) : Prop := 2 * a * x - b * y + 14 = 0

-- Define the function
noncomputable def f (m x : ℝ) : ℝ := m^(x + 1) + 1

-- Define the circle equation
def in_circle (a b x y : ℝ) : Prop := (x - a + 1)^2 + (y + b - 2)^2 ≤ 25

-- State the theorem
theorem fixed_point_range (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0 ∧ m ≠ 1) :
  (∃ x y : ℝ, line a b x y ∧ y = f m x ∧ in_circle a b x y) →
  3/4 ≤ b/a ∧ b/a ≤ 4/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_range_l122_12226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_spending_problem_l122_12259

theorem mark_spending_problem (initial_money : ℝ) 
  (h1 : initial_money = 180)
  (h2 : initial_money > 0)
  (first_store_spending : ℝ)
  (h3 : first_store_spending = initial_money / 2 + 14)
  (second_store_spending : ℝ)
  (h4 : second_store_spending = initial_money - first_store_spending + 16)
  (h5 : first_store_spending + second_store_spending = initial_money) :
  second_store_spending - 16 = initial_money / 3 := by
  sorry

#check mark_spending_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_spending_problem_l122_12259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_frequency_is_twelve_l122_12218

/-- Given a sample size and a ratio of heights for rectangles in a frequency distribution histogram,
    calculate the frequency of the second group. -/
def frequency_of_second_group (sample_size : ℕ) (ratio : List ℕ) : ℕ :=
  let total_parts := ratio.sum
  let second_group_parts := ratio.get! 1
  (sample_size * second_group_parts) / total_parts

/-- Theorem stating that for a sample size of 30 and ratio 2:4:3:1,
    the frequency of the second group is 12. -/
theorem second_group_frequency_is_twelve :
  frequency_of_second_group 30 [2, 4, 3, 1] = 12 := by
  sorry

#eval frequency_of_second_group 30 [2, 4, 3, 1]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_frequency_is_twelve_l122_12218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_festival_proof_l122_12220

/-- Represents the number of contestants who caught n fish -/
def contestants (n : ℕ) : ℕ := sorry

/-- The total number of contestants -/
def total_contestants : ℕ := Finset.sum (Finset.range 16) contestants

/-- The total number of fish caught -/
def total_fish : ℕ := sorry

theorem fishing_festival_proof :
  -- The winner caught 15 fish
  contestants 15 = 1 →
  -- Those who caught 3 or more fish averaged 6 fish each
  Finset.sum (Finset.range 13) (λ n => (n + 3) * contestants (n + 3)) = 
    6 * (Finset.sum (Finset.range 13) (λ n => contestants (n + 3))) →
  -- Those who caught 12 or fewer fish averaged 5 fish each
  Finset.sum (Finset.range 13) (λ n => n * contestants n) = 
    5 * (Finset.sum (Finset.range 13) contestants) →
  -- The total number of fish caught is 127
  total_fish = 127 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_festival_proof_l122_12220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equation_l122_12228

/-- The equation of the angle bisector of ∠Q in a triangle with vertices 
    P = (-7, 6), Q = (-13, -15), and R = (4, -6) is 3x + y - 39 = 0 -/
theorem angle_bisector_equation :
  let P : ℝ × ℝ := (-7, 6)
  let Q : ℝ × ℝ := (-13, -15)
  let R : ℝ × ℝ := (4, -6)
  ∃ (f : ℝ → ℝ), 
    (∀ x y, f x = y ↔ 3 * x + y - 39 = 0) ∧
    (∀ t : ℝ, f (P.1 + t * (Q.1 - P.1)) = P.2 + t * (Q.2 - P.2)) ∧
    (∀ t : ℝ, f (P.1 + t * (R.1 - P.1)) = P.2 + t * (R.2 - P.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equation_l122_12228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l122_12200

/-- Calculates the overall percentage change in membership over three periods -/
theorem membership_change (fall_increase winter_increase spring_decrease : ℝ) :
  fall_increase = 0.08 →
  winter_increase = 0.15 →
  spring_decrease = 0.19 →
  let overall_change := (1 + fall_increase) * (1 + winter_increase) * (1 - spring_decrease) - 1
  abs (overall_change - 0.242) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l122_12200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l122_12213

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_φ : abs φ < π / 2) 
  (h_period : ∀ x, f A ω φ (x + π) = f A ω φ x) 
  (h_highest : f A ω φ (π / 6) = 2 ∧ ∀ x, f A ω φ x ≤ 2) :
  (∀ x, f A ω φ x = 2 * Real.sin (2 * x + π / 6)) ∧
  (∀ θ, 0 < θ → θ < π / 2 → 
    (∀ x ∈ Set.Icc 0 (π / 4), 
      Monotone (fun x => f A ω φ (x - θ))) → 
    π / 12 ≤ θ ∧ θ ≤ π / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l122_12213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_natural_solution_l122_12254

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^(n+1)) / (1 - r)

theorem smallest_natural_solution :
  ∀ n : ℕ, n ≥ 5 ↔ 
    (2023/2022 : ℝ)^(geometric_sum 27 (2/3) n) > (2023/2022 : ℝ)^72 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_natural_solution_l122_12254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_ratio_343_implies_common_ratio_6_l122_12234

/-- The sum of the first n terms of a geometric progression with first term a and common ratio r. -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: If the ratio of the sum of the first 6 terms to the sum of the first 3 terms
    of a geometric progression is 343, then the common ratio is 6. -/
theorem gp_ratio_343_implies_common_ratio_6 (a : ℝ) (r : ℝ) :
  a ≠ 0 → r ≠ 1 →
  (geometric_sum a r 6) / (geometric_sum a r 3) = 343 →
  r = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_ratio_343_implies_common_ratio_6_l122_12234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_formation_l122_12270

theorem five_digit_number_formation (a b : ℕ) 
  (ha : 100 ≤ a ∧ a < 1000) 
  (hb : 10 ≤ b ∧ b < 100) : 
  1000 * b + a = 100000 * (b / 10) + 10000 * (b % 10) + a := by
  sorry -- Proof to be completed

#check five_digit_number_formation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_formation_l122_12270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l122_12255

/-- A two-digit natural number -/
def TwoDigitNumber : Type := { n : ℕ // n ≥ 10 ∧ n < 100 }

/-- The tens digit of a two-digit number -/
def tensDigit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- The units digit of a two-digit number -/
def unitsDigit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The set of two-digit numbers where the tens digit is greater than the units digit -/
def ValidNumbers : Set TwoDigitNumber :=
  { n | tensDigit n > unitsDigit n }

/-- Proof that ValidNumbers is finite -/
instance : Fintype ValidNumbers :=
  sorry

/-- The theorem stating that the count of valid numbers is 45 -/
theorem count_valid_numbers : Fintype.card ValidNumbers = 45 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l122_12255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_equals_one_no_a_b_exist_l122_12288

/-- A function f is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = ((x+1)(x-t))/x^2 --/
noncomputable def f (t : ℝ) : ℝ → ℝ :=
  fun x ↦ ((x + 1) * (x - t)) / (x^2)

/-- Theorem: If f is even, then t = 1 --/
theorem t_equals_one (h : IsEven (f t)) : t = 1 := by
  sorry

/-- The function g(x) = ((x+1)(x-1))/x^2 --/
noncomputable def g : ℝ → ℝ :=
  fun x ↦ ((x + 1) * (x - 1)) / (x^2)

/-- Theorem: There do not exist real numbers b > a > 0 such that
    when x ∈ [a, b], the range of g(x) is [2 - 2/a, 2 - 2/b] --/
theorem no_a_b_exist :
  ¬ ∃ (a b : ℝ), 0 < a ∧ a < b ∧
    (∀ x ∈ Set.Icc a b, 2 - 2/a ≤ g x ∧ g x ≤ 2 - 2/b) ∧
    (∃ x ∈ Set.Icc a b, g x = 2 - 2/a) ∧
    (∃ x ∈ Set.Icc a b, g x = 2 - 2/b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_equals_one_no_a_b_exist_l122_12288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l122_12285

theorem complex_expression_equality (a : ℝ) : 
  (a / (3 * (a^2 + 1)^(1/2 : ℝ)) - 
   ((2 * a^2 + 1 + a * (4 * a^2 + 3)^(1/2 : ℝ))^(1/2 : ℝ)) * 
   ((2 * a^2 + 3 + a * (4 * a^2 + 3)^(1/2 : ℝ))^(-1/2 : ℝ)))^2 = 
  (4 * a^2 + 3) / (9 * (a^2 + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l122_12285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l122_12273

-- Define the function f
noncomputable def f (a : ℝ) : ℝ := ∫ x in (Set.Icc 0 1), 2 * a * x^2 - a^2 * x

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = 2/9 ∧ ∀ (a : ℝ), f a ≤ max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l122_12273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_remainder_three_mod_seven_l122_12235

theorem three_digit_remainder_three_mod_seven : 
  (Finset.filter (λ n : ℕ => 100 ≤ n ∧ n < 1000 ∧ n % 7 = 3) (Finset.range 1000)).card = 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_remainder_three_mod_seven_l122_12235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_nonpositive_l122_12278

open Real

theorem derivative_reciprocal_nonpositive :
  (∀ x : ℝ, ∀ y : ℝ, x < y → (fun x => x^3) x < (fun x => x^3) y) →
  ¬ ∃ x : ℝ, deriv (fun x => 1 / x) x > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_nonpositive_l122_12278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l122_12202

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + (Real.log x) / (Real.log 3)

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem max_value_of_y :
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 9 ∧ ∀ (z : ℝ), 1 ≤ z ∧ z ≤ 9 → y z ≤ y x ∧ y x = 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l122_12202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_representation_l122_12208

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -3 ∧ x ≤ 0 then -2 - x
  else if x ≥ 0 ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x ≥ 2 ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Default value for x outside [-3, 3]

-- State the theorem
theorem abs_f_representation (x : ℝ) :
  (x ≥ -3 ∧ x ≤ 3) →
  (|f x| = if x ≥ -3 ∧ x ≤ 0 then 2 + x
           else if x ≥ 0 ∧ x ≤ 2 then 2 - Real.sqrt (4 - (x - 2)^2)
           else 2 * (x - 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_representation_l122_12208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l122_12230

theorem trig_identity (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 4/3)
  (h2 : π/4 < θ)
  (h3 : θ < π/2) :
  Real.cos θ - Real.sin θ = -Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l122_12230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_boys_in_varsity_clubs_l122_12232

theorem fraction_of_boys_in_varsity_clubs (total_students : ℕ) 
  (percent_girls : ℚ) (boys_not_in_clubs : ℕ) 
  (h1 : total_students = 150)
  (h2 : percent_girls = 60 / 100)
  (h3 : boys_not_in_clubs = 40)
  : (total_students * (1 - percent_girls) - boys_not_in_clubs) / 
    (total_students * (1 - percent_girls)) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_boys_in_varsity_clubs_l122_12232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_bar_weight_l122_12264

/-- The weight loss of the metal bar in water (in kg) -/
noncomputable def water_loss : ℝ := 2

/-- The weight loss of 10 kg of tin in water (in kg) -/
noncomputable def tin_loss : ℝ := 1.375

/-- The weight loss of 5 kg of silver in water (in kg) -/
noncomputable def silver_loss : ℝ := 0.375

/-- The ratio of tin to silver in the metal bar -/
noncomputable def tin_silver_ratio : ℝ := 2/3

/-- The weight of the metal bar (in kg) -/
noncomputable def bar_weight : ℝ := 20

/-- Theorem stating that given the conditions, the weight of the metal bar is 20 kg -/
theorem metal_bar_weight : 
  ∃ (T S : ℝ),
    T / S = tin_silver_ratio ∧
    (tin_loss / 10) * T + (silver_loss / 5) * S = water_loss ∧
    T + S = bar_weight :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_bar_weight_l122_12264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_height_approx_half_meter_l122_12210

/-- Represents the height of a water tower in meters -/
noncomputable def actual_height : ℝ := 40

/-- Represents the volume of water in the actual tower in liters -/
noncomputable def actual_volume : ℝ := 100000

/-- Represents the volume of water in the scaled model in liters -/
noncomputable def model_volume : ℝ := 0.2

/-- Calculates the height of the scaled model tower -/
noncomputable def model_height : ℝ := 
  actual_height * (model_volume / actual_volume) ^ (1/3)

/-- Theorem stating that the model height is approximately 0.5 meters -/
theorem model_height_approx_half_meter : 
  ∃ ε > 0, abs (model_height - 0.5) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_height_approx_half_meter_l122_12210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_is_44_div_3_l122_12215

/-- Represents the number of students in seventh grade -/
def s : ℕ := 2  -- We need a concrete value for s to make the code compilable

/-- The average number of minutes run by sixth graders per day -/
def sixth_grade_minutes : ℚ := 20

/-- The average number of minutes run by eighth graders per day -/
def eighth_grade_minutes : ℚ := 12

/-- The number of sixth grade students -/
def sixth_grade_students : ℕ := 3 * s

/-- The number of eighth grade students -/
def eighth_grade_students : ℕ := s / 2

/-- The total number of students -/
def total_students : ℕ := sixth_grade_students + s + eighth_grade_students

/-- The total minutes run by all students on the particular day -/
def total_minutes_run : ℚ := sixth_grade_minutes * (sixth_grade_students : ℚ) + 
                              eighth_grade_minutes * (eighth_grade_students : ℚ)

/-- The average number of minutes run per student on the particular day -/
noncomputable def average_minutes_run : ℚ := total_minutes_run / (total_students : ℚ)

theorem average_minutes_is_44_div_3 : average_minutes_run = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_is_44_div_3_l122_12215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_regular_triangular_prism_l122_12299

noncomputable def base_edge_length_min_surface_area : ℝ :=
  2 * (4 : ℝ)^(1/3)

theorem min_surface_area_regular_triangular_prism (volume : ℝ) 
  (h_volume : volume = 8) :
  let a := base_edge_length_min_surface_area
  let h := (4 * volume) / (Real.sqrt 3 * a^2)
  let surface_area := 3 * a * h + Real.sqrt 3 * a^2 / 2
  ∀ x > 0, 
    let h_x := (4 * volume) / (Real.sqrt 3 * x^2)
    let surface_area_x := 3 * x * h_x + Real.sqrt 3 * x^2 / 2
    surface_area ≤ surface_area_x :=
by sorry

#check min_surface_area_regular_triangular_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_regular_triangular_prism_l122_12299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phung_minus_chiu_l122_12241

/-- The number of silver dollars Mr. Chiu has -/
def C : ℕ := 56

/-- The number of silver dollars Mr. Phung has -/
def P : ℕ := sorry

/-- The number of silver dollars Mr. Ha has -/
def H : ℕ := sorry

/-- Mr. Ha owns 5 more silver dollars than Mr. Phung -/
axiom ha_vs_phung : H = P + 5

/-- Mr. Phung has more silver dollars than Mr. Chiu -/
axiom phung_vs_chiu : P > C

/-- The total number of silver dollars owned by all three -/
axiom total_dollars : C + P + H = 205

/-- The theorem stating the difference between Mr. Phung's and Mr. Chiu's silver dollars -/
theorem phung_minus_chiu : P - C = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phung_minus_chiu_l122_12241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l122_12206

theorem tan_double_angle (a : Real) (h : Real.tan a = -2) : Real.tan (2 * a) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l122_12206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l122_12229

theorem triangle_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * c * (a - b) + b^2 * a * (b - c) + c^2 * b * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l122_12229
