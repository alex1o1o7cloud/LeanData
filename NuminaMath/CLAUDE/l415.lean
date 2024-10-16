import Mathlib

namespace NUMINAMATH_CALUDE_weight_replacement_l415_41599

theorem weight_replacement (n : ℕ) (avg_increase weight_new : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  weight_new = 70 →
  weight_new - n * avg_increase = 50 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l415_41599


namespace NUMINAMATH_CALUDE_paul_initial_books_l415_41503

/-- The number of books Paul sold in the garage sale -/
def books_sold : ℕ := 78

/-- The number of books Paul has left after the sale -/
def books_left : ℕ := 37

/-- The initial number of books Paul had -/
def initial_books : ℕ := books_sold + books_left

theorem paul_initial_books : initial_books = 115 := by
  sorry

end NUMINAMATH_CALUDE_paul_initial_books_l415_41503


namespace NUMINAMATH_CALUDE_a_share_is_4800_l415_41545

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share (contribution_a : ℕ) (months_a : ℕ) (contribution_b : ℕ) (months_b : ℕ) (total_profit : ℕ) : ℕ :=
  let money_months_a := contribution_a * months_a
  let money_months_b := contribution_b * months_b
  let total_money_months := money_months_a + money_months_b
  (money_months_a * total_profit) / total_money_months

/-- Theorem stating that A's share of the profit is 4800 given the problem conditions --/
theorem a_share_is_4800 :
  calculate_share 5000 8 6000 5 8400 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_4800_l415_41545


namespace NUMINAMATH_CALUDE_stating_right_triangle_constructible_l415_41585

/-- Represents a right triangle --/
structure RightTriangle where
  hypotenuse : ℝ
  angle_difference : ℝ
  h_hypotenuse_positive : hypotenuse > 0
  h_angle_difference_range : 0 < angle_difference ∧ angle_difference < 90

/-- 
Theorem stating that a right triangle can be constructed 
given its hypotenuse and the difference of its two acute angles (ε), 
if and only if 0 < ε < 90°
--/
theorem right_triangle_constructible (h : ℝ) (ε : ℝ) :
  (∃ (t : RightTriangle), t.hypotenuse = h ∧ t.angle_difference = ε) ↔ 
  (h > 0 ∧ 0 < ε ∧ ε < 90) :=
sorry

end NUMINAMATH_CALUDE_stating_right_triangle_constructible_l415_41585


namespace NUMINAMATH_CALUDE_fly_distance_l415_41590

/-- Prove that the distance traveled by a fly between two approaching cyclists is 50 km -/
theorem fly_distance (initial_distance : ℝ) (cyclist1_speed cyclist2_speed fly_speed : ℝ) :
  initial_distance = 50 →
  cyclist1_speed = 40 →
  cyclist2_speed = 60 →
  fly_speed = 100 →
  let relative_speed := cyclist1_speed + cyclist2_speed
  let time := initial_distance / relative_speed
  fly_speed * time = 50 := by sorry

end NUMINAMATH_CALUDE_fly_distance_l415_41590


namespace NUMINAMATH_CALUDE_opposite_absolute_values_l415_41533

theorem opposite_absolute_values (x y : ℝ) : 
  (|x - y + 9| + |2*x + y| = 0) → (x = -3 ∧ y = 6) := by
sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_l415_41533


namespace NUMINAMATH_CALUDE_logical_equivalence_l415_41564

theorem logical_equivalence (P Q R : Prop) :
  (¬P ∧ ¬Q → R) ↔ (P ∨ Q ∨ R) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l415_41564


namespace NUMINAMATH_CALUDE_roots_difference_implies_k_value_l415_41535

theorem roots_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, 
    (r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0) ∧ 
    ((r+4)^2 - k*(r+4) + 10 = 0 ∧ (s+4)^2 - k*(s+4) + 10 = 0)) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_implies_k_value_l415_41535


namespace NUMINAMATH_CALUDE_max_value_polynomial_l415_41572

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 6084/17 ∧
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ 
    x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 = 6084/17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l415_41572


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l415_41551

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Two lines are distinct -/
def distinct_lines (l1 l2 : Line3D) : Prop := sorry

/-- Two planes are distinct -/
def distinct_planes (p1 p2 : Plane3D) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perpendicular (p1 p2 : Plane3D) : Prop := sorry

theorem line_plane_perpendicularity 
  (m n : Line3D) (α β : Plane3D) 
  (h1 : distinct_lines m n)
  (h2 : distinct_planes α β)
  (h3 : line_parallel_to_plane m α)
  (h4 : line_perpendicular_to_plane n β)
  (h5 : lines_parallel m n) :
  planes_perpendicular α β := by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l415_41551


namespace NUMINAMATH_CALUDE_percentage_relationship_l415_41542

theorem percentage_relationship (x y : ℝ) : 
  Real.sqrt (0.3 * (x - y)) = Real.sqrt (0.2 * (x + y)) → y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l415_41542


namespace NUMINAMATH_CALUDE_square_sum_factorization_l415_41527

theorem square_sum_factorization (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_factorization_l415_41527


namespace NUMINAMATH_CALUDE_counterexample_exists_l415_41525

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l415_41525


namespace NUMINAMATH_CALUDE_cosine_sum_problem_l415_41574

theorem cosine_sum_problem (x y z : ℝ) : 
  x = Real.cos (π / 13) → 
  y = Real.cos (3 * π / 13) → 
  z = Real.cos (9 * π / 13) → 
  x * y + y * z + z * x = -1/4 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_problem_l415_41574


namespace NUMINAMATH_CALUDE_equation_solutions_l415_41512

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 2*x - 7 = 0 ↔ x = 1 + 2*Real.sqrt 2 ∨ x = 1 - 2*Real.sqrt 2) ∧
  (∃ x : ℝ, 3*(x-2)^2 = x*(x-2) ↔ x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l415_41512


namespace NUMINAMATH_CALUDE_log_sqrt7_343sqrt7_equals_7_l415_41589

theorem log_sqrt7_343sqrt7_equals_7 :
  Real.log (343 * Real.sqrt 7) / Real.log (Real.sqrt 7) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt7_343sqrt7_equals_7_l415_41589


namespace NUMINAMATH_CALUDE_det_skew_symmetric_nonneg_l415_41558

/-- A 4x4 real matrix is skew-symmetric if its transpose is equal to its negation. -/
def isSkewSymmetric (A : Matrix (Fin 4) (Fin 4) ℝ) : Prop :=
  A.transpose = -A

/-- The determinant of a 4x4 real skew-symmetric matrix is non-negative. -/
theorem det_skew_symmetric_nonneg (A : Matrix (Fin 4) (Fin 4) ℝ) 
  (h : isSkewSymmetric A) : 0 ≤ A.det := by
  sorry

end NUMINAMATH_CALUDE_det_skew_symmetric_nonneg_l415_41558


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_2x2_minus_8x_plus_6_l415_41514

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_2x2_minus_8x_plus_6 :
  let f : ℝ → ℝ := λ x => 2*x^2 - 8*x + 6
  let r₁ := (-(-8) + Real.sqrt ((-8)^2 - 4*2*6)) / (2*2)
  let r₂ := (-(-8) - Real.sqrt ((-8)^2 - 4*2*6)) / (2*2)
  r₁ + r₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_2x2_minus_8x_plus_6_l415_41514


namespace NUMINAMATH_CALUDE_percentage_problem_l415_41529

theorem percentage_problem : 
  let product := 45 * 8
  let total := 900
  let percentage := (product / total) * 100
  percentage = 40 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l415_41529


namespace NUMINAMATH_CALUDE_right_triangle_angle_identity_l415_41536

theorem right_triangle_angle_identity (α β γ : Real) 
  (h_right_triangle : α + β + γ = π)
  (h_right_angle : α = π/2 ∨ β = π/2 ∨ γ = π/2) : 
  Real.sin α * Real.sin β * Real.sin (α - β) + 
  Real.sin β * Real.sin γ * Real.sin (β - γ) + 
  Real.sin γ * Real.sin α * Real.sin (γ - α) + 
  Real.sin (α - β) * Real.sin (β - γ) * Real.sin (γ - α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_identity_l415_41536


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l415_41593

theorem gdp_scientific_notation :
  let gdp_billion : ℝ := 32.07
  let billion : ℝ := 10^9
  let gdp : ℝ := gdp_billion * billion
  ∃ (a : ℝ) (n : ℤ), gdp = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.207 ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l415_41593


namespace NUMINAMATH_CALUDE_library_visits_l415_41581

/-- Proves that William goes to the library 2 times per week given the conditions -/
theorem library_visits (jason_freq : ℕ) (william_freq : ℕ) (jason_total : ℕ) (weeks : ℕ) :
  jason_freq = 4 * william_freq →
  jason_total = 32 →
  weeks = 4 →
  jason_total = jason_freq * weeks →
  william_freq = 2 := by
  sorry

end NUMINAMATH_CALUDE_library_visits_l415_41581


namespace NUMINAMATH_CALUDE_cookie_sales_proof_l415_41554

theorem cookie_sales_proof (total_value : ℝ) (choc_price plain_price : ℝ) (plain_boxes : ℝ) :
  total_value = 1586.25 →
  choc_price = 1.25 →
  plain_price = 0.75 →
  plain_boxes = 793.125 →
  ∃ (choc_boxes : ℝ), 
    choc_price * choc_boxes + plain_price * plain_boxes = total_value ∧
    choc_boxes + plain_boxes = 1586.25 :=
by sorry

end NUMINAMATH_CALUDE_cookie_sales_proof_l415_41554


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l415_41577

/-- Given a square divided into four congruent rectangles, if the perimeter of each rectangle is 28 inches, then the perimeter of the square is 44.8 inches. -/
theorem square_perimeter_from_rectangle_perimeter : 
  ∀ (s : ℝ), 
  s > 0 → -- side length of the square is positive
  (5 * s / 2 = 28) → -- perimeter of each rectangle is 28 inches
  (4 * s = 44.8) -- perimeter of the square is 44.8 inches
:= by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l415_41577


namespace NUMINAMATH_CALUDE_percentage_with_both_pets_l415_41563

def total_students : ℕ := 40
def puppy_percentage : ℚ := 80 / 100
def both_pets : ℕ := 8

theorem percentage_with_both_pets : 
  (both_pets : ℚ) / (puppy_percentage * total_students) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_both_pets_l415_41563


namespace NUMINAMATH_CALUDE_train_length_calculation_l415_41575

/-- The length of a train given its speed, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : 
  train_speed = 60 →
  man_speed = 6 →
  passing_time = 5.999520038396929 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l415_41575


namespace NUMINAMATH_CALUDE_actual_spent_correct_l415_41505

/-- Represents a project budget with monthly allocations -/
structure ProjectBudget where
  total : ℕ
  months : ℕ
  monthly_allocation : ℕ
  h_allocation : monthly_allocation * months = total

/-- Calculates the actual amount spent given a project budget and over-budget amount -/
def actual_spent (budget : ProjectBudget) (over_budget : ℕ) (months_elapsed : ℕ) : ℕ :=
  budget.monthly_allocation * months_elapsed + over_budget

/-- Proves that the actual amount spent is correct given the project conditions -/
theorem actual_spent_correct (budget : ProjectBudget) 
    (h_total : budget.total = 12600)
    (h_months : budget.months = 12)
    (h_over_budget : over_budget = 280)
    (h_months_elapsed : months_elapsed = 6) :
    actual_spent budget over_budget months_elapsed = 6580 := by
  sorry

#eval actual_spent ⟨12600, 12, 1050, rfl⟩ 280 6

end NUMINAMATH_CALUDE_actual_spent_correct_l415_41505


namespace NUMINAMATH_CALUDE_peach_count_difference_l415_41544

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 17

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 16

/-- The difference between the number of red peaches and green peaches -/
def peach_difference : ℕ := red_peaches - green_peaches

theorem peach_count_difference : peach_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_difference_l415_41544


namespace NUMINAMATH_CALUDE_min_distance_sum_l415_41532

/-- A scalene triangle with sides a, b, c where a > b > c -/
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  scalene : a > b ∧ b > c
  positive : a > 0 ∧ b > 0 ∧ c > 0

/-- A point inside or on the boundary of a triangle -/
structure TrianglePoint (t : ScaleneTriangle) where
  x : ℝ
  y : ℝ
  z : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z ≤ t.a

/-- The sum of distances from a point to the sides of the triangle -/
def distance_sum (t : ScaleneTriangle) (p : TrianglePoint t) : ℝ :=
  p.x + p.y + p.z

/-- The vertex opposite to the largest side -/
def opposite_vertex (t : ScaleneTriangle) : TrianglePoint t where
  x := t.a
  y := 0
  z := 0
  in_triangle := by sorry

/-- Theorem: The point that minimizes the sum of distances is the vertex opposite to the largest side -/
theorem min_distance_sum (t : ScaleneTriangle) :
  ∀ p : TrianglePoint t, distance_sum t (opposite_vertex t) ≤ distance_sum t p :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l415_41532


namespace NUMINAMATH_CALUDE_total_rainfall_2012_l415_41576

-- Define the average monthly rainfall for each year
def rainfall_2010 : ℝ := 37.2
def rainfall_2011 : ℝ := rainfall_2010 + 3.5
def rainfall_2012 : ℝ := rainfall_2011 - 1.2

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem statement
theorem total_rainfall_2012 : 
  rainfall_2012 * months_in_year = 474 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_2012_l415_41576


namespace NUMINAMATH_CALUDE_burger_cost_proof_l415_41538

/-- The cost of Uri's purchase in cents -/
def uri_cost : ℕ := 450

/-- The cost of Gen's purchase in cents -/
def gen_cost : ℕ := 480

/-- The number of burgers Uri bought -/
def uri_burgers : ℕ := 3

/-- The number of sodas Uri bought -/
def uri_sodas : ℕ := 2

/-- The number of burgers Gen bought -/
def gen_burgers : ℕ := 2

/-- The number of sodas Gen bought -/
def gen_sodas : ℕ := 3

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 78

theorem burger_cost_proof :
  ∃ (soda_cost : ℕ),
    uri_burgers * burger_cost + uri_sodas * soda_cost = uri_cost ∧
    gen_burgers * burger_cost + gen_sodas * soda_cost = gen_cost :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_proof_l415_41538


namespace NUMINAMATH_CALUDE_franks_goal_amount_l415_41547

-- Define the price of a hamburger
def hamburger_price : ℕ := 5

-- Define the number of customers who bought 4 hamburgers
def customers_buying_four : ℕ := 2

-- Define the number of customers who bought 2 hamburgers
def customers_buying_two : ℕ := 2

-- Define the number of additional hamburgers Frank needs to sell
def additional_hamburgers : ℕ := 4

-- Theorem to prove
theorem franks_goal_amount :
  (customers_buying_four * 4 + customers_buying_two * 2 + additional_hamburgers) * hamburger_price = 80 := by
  sorry


end NUMINAMATH_CALUDE_franks_goal_amount_l415_41547


namespace NUMINAMATH_CALUDE_sentence_reappears_l415_41566

-- Define the type for documents
def Document := List String

-- Define William's word assignment function
def wordAssignment : Char → String := sorry

-- Generate the nth document
def generateDocument : ℕ → Document
  | 0 => [wordAssignment 'A']
  | n + 1 => sorry -- Replace each letter in the previous document with its assigned word

-- The 40th document starts with this sentence
def startingSentence : List String :=
  ["Till", "whatsoever", "star", "that", "guides", "my", "moving"]

-- Main theorem
theorem sentence_reappears (d : Document) (h : d = generateDocument 40) :
  ∃ (i j : ℕ), i < j ∧ j < d.length ∧
  (List.take 7 (List.drop i d) = startingSentence) ∧
  (List.take 7 (List.drop j d) = startingSentence) :=
sorry

end NUMINAMATH_CALUDE_sentence_reappears_l415_41566


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l415_41580

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l
def line_l (x y : ℝ) (m : ℝ) : Prop := x = m * y + 1

-- Define perpendicularity of vectors
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem ellipse_and_line_theorem :
  -- Conditions
  (ellipse_C (-2) 0) ∧
  (ellipse_C (Real.sqrt 2) ((Real.sqrt 2) / 2)) ∧
  -- Existence of intersection points M and N
  ∃ (x1 y1 x2 y2 : ℝ),
    (ellipse_C x1 y1) ∧
    (ellipse_C x2 y2) ∧
    (∃ (m : ℝ), line_l x1 y1 m ∧ line_l x2 y2 m) ∧
    -- M and N are distinct
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    -- OM ⊥ ON
    (perpendicular x1 y1 x2 y2) →
  -- Conclusion
  ∃ (m : ℝ), (m = 1/2 ∨ m = -1/2) ∧ line_l 1 0 m := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l415_41580


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l415_41523

theorem power_of_three_mod_five : 3^2023 % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l415_41523


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l415_41592

theorem sphere_surface_area_of_circumscribed_cube (edge_length : ℝ) 
  (h : edge_length = 2 * Real.sqrt 3) :
  let diagonal := Real.sqrt 3 * edge_length
  let radius := diagonal / 2
  4 * Real.pi * radius ^ 2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l415_41592


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l415_41515

/-- A quadratic function with a positive constant term -/
def f (a b k : ℝ) (hk : k > 0) (x : ℝ) : ℝ := a * x^2 + b * x + k

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_sum_of_coefficients 
  (a b k : ℝ) (hk : k > 0) : 
  (f' a b 0 = 0) → 
  (f' a b 1 = 2) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l415_41515


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l415_41598

/-- Triangle DEF with vertices D(0, 10), E(4, 0), and F(10, 0) -/
structure Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

/-- A line defined by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- Checks if a line bisects the area of a triangle -/
def bisects_area (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The specific triangle DEF from the problem -/
def triangle_DEF : Triangle :=
  { D := (0, 10),
    E := (4, 0),
    F := (10, 0) }

/-- Main theorem: The line through E that bisects the area of triangle DEF
    has a slope and y-intercept whose sum is -15 -/
theorem bisecting_line_sum (l : Line) :
  bisects_area triangle_DEF l → l.slope + l.y_intercept = -15 :=
by sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l415_41598


namespace NUMINAMATH_CALUDE_johns_journey_distance_l415_41567

theorem johns_journey_distance :
  let total_distance : ℚ := 360 / 7
  let highway_distance : ℚ := total_distance / 4
  let city_distance : ℚ := 30
  let country_distance : ℚ := total_distance / 6
  highway_distance + city_distance + country_distance = total_distance := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_distance_l415_41567


namespace NUMINAMATH_CALUDE_vector_perpendicular_problem_l415_41511

theorem vector_perpendicular_problem (a b c : ℝ × ℝ) (k : ℝ) :
  a = (1, 2) →
  b = (1, 1) →
  c = (a.1 + k * b.1, a.2 + k * b.2) →
  b.1 * c.1 + b.2 * c.2 = 0 →
  k = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_problem_l415_41511


namespace NUMINAMATH_CALUDE_power_of_three_product_l415_41568

theorem power_of_three_product (x : ℕ) : 3^12 * 3^18 = x^6 → x = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_product_l415_41568


namespace NUMINAMATH_CALUDE_apple_distribution_l415_41552

theorem apple_distribution (total_apples : ℕ) (apples_per_classmate : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) : 
  total_apples / apples_per_classmate = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l415_41552


namespace NUMINAMATH_CALUDE_cats_sold_l415_41583

theorem cats_sold (siamese : ℕ) (house : ℕ) (remaining : ℕ) :
  siamese = 19 →
  house = 45 →
  remaining = 8 →
  siamese + house - remaining = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_cats_sold_l415_41583


namespace NUMINAMATH_CALUDE_circle_existence_condition_l415_41543

theorem circle_existence_condition (x y c : ℝ) : 
  (∃ h k r, (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔ 
  (x^2 + y^2 + 4*x - 2*y - 5*c = 0 → c > -1) :=
sorry

end NUMINAMATH_CALUDE_circle_existence_condition_l415_41543


namespace NUMINAMATH_CALUDE_nested_rectangles_exist_l415_41526

/-- Represents a rectangle with integer sides --/
structure Rectangle where
  width : Nat
  height : Nat
  width_bound : width ≤ 100
  height_bound : height ≤ 100

/-- Checks if rectangle a can be nested inside rectangle b --/
def can_nest (a b : Rectangle) : Prop :=
  a.width ≤ b.width ∧ a.height ≤ b.height

theorem nested_rectangles_exist (rectangles : Finset Rectangle) 
  (h : rectangles.card = 101) :
  ∃ (A B C : Rectangle), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles ∧
    can_nest A B ∧ can_nest B C := by
  sorry

end NUMINAMATH_CALUDE_nested_rectangles_exist_l415_41526


namespace NUMINAMATH_CALUDE_angle_30_less_than_complement_l415_41521

theorem angle_30_less_than_complement : 
  ∀ x : ℝ, x = 90 - x - 30 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_30_less_than_complement_l415_41521


namespace NUMINAMATH_CALUDE_num_lineups_eq_1782_l415_41530

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 starters from a team of 15 players,
    including 4 quadruplets, with at most one quadruplet in the starting lineup -/
def num_lineups : ℕ :=
  let total_players : ℕ := 15
  let num_quadruplets : ℕ := 4
  let non_quadruplet_players : ℕ := total_players - num_quadruplets
  let starters : ℕ := 5
  (choose non_quadruplet_players starters) +
  (num_quadruplets * choose non_quadruplet_players (starters - 1))

theorem num_lineups_eq_1782 : num_lineups = 1782 := by
  sorry

end NUMINAMATH_CALUDE_num_lineups_eq_1782_l415_41530


namespace NUMINAMATH_CALUDE_lcm_problem_l415_41595

theorem lcm_problem (n : ℕ+) (h1 : Nat.lcm 40 n = 200) (h2 : Nat.lcm n 45 = 180) : n = 100 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l415_41595


namespace NUMINAMATH_CALUDE_min_max_values_min_value_three_variables_l415_41570

-- Problem 1
theorem min_max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1 / a^2)^2 + (b + 1 / b^2)^2 ≥ 25/4 ∧ (a + 1/a) * (b + 1/b) ≤ 25/4 := by
  sorry

-- Problem 2
theorem min_value_three_variables (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_min_value_three_variables_l415_41570


namespace NUMINAMATH_CALUDE_tank_height_is_16_l415_41502

/-- The height of a cylindrical water tank with specific conditions -/
def tank_height : ℝ := 16

/-- The base radius of the cylindrical water tank -/
def base_radius : ℝ := 3

/-- Theorem stating that the height of the tank is 16 cm under given conditions -/
theorem tank_height_is_16 :
  tank_height = 16 ∧
  base_radius = 3 ∧
  (π * base_radius^2 * (tank_height / 2) = 2 * (4/3) * π * base_radius^3) :=
by sorry

end NUMINAMATH_CALUDE_tank_height_is_16_l415_41502


namespace NUMINAMATH_CALUDE_modulo_equivalence_56234_l415_41508

theorem modulo_equivalence_56234 :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 56234 ≡ n [ZMOD 23] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_56234_l415_41508


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l415_41504

theorem no_linear_term_condition (p q : ℝ) : 
  (∀ x : ℝ, (x^2 - p*x + q)*(x - 3) = x^3 + (-p-3)*x^2 + 0*x + (-3*q)) → 
  q + 3*p = 0 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l415_41504


namespace NUMINAMATH_CALUDE_four_percent_of_fifty_l415_41507

theorem four_percent_of_fifty : ∃ x : ℝ, x = 50 * (4 / 100) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_percent_of_fifty_l415_41507


namespace NUMINAMATH_CALUDE_discount_calculation_l415_41548

/-- The discount received when buying multiple parts with a given original price,
    number of parts, and final price paid. -/
def discount (original_price : ℕ) (num_parts : ℕ) (final_price : ℕ) : ℕ :=
  original_price * num_parts - final_price

/-- Theorem stating that the discount is $121 given the problem conditions. -/
theorem discount_calculation :
  let original_price : ℕ := 80
  let num_parts : ℕ := 7
  let final_price : ℕ := 439
  discount original_price num_parts final_price = 121 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l415_41548


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l415_41522

theorem solution_set_of_inequality (x : ℝ) :
  Set.Icc (-5 : ℝ) 3 \ {3} = {x | (x + 5) / (3 - x) ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l415_41522


namespace NUMINAMATH_CALUDE_class_2003_ice_cream_picnic_student_ticket_cost_l415_41539

/-- The cost of a student ticket for the Class of 2003 ice cream picnic -/
def student_ticket_cost : ℚ := sorry

/-- The theorem stating the cost of a student ticket for the Class of 2003 ice cream picnic -/
theorem class_2003_ice_cream_picnic_student_ticket_cost :
  let total_tickets : ℕ := 193
  let non_student_ticket_cost : ℚ := 3/2
  let total_revenue : ℚ := 413/2
  let student_tickets : ℕ := 83
  let non_student_tickets : ℕ := total_tickets - student_tickets
  student_ticket_cost * student_tickets + non_student_ticket_cost * non_student_tickets = total_revenue ∧
  student_ticket_cost = 1/2
  := by sorry

end NUMINAMATH_CALUDE_class_2003_ice_cream_picnic_student_ticket_cost_l415_41539


namespace NUMINAMATH_CALUDE_green_chips_count_l415_41579

theorem green_chips_count (total : ℕ) (blue : ℕ) (white : ℕ) (green : ℕ) : 
  blue = 3 →
  blue = (10 * total) / 100 →
  white = (50 * total) / 100 →
  green = total - blue - white →
  green = 12 := by
sorry

end NUMINAMATH_CALUDE_green_chips_count_l415_41579


namespace NUMINAMATH_CALUDE_convergence_and_bound_l415_41569

def u : ℕ → ℚ
  | 0 => 1/6
  | n + 1 => 2 * u n - 2 * (u n)^2 + 1/3

def L : ℚ := 5/6

theorem convergence_and_bound :
  (∃ (k : ℕ), ∀ (n : ℕ), n ≥ k → |u n - L| ≤ 1 / 2^500) ∧
  (∀ (k : ℕ), k < 9 → ∃ (n : ℕ), n ≥ k ∧ |u n - L| > 1 / 2^500) ∧
  (∀ (n : ℕ), n ≥ 9 → |u n - L| ≤ 1 / 2^500) :=
sorry

end NUMINAMATH_CALUDE_convergence_and_bound_l415_41569


namespace NUMINAMATH_CALUDE_average_of_numbers_l415_41586

def numbers : List ℕ := [12, 13, 14, 510, 520, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125789 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l415_41586


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l415_41501

theorem angle_sum_at_point (y : ℝ) : 
  150 + y + 2*y = 360 → y = 70 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l415_41501


namespace NUMINAMATH_CALUDE_log_inequality_l415_41509

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((Real.sqrt a + Real.sqrt b) / 2) > Real.log (Real.sqrt (a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l415_41509


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l415_41555

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ n - sum_of_digits n = 2007

theorem special_numbers_theorem : 
  {n : ℕ | satisfies_condition n} = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019} :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l415_41555


namespace NUMINAMATH_CALUDE_no_three_five_powers_l415_41520

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem no_three_five_powers (n : ℕ) :
  ∀ α β : ℕ, v n ≠ 3^α * 5^β :=
by sorry

end NUMINAMATH_CALUDE_no_three_five_powers_l415_41520


namespace NUMINAMATH_CALUDE_football_practice_missed_days_l415_41561

/-- Calculates the number of days a football team missed practice due to rain. -/
theorem football_practice_missed_days
  (daily_hours : ℕ)
  (total_hours : ℕ)
  (days_in_week : ℕ)
  (h1 : daily_hours = 5)
  (h2 : total_hours = 30)
  (h3 : days_in_week = 7) :
  days_in_week - (total_hours / daily_hours) = 1 :=
by sorry

end NUMINAMATH_CALUDE_football_practice_missed_days_l415_41561


namespace NUMINAMATH_CALUDE_pool_capacity_l415_41506

/-- Represents the capacity of a pool and properties of a pump -/
structure Pool :=
  (capacity : ℝ)
  (pumpRate : ℝ)
  (pumpTime : ℝ)
  (remainingWater : ℝ)

/-- Theorem stating the capacity of the pool given the conditions -/
theorem pool_capacity 
  (p : Pool)
  (h1 : p.pumpRate = 2/3)
  (h2 : p.pumpTime = 7.5)
  (h3 : p.pumpTime * 8 = 0.15 * 60)
  (h4 : p.remainingWater = 25)
  (h5 : p.capacity * (1 - p.pumpRate * (0.15 * 60 / p.pumpTime)) = p.remainingWater) :
  p.capacity = 125 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l415_41506


namespace NUMINAMATH_CALUDE_max_log_sum_l415_41562

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 4*y = 40) :
  ∃ (max : ℝ), max = 8 * Real.log 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l415_41562


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l415_41537

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the condition given in the problem
def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n

-- Theorem statement
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ r : ℝ, (∀ n : ℕ, a (n + 1) = r * a n) ∧ r = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l415_41537


namespace NUMINAMATH_CALUDE_function_derivative_equality_l415_41546

theorem function_derivative_equality (f : ℝ → ℝ) (x : ℝ) : 
  (∀ x, f x = x^2 * (x - 1)) → 
  (deriv f) x = x → 
  x = 0 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_function_derivative_equality_l415_41546


namespace NUMINAMATH_CALUDE_production_days_l415_41584

/-- Given an initial average production of 50 units over n days, 
    adding 95 units on the next day results in a new average of 55 units,
    prove that n must be 8. -/
theorem production_days (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Initial average of 50 units over n days
  (h2 : ((n * 50 + 95 : ℝ) / (n + 1) = 55)) -- New average of 55 units over n+1 days
  : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l415_41584


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l415_41531

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube. -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let sphere_radius := sphere_diameter / 2
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l415_41531


namespace NUMINAMATH_CALUDE_birth_date_satisfies_conditions_l415_41591

/-- Represents a date with year, month, and day components -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Calculates the age of a person at a given year, given their birth date -/
def age (birthDate : Date) (currentYear : ℕ) : ℕ :=
  currentYear - birthDate.year

/-- Represents the problem conditions -/
def satisfiesConditions (birthDate : Date) : Prop :=
  let ageIn1937 := age birthDate 1937
  ageIn1937 * ageIn1937 = 1937 - birthDate.year ∧ 
  ageIn1937 + birthDate.month = birthDate.day * birthDate.day

/-- The main theorem to prove -/
theorem birth_date_satisfies_conditions : 
  satisfiesConditions (Date.mk 1892 5 7) :=
sorry

end NUMINAMATH_CALUDE_birth_date_satisfies_conditions_l415_41591


namespace NUMINAMATH_CALUDE_q_necessary_not_sufficient_l415_41510

/-- A function f is monotonically increasing on an interval if for any two points x and y in that interval, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x³ + 2x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The statement p: f(x) is monotonically increasing in (-∞, +∞) -/
def p (m : ℝ) : Prop := MonotonicallyIncreasing (f m)

/-- The statement q: m > 4/3 -/
def q (m : ℝ) : Prop := m > 4/3

/-- Theorem stating that q is a necessary but not sufficient condition for p -/
theorem q_necessary_not_sufficient :
  (∀ m : ℝ, p m → q m) ∧ (∃ m : ℝ, q m ∧ ¬(p m)) := by sorry

end NUMINAMATH_CALUDE_q_necessary_not_sufficient_l415_41510


namespace NUMINAMATH_CALUDE_train_crossing_time_l415_41587

/-- The time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 150 ∧ 
  train_speed = 85 * (1000 / 3600) ∧ 
  man_speed = 5 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l415_41587


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l415_41540

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_valid_votes : ℕ)
  (h_total : total_votes = 560000)
  (h_percentage : candidate_a_percentage = 70 / 100)
  (h_valid_votes : candidate_a_valid_votes = 333200) :
  (total_votes - (candidate_a_valid_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l415_41540


namespace NUMINAMATH_CALUDE_peter_additional_miles_l415_41559

/-- The number of additional miles Peter runs compared to Andrew each day -/
def additional_miles : ℝ := sorry

/-- Andrew's daily miles -/
def andrew_miles : ℝ := 2

/-- Number of days they run -/
def days : ℕ := 5

/-- Total miles run by both after 5 days -/
def total_miles : ℝ := 35

theorem peter_additional_miles :
  additional_miles = 3 ∧
  days * (andrew_miles + additional_miles) + days * andrew_miles = total_miles :=
sorry

end NUMINAMATH_CALUDE_peter_additional_miles_l415_41559


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l415_41528

/-- The average number of visitors on non-Sunday days in a library -/
def average_visitors_non_sunday (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) : ℚ :=
  let sundays := total_days / 7 + 1
  let other_days := total_days - sundays
  (total_days * month_avg - sundays * sunday_avg) / other_days

/-- Theorem stating the average number of visitors on non-Sunday days -/
theorem library_visitors_theorem :
  average_visitors_non_sunday 630 30 305 = 240 := by
  sorry

#eval average_visitors_non_sunday 630 30 305

end NUMINAMATH_CALUDE_library_visitors_theorem_l415_41528


namespace NUMINAMATH_CALUDE_holistic_substitution_l415_41513

theorem holistic_substitution (a : ℝ) (x : ℝ) :
  (a^2 + 3*a - 2 = 0) →
  (5*a^3 + 15*a^2 - 10*a + 2020 = 2020) ∧
  ((x^2 + 2*x - 3 = 0) → 
   (x = 1 ∨ x = -3) →
   ((2*x + 3)^2 + 2*(2*x + 3) - 3 = 0) →
   (x = -1 ∨ x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_holistic_substitution_l415_41513


namespace NUMINAMATH_CALUDE_pasta_calculation_l415_41594

/-- Given a recipe that uses 2 pounds of pasta to serve 7 people,
    calculate the amount of pasta needed to serve 35 people. -/
theorem pasta_calculation (original_pasta : ℝ) (original_servings : ℕ) 
    (target_servings : ℕ) (h1 : original_pasta = 2) 
    (h2 : original_servings = 7) (h3 : target_servings = 35) : 
    (original_pasta * target_servings / original_servings : ℝ) = 10 := by
  sorry

#check pasta_calculation

end NUMINAMATH_CALUDE_pasta_calculation_l415_41594


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l415_41518

theorem imaginary_part_of_2_plus_i_times_i (i : ℂ) : 
  Complex.im ((2 : ℂ) + i * i) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l415_41518


namespace NUMINAMATH_CALUDE_right_triangle_exterior_angles_sum_l415_41516

theorem right_triangle_exterior_angles_sum (α β γ δ ε : Real) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle
  γ = 90 →           -- Right angle in the triangle
  α + δ = 180 →      -- Linear pair for first non-right angle
  β + ε = 180 →      -- Linear pair for second non-right angle
  δ + ε = 270 :=     -- Sum of exterior angles
by sorry

end NUMINAMATH_CALUDE_right_triangle_exterior_angles_sum_l415_41516


namespace NUMINAMATH_CALUDE_dodge_trucks_count_l415_41541

theorem dodge_trucks_count (ford dodge toyota vw : ℕ) 
  (h1 : ford = dodge / 3)
  (h2 : ford = 2 * toyota)
  (h3 : vw = toyota / 2)
  (h4 : vw = 5) : 
  dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_dodge_trucks_count_l415_41541


namespace NUMINAMATH_CALUDE_product_equals_one_l415_41565

theorem product_equals_one :
  (∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  6 * 15 * 11 = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l415_41565


namespace NUMINAMATH_CALUDE_sequence_problem_l415_41578

theorem sequence_problem (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, b (n + 1) / b n = b 2 / b 1) →  -- geometric sequence condition
  a 1 + a 2 = 10 →
  a 4 - a 3 = 2 →
  b 2 = a 3 →
  b 3 = a 7 →
  b 5 = 64 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l415_41578


namespace NUMINAMATH_CALUDE_largest_calculation_l415_41588

theorem largest_calculation :
  let a := 2 + 0 + 1 + 8
  let b := 2 * 0 + 1 + 8
  let c := 2 + 0 * 1 + 8
  let d := 2 + 0 + 1 * 8
  let e := 2 * 0 + 1 * 8
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_calculation_l415_41588


namespace NUMINAMATH_CALUDE_no_solution_l415_41571

def connection (a b : ℕ+) : ℚ :=
  (Nat.lcm a.val b.val : ℚ) / (a.val * b.val)

theorem no_solution : ¬ ∃ y : ℕ+, y.val < 50 ∧ connection y 13 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l415_41571


namespace NUMINAMATH_CALUDE_remainder_of_geometric_sum_l415_41549

def geometric_sum (r : ℕ) (n : ℕ) : ℕ :=
  (r^(n + 1) - 1) / (r - 1)

theorem remainder_of_geometric_sum :
  (geometric_sum 7 2004) % 1000 = 801 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_geometric_sum_l415_41549


namespace NUMINAMATH_CALUDE_angle_A_is_pi_third_max_perimeter_is_3_sqrt_3_l415_41553

/-- Triangle ABC with angles A, B, C and opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vectors m and n are orthogonal -/
def vectors_orthogonal (t : Triangle) : Prop :=
  t.a * (Real.cos t.C + Real.sqrt 3 * Real.sin t.C) + (t.b + t.c) * (-1) = 0

/-- Theorem: If vectors are orthogonal, then angle A is π/3 -/
theorem angle_A_is_pi_third (t : Triangle) (h : vectors_orthogonal t) : t.A = π / 3 := by
  sorry

/-- Maximum perimeter when a = √3 -/
def max_perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: When a = √3, the maximum perimeter is 3√3 -/
theorem max_perimeter_is_3_sqrt_3 (t : Triangle) (h : t.a = Real.sqrt 3) :
  max_perimeter t ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_third_max_perimeter_is_3_sqrt_3_l415_41553


namespace NUMINAMATH_CALUDE_artist_payment_multiple_l415_41596

theorem artist_payment_multiple : ∃ (x : ℕ+) (D : ℕ+), 
  D + (x * D + 1000) = 50000 ∧ 
  ∀ (y : ℕ+), y > x → ¬∃ (E : ℕ+), E + (y * E + 1000) = 50000 := by
  sorry

end NUMINAMATH_CALUDE_artist_payment_multiple_l415_41596


namespace NUMINAMATH_CALUDE_barn_paint_area_l415_41524

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular opening (door or window) -/
structure OpeningDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a barn -/
def totalPaintArea (barn : BarnDimensions) (doors : List OpeningDimensions) (windows : List OpeningDimensions) : ℝ :=
  let wallArea := 2 * (barn.width * barn.height + barn.length * barn.height)
  let floorCeilingArea := 2 * (barn.width * barn.length)
  let doorArea := doors.map (fun d => d.width * d.height) |>.sum
  let windowArea := windows.map (fun w => w.width * w.height) |>.sum
  2 * (wallArea - doorArea - windowArea) + floorCeilingArea

/-- Theorem stating that the total area to be painted is 1588 sq yd -/
theorem barn_paint_area :
  let barn := BarnDimensions.mk 15 20 8
  let doors := [OpeningDimensions.mk 3 7, OpeningDimensions.mk 3 7]
  let windows := [OpeningDimensions.mk 2 4, OpeningDimensions.mk 2 4, OpeningDimensions.mk 2 4]
  totalPaintArea barn doors windows = 1588 := by
  sorry

end NUMINAMATH_CALUDE_barn_paint_area_l415_41524


namespace NUMINAMATH_CALUDE_quadratic_sum_abc_l415_41519

/-- The quadratic function f(x) = -4x^2 + 20x + 196 -/
def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x + 196

/-- The sum of a, b, and c when f(x) is expressed as a(x+b)^2 + c -/
def sum_abc : ℝ := 213.5

theorem quadratic_sum_abc :
  ∃ (a b c : ℝ), (∀ x, f x = a * (x + b)^2 + c) ∧ (a + b + c = sum_abc) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_abc_l415_41519


namespace NUMINAMATH_CALUDE_max_triangle_area_l415_41556

/-- The maximum area of a triangle ABC with side lengths satisfying the given constraints is 1 -/
theorem max_triangle_area (AB BC CA : ℝ) (h1 : 0 ≤ AB ∧ AB ≤ 1) (h2 : 1 ≤ BC ∧ BC ≤ 2) (h3 : 2 ≤ CA ∧ CA ≤ 3) :
  (∃ (S : ℝ), S = Real.sqrt ((AB + BC + CA) / 2 * ((AB + BC + CA) / 2 - AB) * ((AB + BC + CA) / 2 - BC) * ((AB + BC + CA) / 2 - CA))) →
  (∀ (area : ℝ), area ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l415_41556


namespace NUMINAMATH_CALUDE_panthers_score_l415_41517

theorem panthers_score (total_points margin : ℕ) 
  (h1 : total_points = 34)
  (h2 : margin = 14) : 
  total_points - (total_points + margin) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_panthers_score_l415_41517


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_15120_l415_41557

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio_equals_15120 : 
  factorial 10 / (factorial 5 * factorial 2) = 15120 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_15120_l415_41557


namespace NUMINAMATH_CALUDE_k_range_l415_41560

/-- The condition that for any real b, the line y = kx + b and the hyperbola x^2 - 2y^2 = 1 always have common points -/
def always_intersect (k : ℝ) : Prop :=
  ∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1

/-- The theorem stating the range of k given the always_intersect condition -/
theorem k_range (k : ℝ) : always_intersect k ↔ -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l415_41560


namespace NUMINAMATH_CALUDE_correct_operation_l415_41573

theorem correct_operation (a b : ℝ) : 2 * a^2 * b * (4 * a * b^3) = 8 * a^3 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l415_41573


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_l415_41597

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Side lengths

-- Define the properties of the triangle
def IsObtuseTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = 180 ∧
  (t.A > 90 ∨ t.B > 90 ∨ t.C > 90)

-- Theorem statement
theorem angle_C_is_30_degrees (t : Triangle) 
  (h1 : IsObtuseTriangle t)
  (h2 : t.a = 4)
  (h3 : t.b = 4 * Real.sqrt 3)
  (h4 : t.A = 30) :
  t.C = 30 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_l415_41597


namespace NUMINAMATH_CALUDE_rectangle_y_value_l415_41534

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices := [(1, y), (-5, y), (1, -2), (-5, -2)]
  let length := 1 - (-5)
  let height := y - (-2)
  let area := length * height
  area = 56 → y = 22/3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l415_41534


namespace NUMINAMATH_CALUDE_divisibility_condition_solutions_l415_41582

theorem divisibility_condition_solutions (n p : ℕ) (h_prime : Nat.Prime p) (h_range : 0 < n ∧ n ≤ 2 * p) :
  n^(p-1) ∣ (p-1)^n + 1 ↔ 
    (n = 1 ∧ p ≥ 2) ∨
    (n = 2 ∧ p = 2) ∨
    (n = 3 ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_solutions_l415_41582


namespace NUMINAMATH_CALUDE_distribute_five_three_l415_41500

/-- The number of ways to distribute n distinct elements into k distinct groups,
    where each group must contain at least one element. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinct elements into 3 distinct groups,
    where each group must contain at least one element. -/
theorem distribute_five_three : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_three_l415_41500


namespace NUMINAMATH_CALUDE_sum_y_z_is_twice_x_l415_41550

theorem sum_y_z_is_twice_x (x y z : ℝ) 
  (h1 : 0.6 * (x - y) = 0.3 * (x + y)) 
  (h2 : 0.4 * (x + z) = 0.2 * (y + z)) : 
  (y + z) / x = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_y_z_is_twice_x_l415_41550
