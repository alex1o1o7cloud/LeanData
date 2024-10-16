import Mathlib

namespace NUMINAMATH_CALUDE_min_fold_length_l2634_263435

/-- Given a rectangle ABCD with AB = 6 and AD = 12, when corner B is folded to edge AD
    creating a fold line MN, this function represents the length of MN (l)
    as a function of t, where t = sin θ and θ = �angle MNB -/
def fold_length (t : ℝ) : ℝ := 6 * t

/-- The theorem states that the minimum value of the fold length is 0 -/
theorem min_fold_length :
  ∃ (t : ℝ), t ≥ 0 ∧ t ≤ 1 ∧ ∀ (s : ℝ), s ≥ 0 → s ≤ 1 → fold_length t ≤ fold_length s :=
by sorry

end NUMINAMATH_CALUDE_min_fold_length_l2634_263435


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2634_263489

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 8 * x + k = 0) ↔ k = 16/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2634_263489


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2634_263486

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) = Real.sqrt 70 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2634_263486


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2634_263464

/-- The line equation y = 2mx + 2 intersects the ellipse 2x^2 + 8y^2 = 8 exactly once if and only if m^2 = 3/16 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! p : ℝ × ℝ, (2 * p.1^2 + 8 * p.2^2 = 8) ∧ (p.2 = 2 * m * p.1 + 2)) ↔ m^2 = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2634_263464


namespace NUMINAMATH_CALUDE_solve_equation_l2634_263426

theorem solve_equation (x : ℝ) : 5 * x + 3 = 10 * x - 17 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2634_263426


namespace NUMINAMATH_CALUDE_existence_of_numbers_l2634_263405

theorem existence_of_numbers : ∃ (a b c d : ℕ), 
  (a : ℚ) / b + (c : ℚ) / d = 1 ∧ (a : ℚ) / d + (c : ℚ) / b = 2008 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_numbers_l2634_263405


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2634_263421

theorem sum_of_a_and_b (a b : ℝ) : 
  (a + Real.sqrt b + (a - Real.sqrt b) = -6) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 4) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2634_263421


namespace NUMINAMATH_CALUDE_correct_division_result_l2634_263400

theorem correct_division_result (x : ℚ) (h : 9 - x = 3) : 96 / x = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_result_l2634_263400


namespace NUMINAMATH_CALUDE_dropped_student_score_l2634_263499

theorem dropped_student_score (initial_students : ℕ) (remaining_students : ℕ) 
  (initial_average : ℚ) (new_average : ℚ) : 
  initial_students = 16 →
  remaining_students = 15 →
  initial_average = 60.5 →
  new_average = 64 →
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 8 := by
  sorry

end NUMINAMATH_CALUDE_dropped_student_score_l2634_263499


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2634_263412

/-- Given vectors a and b, with c parallel to b and its projection on a being 2, prove |c| = 2√5 -/
theorem vector_magnitude_proof (a b c : ℝ × ℝ) : 
  a = (1, 0) →
  b = (1, 2) →
  c.fst / c.snd = b.fst / b.snd →  -- c is parallel to b
  (c.fst * a.fst + c.snd * a.snd) / Real.sqrt (a.fst^2 + a.snd^2) = 2 →  -- projection of c on a is 2
  Real.sqrt (c.fst^2 + c.snd^2) = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2634_263412


namespace NUMINAMATH_CALUDE_issacs_pens_l2634_263461

theorem issacs_pens (total : ℕ) (pens : ℕ) (pencils : ℕ) : 
  total = 108 →
  total = pens + pencils →
  pencils = 5 * pens + 12 →
  pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_issacs_pens_l2634_263461


namespace NUMINAMATH_CALUDE_distribution_plans_count_l2634_263401

/-- The number of ways to distribute 3 volunteer teachers among 6 schools, with at most 2 teachers per school -/
def distribution_plans : ℕ := 210

/-- The number of schools -/
def num_schools : ℕ := 6

/-- The number of volunteer teachers -/
def num_teachers : ℕ := 3

/-- The maximum number of teachers allowed per school -/
def max_teachers_per_school : ℕ := 2

theorem distribution_plans_count :
  distribution_plans = 210 :=
sorry

end NUMINAMATH_CALUDE_distribution_plans_count_l2634_263401


namespace NUMINAMATH_CALUDE_million_is_ten_to_six_roundness_of_million_l2634_263408

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ) : ℕ := sorry

/-- 1,000,000 can be expressed as 10^6 -/
theorem million_is_ten_to_six : 1000000 = 10^6 := by sorry

/-- The roundness of 1,000,000 is 12 -/
theorem roundness_of_million : roundness 1000000 = 12 := by sorry

end NUMINAMATH_CALUDE_million_is_ten_to_six_roundness_of_million_l2634_263408


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2634_263495

theorem cube_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a * b + a * c + b * c = 3)
  (h3 : a * b * c = 5) :
  a^3 + b^3 + c^3 = 15 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2634_263495


namespace NUMINAMATH_CALUDE_recursive_sum_value_l2634_263467

def recursive_sum (n : ℕ) : ℚ :=
  if n = 0 then 3
  else (n + 3 : ℚ) + (1 / 3) * recursive_sum (n - 1)

theorem recursive_sum_value : 
  recursive_sum 3000 = 4504 - (1 / 4) * (1 - 1 / 3^2999) :=
by sorry

end NUMINAMATH_CALUDE_recursive_sum_value_l2634_263467


namespace NUMINAMATH_CALUDE_pool_filling_rate_l2634_263469

/-- Given a pool filled by four hoses, this theorem proves the rate of two unknown hoses. -/
theorem pool_filling_rate 
  (pool_volume : ℝ) 
  (fill_time : ℝ) 
  (known_hose_rate : ℝ) 
  (h_volume : pool_volume = 15000)
  (h_time : fill_time = 25 * 60)  -- Convert hours to minutes
  (h_known_rate : known_hose_rate = 2)
  : ∃ (unknown_hose_rate : ℝ), 
    2 * known_hose_rate + 2 * unknown_hose_rate = pool_volume / fill_time ∧ 
    unknown_hose_rate = 3 :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_rate_l2634_263469


namespace NUMINAMATH_CALUDE_factor_divisibility_l2634_263417

theorem factor_divisibility : ∃ (n m : ℕ), (4 ∣ 24) ∧ (9 ∣ 180) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisibility_l2634_263417


namespace NUMINAMATH_CALUDE_original_alcohol_percentage_l2634_263480

/-- Proves that a 20-litre mixture of alcohol and water, when mixed with 3 litres of water,
    resulting in a new mixture with 17.391304347826086% alcohol, must have originally
    contained 20% alcohol. -/
theorem original_alcohol_percentage
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_percentage : ℝ)
  (h1 : original_volume = 20)
  (h2 : added_water = 3)
  (h3 : new_percentage = 17.391304347826086) :
  (original_volume * (100 / (original_volume + added_water)) * new_percentage / 100) = 20 :=
sorry

end NUMINAMATH_CALUDE_original_alcohol_percentage_l2634_263480


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l2634_263479

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_discount : ℝ) :
  list_price = 80 →
  max_regular_discount = 0.5 →
  summer_discount = 0.2 →
  let regular_discounted_price := list_price * (1 - max_regular_discount)
  let summer_discount_amount := list_price * summer_discount
  let final_sale_price := regular_discounted_price - summer_discount_amount
  (final_sale_price / list_price) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l2634_263479


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l2634_263482

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l2634_263482


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2634_263484

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 3 = 13 →
  (∃ n : ℕ, a n = 33 ∧ a (n - 2) + a (n - 1) = 51) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2634_263484


namespace NUMINAMATH_CALUDE_nested_sqrt_evaluation_l2634_263433

theorem nested_sqrt_evaluation (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 * Real.sqrt (x^2 * Real.sqrt (x^2))) = x^(7/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_evaluation_l2634_263433


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2634_263481

theorem triangle_angle_sum (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = 180) (h5 : A ≤ B) (h6 : A ≤ C) (h7 : C ≤ B) (h8 : 2 * B = 5 * A) : 
  ∃ (m n : ℝ), m = max B C ∧ n = min B C ∧ m + n = 175 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2634_263481


namespace NUMINAMATH_CALUDE_fraction_equality_l2634_263403

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (4*x + 2*y) / (2*x - 4*y) = 3) : 
  (2*x + 4*y) / (4*x - 2*y) = 9/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2634_263403


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2634_263477

theorem triangle_angle_measure (a b : ℝ) (ha : a = 117) (hb : b = 31) :
  180 - a + b = 86 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2634_263477


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l2634_263485

/-- Given two points P and Q in a plane, where Q is the midpoint of PR, 
    prove that R has specific coordinates. -/
theorem midpoint_coordinates (P Q : ℝ × ℝ) (h1 : P = (1, 3)) (h2 : Q = (4, 7)) 
    (h3 : Q = ((P.1 + R.1) / 2, (P.2 + R.2) / 2)) : R = (7, 11) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l2634_263485


namespace NUMINAMATH_CALUDE_max_blank_squares_l2634_263453

/-- Represents a grid of unit squares -/
structure Grid :=
  (size : ℕ)

/-- Represents a triangle placement on the grid -/
structure TrianglePlacement :=
  (grid : Grid)
  (covers_all_segments : Prop)

/-- Represents the count of squares without triangles -/
def blank_squares (tp : TrianglePlacement) : ℕ := sorry

/-- The main theorem: maximum number of blank squares in a 100x100 grid -/
theorem max_blank_squares :
  ∀ (tp : TrianglePlacement),
    tp.grid.size = 100 →
    tp.covers_all_segments →
    blank_squares tp ≤ 2450 :=
by sorry

end NUMINAMATH_CALUDE_max_blank_squares_l2634_263453


namespace NUMINAMATH_CALUDE_penguin_arrangements_l2634_263434

def word_length : ℕ := 7
def repeated_letter_count : ℕ := 2

theorem penguin_arrangements :
  (word_length.factorial / repeated_letter_count.factorial) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_penguin_arrangements_l2634_263434


namespace NUMINAMATH_CALUDE_parabola_equation_l2634_263432

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line x = 2
def Line (x : ℝ) : Prop := x = 2

-- Define the intersection points D and E
def Intersect (p : ℝ) (D E : ℝ × ℝ) : Prop :=
  C p D.1 D.2 ∧ C p E.1 E.2 ∧ Line D.1 ∧ Line E.1

-- Define the orthogonality condition
def Orthogonal (O D E : ℝ × ℝ) : Prop :=
  (D.1 - O.1) * (E.1 - O.1) + (D.2 - O.2) * (E.2 - O.2) = 0

-- The main theorem
theorem parabola_equation (p : ℝ) (D E : ℝ × ℝ) :
  C p D.1 D.2 ∧ C p E.1 E.2 ∧ Line D.1 ∧ Line E.1 ∧ 
  Orthogonal (0, 0) D E →
  ∀ x y : ℝ, C p x y ↔ y^2 = 2*x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2634_263432


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l2634_263490

-- Define the triangle vertices
def P : ℝ × ℝ := (-8, 2)
def Q : ℝ × ℝ := (-10, -10)
def R : ℝ × ℝ := (2, -4)

-- Define the angle bisector equation coefficients
def b : ℝ := sorry
def d : ℝ := sorry

-- State the theorem
theorem angle_bisector_sum (h : ∀ (x y : ℝ), b * x + 2 * y + d = 0 ↔ 
  (y - P.2) = (y - P.2) / (x - P.1) * (x - P.1)) : 
  abs (b + d + 64.226) < 0.001 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l2634_263490


namespace NUMINAMATH_CALUDE_room_population_problem_l2634_263415

theorem room_population_problem (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →  -- Initial ratio of men to women is 4:5
  (initial_men + 2) = 14 →  -- After 2 men entered, there are now 14 men
  (2 * (initial_women - 3)) = 24 :=  -- Number of women after changes
by
  sorry

end NUMINAMATH_CALUDE_room_population_problem_l2634_263415


namespace NUMINAMATH_CALUDE_girls_in_class_l2634_263478

theorem girls_in_class (total : ℕ) (g b t : ℕ) : 
  total = 60 →
  g + b + t = total →
  3 * t = g →
  2 * t = b →
  g = 30 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2634_263478


namespace NUMINAMATH_CALUDE_special_function_property_l2634_263475

/-- A function f: ℤ → ℤ satisfying specific properties -/
def special_function (f : ℤ → ℤ) : Prop :=
  f 0 = 2 ∧ ∀ x : ℤ, f (x + 1) + f (x - 1) = f x * f 1

/-- Theorem stating the property to be proved for the special function -/
theorem special_function_property (f : ℤ → ℤ) (h : special_function f) :
  ∀ x y : ℤ, f (x + y) + f (x - y) = f x * f y := by
  sorry


end NUMINAMATH_CALUDE_special_function_property_l2634_263475


namespace NUMINAMATH_CALUDE_inequality_proof_l2634_263468

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2634_263468


namespace NUMINAMATH_CALUDE_mixture_volume_l2634_263456

/-- Given a mixture of milk and water, prove that the initial volume is 145 liters -/
theorem mixture_volume (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk / initial_water = 3 / 2 →
  initial_milk / (initial_water + 58) = 3 / 4 →
  initial_milk + initial_water = 145 := by
sorry

end NUMINAMATH_CALUDE_mixture_volume_l2634_263456


namespace NUMINAMATH_CALUDE_figure_perimeter_l2634_263487

/-- The figure in the coordinate plane defined by |x + y| + |x - y| = 8 -/
def Figure := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| = 8}

/-- The perimeter of a set in ℝ² -/
noncomputable def perimeter (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem figure_perimeter : perimeter Figure = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_figure_perimeter_l2634_263487


namespace NUMINAMATH_CALUDE_magnitude_of_perpendicular_vector_l2634_263474

/-- Given two planar vectors a and b, where a is perpendicular to b,
    prove that the magnitude of b is √5 --/
theorem magnitude_of_perpendicular_vector
  (a b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b.1 = -2)
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_perpendicular_vector_l2634_263474


namespace NUMINAMATH_CALUDE_sum_of_ratios_ge_six_l2634_263449

theorem sum_of_ratios_ge_six (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z + z / y + y / x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_ge_six_l2634_263449


namespace NUMINAMATH_CALUDE_first_project_length_l2634_263443

/-- Represents the dimensions and duration of an earth digging project -/
structure DiggingProject where
  length : ℝ
  breadth : ℝ
  depth : ℝ
  days : ℝ

/-- The volume of earth dug in a project -/
def volume (p : DiggingProject) : ℝ := p.length * p.breadth * p.depth

/-- The theorem stating that the length of the first digging project is 25 meters -/
theorem first_project_length :
  ∀ (L : ℝ) (n : ℝ),
  let p1 : DiggingProject := { length := L, breadth := 30, depth := 100, days := 12 }
  let p2 : DiggingProject := { length := 20, breadth := 50, depth := 75, days := 12 }
  n > 0 → volume p1 = volume p2 → L = 25 := by
  sorry

end NUMINAMATH_CALUDE_first_project_length_l2634_263443


namespace NUMINAMATH_CALUDE_sara_remaining_marbles_l2634_263460

def initial_black_marbles : ℕ := 792
def marbles_taken : ℕ := 233

theorem sara_remaining_marbles :
  initial_black_marbles - marbles_taken = 559 :=
by sorry

end NUMINAMATH_CALUDE_sara_remaining_marbles_l2634_263460


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_18_l2634_263463

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun difference =>
    ∃ x₁ x₂ : ℝ,
      (|2 * x₁ - 3| = 18) ∧
      (|2 * x₂ - 3| = 18) ∧
      (x₁ ≠ x₂) ∧
      (difference = |x₁ - x₂|) ∧
      (difference = 18)

-- The proof goes here
theorem absolute_value_equation_solution_difference_is_18 :
  absolute_value_equation_solution_difference 18 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_18_l2634_263463


namespace NUMINAMATH_CALUDE_extremum_values_l2634_263457

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem stating that if f(x) has an extremum of 10 at x = 1, then a = 4 and b = -11 -/
theorem extremum_values (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b 1 ≥ f a b x) ∧ 
  (f a b 1 = 10) →
  a = 4 ∧ b = -11 := by
sorry

end NUMINAMATH_CALUDE_extremum_values_l2634_263457


namespace NUMINAMATH_CALUDE_line_up_count_l2634_263422

/-- The number of ways to arrange n distinct objects --/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange k distinct objects from n objects --/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / factorial (n - k)

/-- The number of boys in the group --/
def num_boys : ℕ := 2

/-- The number of girls in the group --/
def num_girls : ℕ := 3

/-- The total number of people in the group --/
def total_people : ℕ := num_boys + num_girls

theorem line_up_count : 
  factorial total_people - factorial (total_people - 1) * factorial num_boys = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_up_count_l2634_263422


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2634_263462

variables (P S M t n : ℝ)

/-- The margin M can be expressed in terms of the selling price S, given the production cost P, tax rate t, and a constant n. -/
theorem margin_in_terms_of_selling_price
  (h1 : S = P * (1 + t/100))  -- Selling price including tax
  (h2 : M = P / n)            -- Margin definition
  (h3 : n > 0)                -- n is positive (implied by the context)
  (h4 : t ≥ 0)                -- Tax rate is non-negative (implied by the context)
  : M = S / (n * (1 + t/100)) :=
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2634_263462


namespace NUMINAMATH_CALUDE_exam_score_97_impossible_l2634_263430

theorem exam_score_97_impossible :
  ¬ ∃ (correct unanswered : ℕ),
    correct + unanswered ≤ 20 ∧
    5 * correct + unanswered = 97 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_97_impossible_l2634_263430


namespace NUMINAMATH_CALUDE_pencil_distribution_l2634_263497

theorem pencil_distribution (total_pencils : ℕ) (num_people : ℕ) 
  (h1 : total_pencils = 24) 
  (h2 : num_people = 3) : 
  total_pencils / num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2634_263497


namespace NUMINAMATH_CALUDE_linear_function_above_x_axis_l2634_263436

/-- A linear function y = ax + a + 2 is above the x-axis for -2 ≤ x ≤ 1 if and only if
    -1 < a < 2 and a ≠ 0 -/
theorem linear_function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → a * x + a + 2 > 0) ↔ (-1 < a ∧ a < 2 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_above_x_axis_l2634_263436


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2634_263471

/-- Represents a tetrahedron SABC with mutually perpendicular edges SA, SB, SC -/
structure Tetrahedron where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  perpendicular : True -- Represents that SA, SB, SC are mutually perpendicular

/-- The radius of the circumscribed sphere of the tetrahedron -/
def circumscribedSphereRadius (t : Tetrahedron) : ℝ :=
  sorry

/-- Determines if there exists a sphere with radius smaller than R that contains the tetrahedron -/
def existsSmallerSphere (t : Tetrahedron) (R : ℝ) : Prop :=
  sorry

theorem tetrahedron_properties (t : Tetrahedron) 
    (h1 : t.SA = 2) (h2 : t.SB = 3) (h3 : t.SC = 6) : 
    circumscribedSphereRadius t = 7/2 ∧ existsSmallerSphere t (7/2) :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2634_263471


namespace NUMINAMATH_CALUDE_range_of_inequality_l2634_263473

-- Define an even function that is monotonically increasing on [0, +∞)
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_monotone : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- Theorem statement
theorem range_of_inequality :
  ∀ x : ℝ, f (2 * x - 1) ≤ f 3 ↔ -1 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_inequality_l2634_263473


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l2634_263493

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by the repeating decimal 0.333... --/
def result : ℚ := 8 / repeating_third

theorem eight_divided_by_repeating_third :
  result = 24 := by sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l2634_263493


namespace NUMINAMATH_CALUDE_zu_chongzhi_complex_theory_incorrect_l2634_263425

-- Define a structure for a scientist-field pairing
structure ScientistFieldPair where
  scientist : String
  field : String

-- Define the list of pairings
def pairings : List ScientistFieldPair := [
  { scientist := "Descartes", field := "Analytic Geometry" },
  { scientist := "Pascal", field := "Probability Theory" },
  { scientist := "Cantor", field := "Set Theory" },
  { scientist := "Zu Chongzhi", field := "Complex Number Theory" }
]

-- Define a function to check if a pairing is correct based on historical contributions
def isCorrectPairing (pair : ScientistFieldPair) : Bool :=
  match pair with
  | { scientist := "Descartes", field := "Analytic Geometry" } => true
  | { scientist := "Pascal", field := "Probability Theory" } => true
  | { scientist := "Cantor", field := "Set Theory" } => true
  | { scientist := "Zu Chongzhi", field := "Complex Number Theory" } => false
  | _ => false

-- Theorem: The pairing of Zu Chongzhi with Complex Number Theory is incorrect
theorem zu_chongzhi_complex_theory_incorrect :
  ∃ pair ∈ pairings, pair.scientist = "Zu Chongzhi" ∧ pair.field = "Complex Number Theory" ∧ ¬(isCorrectPairing pair) :=
by
  sorry

end NUMINAMATH_CALUDE_zu_chongzhi_complex_theory_incorrect_l2634_263425


namespace NUMINAMATH_CALUDE_friends_who_ate_bread_l2634_263413

theorem friends_who_ate_bread (loaves : ℕ) (slices_per_loaf : ℕ) (slices_per_friend : ℕ) :
  loaves = 4 →
  slices_per_loaf = 15 →
  slices_per_friend = 6 →
  (loaves * slices_per_loaf) % slices_per_friend = 0 →
  (loaves * slices_per_loaf) / slices_per_friend = 10 := by
  sorry

end NUMINAMATH_CALUDE_friends_who_ate_bread_l2634_263413


namespace NUMINAMATH_CALUDE_mary_total_spent_approx_l2634_263442

/-- Calculates the total amount Mary spent at the mall --/
def total_spent (shirt_price : ℝ) (shirt_tax : ℝ) 
                (jacket_price : ℝ) (jacket_discount : ℝ) (jacket_tax : ℝ) 
                (currency_rate : ℝ)
                (scarf_price : ℝ) (hat_price : ℝ) (accessories_tax : ℝ) : ℝ :=
  let shirt_total := shirt_price * (1 + shirt_tax)
  let jacket_discounted := jacket_price * (1 - jacket_discount)
  let jacket_total := jacket_discounted * (1 + jacket_tax) * currency_rate
  let accessories_total := (scarf_price + hat_price) * (1 + accessories_tax)
  shirt_total + jacket_total + accessories_total

/-- The theorem stating that Mary's total spent is approximately $49.13 --/
theorem mary_total_spent_approx :
  ∃ ε > 0, abs (total_spent 13.04 0.07 15.34 0.20 0.085 1.28 7.90 9.13 0.065 - 49.13) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_mary_total_spent_approx_l2634_263442


namespace NUMINAMATH_CALUDE_prime_factors_count_l2634_263410

/-- The total number of prime factors in the expression (4)^11 × (7)^3 × (11)^2 -/
def totalPrimeFactors : ℕ := 27

/-- The exponent of 4 in the expression -/
def exponent4 : ℕ := 11

/-- The exponent of 7 in the expression -/
def exponent7 : ℕ := 3

/-- The exponent of 11 in the expression -/
def exponent11 : ℕ := 2

theorem prime_factors_count : 
  totalPrimeFactors = 2 * exponent4 + exponent7 + exponent11 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_l2634_263410


namespace NUMINAMATH_CALUDE_certain_number_proof_l2634_263472

theorem certain_number_proof (x : ℤ) (N : ℝ) 
  (h1 : N * (10 : ℝ)^(x : ℝ) < 220000)
  (h2 : ∀ y : ℤ, y > 5 → N * (10 : ℝ)^(y : ℝ) ≥ 220000) :
  N = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2634_263472


namespace NUMINAMATH_CALUDE_table_and_chair_price_l2634_263450

/-- The price of a chair in dollars -/
def chair_price : ℝ := by sorry

/-- The price of a table in dollars -/
def table_price : ℝ := 52.5

/-- The relation between chair and table prices -/
axiom price_relation : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

theorem table_and_chair_price : table_price + chair_price = 60 := by sorry

end NUMINAMATH_CALUDE_table_and_chair_price_l2634_263450


namespace NUMINAMATH_CALUDE_amount_calculation_l2634_263420

theorem amount_calculation (a b : ℝ) 
  (h1 : a + b = 1210)
  (h2 : (1/3) * a = (1/4) * b) : 
  b = 4840 / 7 := by
  sorry

end NUMINAMATH_CALUDE_amount_calculation_l2634_263420


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_eq_two_thirds_l2634_263470

theorem tan_alpha_2_implies_fraction_eq_two_thirds (α : Real) 
  (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_eq_two_thirds_l2634_263470


namespace NUMINAMATH_CALUDE_last_red_ball_fourth_draw_probability_l2634_263409

def initial_white_balls : ℕ := 8
def initial_red_balls : ℕ := 2
def total_balls : ℕ := initial_white_balls + initial_red_balls
def draws : ℕ := 4

def favorable_outcomes : ℕ := (Nat.choose 3 1) * (Nat.choose initial_white_balls 2)
def total_outcomes : ℕ := Nat.choose total_balls draws

theorem last_red_ball_fourth_draw_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_last_red_ball_fourth_draw_probability_l2634_263409


namespace NUMINAMATH_CALUDE_triangle_base_length_l2634_263407

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 3 → height = 3 → area = (base * height) / 2 → base = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2634_263407


namespace NUMINAMATH_CALUDE_triangular_pyramid_angles_l2634_263429

/-- 
Given a triangular pyramid with lateral surface area S and lateral edge length l,
if the plane angles at the apex form an arithmetic progression with common difference π/6,
then the angles are as specified.
-/
theorem triangular_pyramid_angles (S l : ℝ) (h_positive_S : S > 0) (h_positive_l : l > 0) :
  let α := Real.arcsin ((S * (Real.sqrt 3 - 1)) / l^2)
  ∃ (θ₁ θ₂ θ₃ : ℝ),
    (θ₁ = α - π/6 ∧ θ₂ = α ∧ θ₃ = α + π/6) ∧
    (θ₁ + θ₂ + θ₃ = π/2) ∧
    (θ₃ - θ₂ = θ₂ - θ₁) ∧
    (θ₃ - θ₂ = π/6) ∧
    (S = (l^2 / 2) * (Real.sin θ₁ + Real.sin θ₂ + Real.sin θ₃)) :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_angles_l2634_263429


namespace NUMINAMATH_CALUDE_jills_age_l2634_263441

theorem jills_age (henry_age jill_age : ℕ) : 
  (henry_age + jill_age = 48) →
  (henry_age - 9 = 2 * (jill_age - 9)) →
  jill_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_jills_age_l2634_263441


namespace NUMINAMATH_CALUDE_sum_is_parabola_l2634_263402

-- Define the original parabola
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the reflected parabola
def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + c

-- Define the translated original parabola (f)
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c x + 3

-- Define the translated reflected parabola (g)
def g (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c x - 2

-- Theorem: The sum of f and g is a parabola
theorem sum_is_parabola (a b c : ℝ) :
  ∃ (A C : ℝ), ∀ x, f a b c x + g a b c x = A * x^2 + C :=
sorry

end NUMINAMATH_CALUDE_sum_is_parabola_l2634_263402


namespace NUMINAMATH_CALUDE_blocks_used_in_tower_l2634_263459

/-- Given that Randy initially had 59 blocks and now has 23 blocks left,
    prove that he used 36 blocks to build the tower. -/
theorem blocks_used_in_tower (initial_blocks : ℕ) (remaining_blocks : ℕ) 
  (h1 : initial_blocks = 59)
  (h2 : remaining_blocks = 23) : 
  initial_blocks - remaining_blocks = 36 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_in_tower_l2634_263459


namespace NUMINAMATH_CALUDE_gwens_birthday_money_l2634_263424

/-- The amount of money Gwen spent -/
def amount_spent : ℕ := 8

/-- The amount of money Gwen has left -/
def amount_left : ℕ := 6

/-- The total amount of money Gwen received for her birthday -/
def total_amount : ℕ := amount_spent + amount_left

theorem gwens_birthday_money : total_amount = 14 := by
  sorry

end NUMINAMATH_CALUDE_gwens_birthday_money_l2634_263424


namespace NUMINAMATH_CALUDE_find_A_l2634_263446

/-- Represents a three-digit number of the form 2A3 where A is a single digit -/
def threeDigitNumber (A : Nat) : Nat :=
  200 + 10 * A + 3

/-- Condition that A is a single digit -/
def isSingleDigit (A : Nat) : Prop :=
  A ≥ 0 ∧ A ≤ 9

theorem find_A :
  ∀ A : Nat,
    isSingleDigit A →
    (threeDigitNumber A).mod 11 = 0 →
    A = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_find_A_l2634_263446


namespace NUMINAMATH_CALUDE_half_less_than_product_l2634_263444

theorem half_less_than_product : (1/2 : ℝ) < 2^(1/3) * Real.log 2 / Real.log 3 := by sorry

end NUMINAMATH_CALUDE_half_less_than_product_l2634_263444


namespace NUMINAMATH_CALUDE_cube_of_negative_double_l2634_263458

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_double_l2634_263458


namespace NUMINAMATH_CALUDE_binder_cost_l2634_263431

theorem binder_cost (book_cost : ℕ) (num_binders : ℕ) (num_notebooks : ℕ) 
  (notebook_cost : ℕ) (total_cost : ℕ) : ℕ :=
by
  have h1 : book_cost = 16 := by sorry
  have h2 : num_binders = 3 := by sorry
  have h3 : num_notebooks = 6 := by sorry
  have h4 : notebook_cost = 1 := by sorry
  have h5 : total_cost = 28 := by sorry
  
  have binder_cost : ℕ := (total_cost - (book_cost + num_notebooks * notebook_cost)) / num_binders
  
  exact binder_cost

end NUMINAMATH_CALUDE_binder_cost_l2634_263431


namespace NUMINAMATH_CALUDE_simplify_expression_l2634_263496

theorem simplify_expression (x : ℝ) : x + 3 - 5*x + 6 + 7*x - 2 - 9*x + 8 = -6*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2634_263496


namespace NUMINAMATH_CALUDE_tire_price_proof_l2634_263418

/-- The regular price of a single tire -/
def regular_price : ℝ := 104.17

/-- The discounted price of three tires -/
def discounted_price (p : ℝ) : ℝ := 3 * (0.8 * p)

/-- The price of the fourth tire -/
def fourth_tire_price : ℝ := 5

/-- The total price paid for four tires -/
def total_price : ℝ := 255

/-- Theorem stating that the regular price of a tire is approximately 104.17 dollars 
    given the discount and total price conditions -/
theorem tire_price_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  discounted_price regular_price + fourth_tire_price = total_price - ε :=
sorry

end NUMINAMATH_CALUDE_tire_price_proof_l2634_263418


namespace NUMINAMATH_CALUDE_equilateral_triangle_circumradius_ratio_l2634_263448

/-- Given two equilateral triangles with side lengths B and b (B ≠ b) and circumradii S and s respectively,
    the ratio of their circumradii S/s is always equal to the ratio of their side lengths B/b. -/
theorem equilateral_triangle_circumradius_ratio 
  (B b S s : ℝ) 
  (hB : B > 0) 
  (hb : b > 0) 
  (hne : B ≠ b) 
  (hS : S = B * Real.sqrt 3 / 3) 
  (hs : s = b * Real.sqrt 3 / 3) : 
  S / s = B / b := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circumradius_ratio_l2634_263448


namespace NUMINAMATH_CALUDE_cake_price_problem_l2634_263491

theorem cake_price_problem (original_price : ℝ) : 
  (8 * original_price = 320) → 
  (10 * (0.8 * original_price) = 320) → 
  original_price = 40 := by
sorry

end NUMINAMATH_CALUDE_cake_price_problem_l2634_263491


namespace NUMINAMATH_CALUDE_ellipse_condition_l2634_263404

-- Define the condition for an ellipse with foci on the x-axis
def is_ellipse_x_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

-- State the theorem
theorem ellipse_condition (m n : ℝ) :
  is_ellipse_x_axis m n ↔ n > m ∧ m > 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2634_263404


namespace NUMINAMATH_CALUDE_circle_properties_l2634_263455

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (0, 1)

theorem circle_properties :
  ∀ x y : ℝ,
    (line1 x y ∧ line2 x y → (x, y) = center) ∧
    circle_equation 1 0 ∧
    (∀ a b : ℝ, (a - center.1)^2 + (b - center.2)^2 = 2 ↔ circle_equation a b) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2634_263455


namespace NUMINAMATH_CALUDE_line_parametric_equation_l2634_263445

/-- Parametric equation of a line passing through (1, 5) with slope angle π/3 -/
theorem line_parametric_equation :
  let M : ℝ × ℝ := (1, 5)
  let slope_angle : ℝ := π / 3
  let parametric_equation (t : ℝ) : ℝ × ℝ :=
    (M.1 + t * Real.cos slope_angle, M.2 + t * Real.sin slope_angle)
  ∀ t : ℝ, parametric_equation t = (1 + (1/2) * t, 5 + (Real.sqrt 3 / 2) * t) :=
by sorry

end NUMINAMATH_CALUDE_line_parametric_equation_l2634_263445


namespace NUMINAMATH_CALUDE_food_fraction_is_one_fifth_l2634_263488

def salary : ℚ := 150000.00000000003
def house_rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5
def amount_left : ℚ := 15000

theorem food_fraction_is_one_fifth :
  let food_fraction := 1 - house_rent_fraction - clothes_fraction - amount_left / salary
  food_fraction = 1/5 := by sorry

end NUMINAMATH_CALUDE_food_fraction_is_one_fifth_l2634_263488


namespace NUMINAMATH_CALUDE_tangent_line_slope_intercept_difference_l2634_263452

/-- A line passing through two points and tangent to a circle -/
structure TangentLine where
  a : ℝ
  b : ℝ
  passes_through_first : 7 = a * 5 + b
  passes_through_second : 20 = a * 9 + b
  tangent_at : (5, 7) ∈ {(x, y) | y = a * x + b}

/-- The difference between the slope and y-intercept of the tangent line -/
def slope_intercept_difference (line : TangentLine) : ℝ := line.a - line.b

/-- Theorem stating that the difference between slope and y-intercept is 12.5 -/
theorem tangent_line_slope_intercept_difference :
  ∀ (line : TangentLine), slope_intercept_difference line = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_intercept_difference_l2634_263452


namespace NUMINAMATH_CALUDE_system_solution_l2634_263447

theorem system_solution (a b c : ℝ) : 
  a^2 + a*b + c^2 = 31 ∧ 
  b^2 + a*b - c^2 = 18 ∧ 
  a^2 - b^2 = 7 → 
  c = Real.sqrt 3 ∨ c = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2634_263447


namespace NUMINAMATH_CALUDE_secret_codes_count_l2634_263498

/-- The number of colors available for the secret code -/
def num_colors : ℕ := 7

/-- The number of slots in the secret code -/
def num_slots : ℕ := 4

/-- The number of possible secret codes -/
def num_codes : ℕ := num_colors ^ num_slots

/-- Theorem: The number of possible secret codes is 2401 -/
theorem secret_codes_count : num_codes = 2401 := by
  sorry

end NUMINAMATH_CALUDE_secret_codes_count_l2634_263498


namespace NUMINAMATH_CALUDE_bennys_books_l2634_263414

/-- Given the number of books Sandy, Tim, and the total, find Benny's books --/
theorem bennys_books (sandy_books tim_books total_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : total_books = 67)
  (h4 : total_books = sandy_books + tim_books + benny_books) :
  benny_books = 24 := by
  sorry

end NUMINAMATH_CALUDE_bennys_books_l2634_263414


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2634_263451

/-- Line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Circle in polar form -/
structure PolarCircle where
  ρ : ℝ → ℝ

/-- Chord length calculation -/
def chordLength (l : ParametricLine) (c : PolarCircle) : ℝ := sorry

/-- Main theorem -/
theorem intersection_chord_length :
  let l : ParametricLine := { x := fun t => t + 1, y := fun t => t - 3 }
  let c : PolarCircle := { ρ := fun θ => 4 * Real.cos θ }
  chordLength l c = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2634_263451


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2634_263416

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The problem statement -/
theorem line_through_point_parallel_to_line :
  let l1 : Line := { a := 3, b := 1, c := -2 }
  let l2 : Line := { a := 3, b := 1, c := -5 }
  l2.contains 2 (-1) ∧ Line.parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2634_263416


namespace NUMINAMATH_CALUDE_existence_of_triangle_l2634_263428

theorem existence_of_triangle (l : Fin 7 → ℝ) 
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    l i + l j > l k ∧ 
    l j + l k > l i ∧ 
    l k + l i > l j :=
sorry

end NUMINAMATH_CALUDE_existence_of_triangle_l2634_263428


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2634_263483

theorem quadratic_function_theorem (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^2 + b * x) →
  (f 0 = 0) →
  (∀ x, f (x + 1) = f x + x + 1) →
  (a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2634_263483


namespace NUMINAMATH_CALUDE_pineapple_profit_l2634_263423

/-- Calculates Jonah's profit from selling pineapples --/
theorem pineapple_profit : 
  let num_pineapples : ℕ := 6
  let price_per_pineapple : ℚ := 3
  let discount_rate : ℚ := 0.2
  let discount_threshold : ℕ := 4
  let rings_per_pineapple : ℕ := 10
  let price_per_two_rings : ℚ := 5
  let price_per_four_ring_set : ℚ := 16

  let total_cost : ℚ := if num_pineapples > discount_threshold
    then num_pineapples * price_per_pineapple * (1 - discount_rate)
    else num_pineapples * price_per_pineapple

  let total_rings : ℕ := num_pineapples * rings_per_pineapple
  let revenue_from_two_rings : ℚ := price_per_two_rings
  let remaining_rings : ℕ := total_rings - 2
  let full_sets : ℕ := remaining_rings / 4
  let revenue_from_sets : ℚ := full_sets * price_per_four_ring_set

  let total_revenue : ℚ := revenue_from_two_rings + revenue_from_sets
  let profit : ℚ := total_revenue - total_cost

  profit = 219.6 := by sorry

end NUMINAMATH_CALUDE_pineapple_profit_l2634_263423


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2634_263439

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  1 / a + 1 / b ≥ 4 ∧ (1 / a + 1 / b = 4 ↔ a = 1 / 2 ∧ b = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2634_263439


namespace NUMINAMATH_CALUDE_ivan_pension_sufficient_for_ticket_l2634_263466

theorem ivan_pension_sufficient_for_ticket : 
  (149^6 - 199^3) / (149^4 + 199^2 + 199 * 149^2) > 22000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_pension_sufficient_for_ticket_l2634_263466


namespace NUMINAMATH_CALUDE_range_of_a_l2634_263406

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (ha : a ∈ A) : a ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2634_263406


namespace NUMINAMATH_CALUDE_power_product_equals_reciprocal_l2634_263494

theorem power_product_equals_reciprocal (n : ℕ) :
  (125 : ℚ) ^ (2015 : ℕ) * (-1/125 : ℚ) ^ (2016 : ℕ) = 1/125 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_reciprocal_l2634_263494


namespace NUMINAMATH_CALUDE_cubic_factorization_l2634_263492

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m+2)*(m-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2634_263492


namespace NUMINAMATH_CALUDE_area_increase_rect_to_circle_l2634_263437

/-- Increase in area when changing a rectangular field to a circular field -/
theorem area_increase_rect_to_circle (length width : ℝ) (h1 : length = 60) (h2 : width = 20) :
  let rect_area := length * width
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let circle_area := Real.pi * radius^2
  ∃ ε > 0, abs (circle_area - rect_area - 837.94) < ε :=
by sorry

end NUMINAMATH_CALUDE_area_increase_rect_to_circle_l2634_263437


namespace NUMINAMATH_CALUDE_quadratic_max_condition_l2634_263454

/-- Given a quadratic function y = ax² - 2ax + c with a ≠ 0 and maximum value 2,
    prove that c - a = 2 and a < 0 -/
theorem quadratic_max_condition (a c : ℝ) (h1 : a ≠ 0) :
  (∀ x, a * x^2 - 2*a*x + c ≤ 2) ∧ 
  (∃ x, a * x^2 - 2*a*x + c = 2) →
  c - a = 2 ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_condition_l2634_263454


namespace NUMINAMATH_CALUDE_circle_radius_l2634_263411

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 16

-- Define the ellipse equation (not used in the proof, but included for completeness)
def ellipse_equation (x y : ℝ) : Prop :=
  (x - 2)^2 / 25 + (y - 1)^2 / 9 = 1

-- Theorem: The radius of the circle is 4
theorem circle_radius : ∃ (r : ℝ), r = 4 ∧ ∀ (x y : ℝ), circle_equation x y ↔ (x - 2)^2 + (y - 1)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l2634_263411


namespace NUMINAMATH_CALUDE_cubic_equation_q_expression_l2634_263476

theorem cubic_equation_q_expression (a b q r : ℝ) (h1 : b ≠ 0) :
  (∃ (x : ℂ), x^3 + q*x + r = 0 ∧ (x = a + b*I ∨ x = a - b*I)) →
  q = b^2 - 3*a^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_q_expression_l2634_263476


namespace NUMINAMATH_CALUDE_factorization_equality_l2634_263465

theorem factorization_equality (x y a b : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_factorization_equality_l2634_263465


namespace NUMINAMATH_CALUDE_eighth_grade_percentage_combined_schools_combined_schools_eighth_grade_percentage_l2634_263427

theorem eighth_grade_percentage_combined_schools : ℝ → Prop :=
  fun p =>
    let pinecrest_total : ℕ := 160
    let mapleridge_total : ℕ := 250
    let pinecrest_eighth_percent : ℝ := 18
    let mapleridge_eighth_percent : ℝ := 22
    let pinecrest_eighth : ℝ := (pinecrest_eighth_percent / 100) * pinecrest_total
    let mapleridge_eighth : ℝ := (mapleridge_eighth_percent / 100) * mapleridge_total
    let total_eighth : ℝ := pinecrest_eighth + mapleridge_eighth
    let total_students : ℝ := pinecrest_total + mapleridge_total
    p = (total_eighth / total_students) * 100 ∧ p = 20

/-- The percentage of 8th grade students in both schools combined is 20%. -/
theorem combined_schools_eighth_grade_percentage :
  ∃ p, eighth_grade_percentage_combined_schools p :=
sorry

end NUMINAMATH_CALUDE_eighth_grade_percentage_combined_schools_combined_schools_eighth_grade_percentage_l2634_263427


namespace NUMINAMATH_CALUDE_power_8_2048_mod_50_l2634_263440

theorem power_8_2048_mod_50 : 8^2048 % 50 = 38 := by sorry

end NUMINAMATH_CALUDE_power_8_2048_mod_50_l2634_263440


namespace NUMINAMATH_CALUDE_group_size_proof_l2634_263419

theorem group_size_proof (average_increase : ℝ) (new_weight : ℝ) (old_weight : ℝ) :
  average_increase = 6 →
  new_weight = 88 →
  old_weight = 40 →
  (average_increase * (new_weight - old_weight) / average_increase : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2634_263419


namespace NUMINAMATH_CALUDE_positive_sum_product_l2634_263438

theorem positive_sum_product (a b c : ℝ) 
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : 
  a * b + b * c + c * a + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_product_l2634_263438
