import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l964_96495

theorem smallest_n_multiple_of_five (x y : ℤ) 
  (hx : 5 ∣ (x - 2)) 
  (hy : 5 ∣ (y + 4)) : 
  ∃ n : ℕ+, 
    5 ∣ (x^2 + 2*x*y + y^2 + n) ∧ 
    ∀ m : ℕ+, (5 ∣ (x^2 + 2*x*y + y^2 + m) → n ≤ m) ∧
    n = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l964_96495


namespace NUMINAMATH_CALUDE_inscribed_polygon_radius_l964_96424

/-- A 12-sided convex polygon inscribed in a circle -/
structure InscribedPolygon where
  /-- The number of sides of the polygon -/
  sides : ℕ
  /-- The number of sides with length √2 -/
  short_sides : ℕ
  /-- The number of sides with length √24 -/
  long_sides : ℕ
  /-- The length of the short sides -/
  short_length : ℝ
  /-- The length of the long sides -/
  long_length : ℝ
  /-- Condition: The polygon has 12 sides -/
  sides_eq : sides = 12
  /-- Condition: There are 6 short sides -/
  short_sides_eq : short_sides = 6
  /-- Condition: There are 6 long sides -/
  long_sides_eq : long_sides = 6
  /-- Condition: The short sides have length √2 -/
  short_length_eq : short_length = Real.sqrt 2
  /-- Condition: The long sides have length √24 -/
  long_length_eq : long_length = Real.sqrt 24

/-- The theorem stating that the radius of the circle is 4√2 -/
theorem inscribed_polygon_radius (p : InscribedPolygon) : 
  ∃ (r : ℝ), r = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polygon_radius_l964_96424


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l964_96422

/-- Given a rational function decomposition, prove the value of B -/
theorem partial_fraction_decomposition (x A B C : ℝ) : 
  (2 : ℝ) / (x^3 + 5*x^2 - 13*x - 35) = A / (x-7) + B / (x+1) + C / (x+1)^2 →
  x^3 + 5*x^2 - 13*x - 35 = (x-7)*(x+1)^2 →
  B = (1 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l964_96422


namespace NUMINAMATH_CALUDE_percentage_same_grade_l964_96454

def total_students : ℕ := 40
def students_with_all_As : ℕ := 3
def students_with_all_Bs : ℕ := 2
def students_with_all_Cs : ℕ := 6
def students_with_all_Ds : ℕ := 1

def students_with_same_grade : ℕ := 
  students_with_all_As + students_with_all_Bs + students_with_all_Cs + students_with_all_Ds

theorem percentage_same_grade : 
  (students_with_same_grade : ℚ) / total_students * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_same_grade_l964_96454


namespace NUMINAMATH_CALUDE_fraction_ordering_l964_96453

theorem fraction_ordering : 
  let a : ℚ := 6 / 29
  let b : ℚ := 8 / 31
  let c : ℚ := 10 / 39
  a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l964_96453


namespace NUMINAMATH_CALUDE_sum_of_specific_S_values_l964_96439

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S_values : S 17 + S 33 + S 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_values_l964_96439


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l964_96414

/-- A quadrilateral with given side lengths -/
structure Quadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)

/-- The largest possible inscribed circle in a quadrilateral -/
def largest_inscribed_circle (q : Quadrilateral) : ℝ := sorry

/-- Theorem: The radius of the largest inscribed circle in the given quadrilateral is 2√6 -/
theorem largest_inscribed_circle_radius :
  ∀ q : Quadrilateral,
    q.AB = 15 ∧ q.BC = 10 ∧ q.CD = 8 ∧ q.DA = 13 →
    largest_inscribed_circle q = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l964_96414


namespace NUMINAMATH_CALUDE_plan2_better_l964_96436

/-- The number of optional questions -/
def total_questions : ℕ := 5

/-- The number of questions Student A can solve -/
def solvable_questions : ℕ := 3

/-- Probability of participating under Plan 1 -/
def prob_plan1 : ℚ := solvable_questions / total_questions

/-- Probability of participating under Plan 2 -/
def prob_plan2 : ℚ := (Nat.choose solvable_questions 2 * Nat.choose (total_questions - solvable_questions) 1 + 
                       Nat.choose solvable_questions 3) / 
                      Nat.choose total_questions 3

/-- Theorem stating that Plan 2 gives a higher probability for Student A -/
theorem plan2_better : prob_plan2 > prob_plan1 := by
  sorry

end NUMINAMATH_CALUDE_plan2_better_l964_96436


namespace NUMINAMATH_CALUDE_infinite_product_l964_96473

open Set Filter

-- Define the concept of a function being infinite at a point
def IsInfiniteAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ K > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| ∧ |x - x₀| < δ → |f x| > K

-- Define the theorem
theorem infinite_product (f g : ℝ → ℝ) (x₀ M : ℝ) (hM : M > 0)
    (hg : ∀ x, |x - x₀| > 0 → |g x| ≥ M)
    (hf : IsInfiniteAt f x₀) :
    IsInfiniteAt (fun x ↦ f x * g x) x₀ := by
  sorry


end NUMINAMATH_CALUDE_infinite_product_l964_96473


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l964_96462

/-- Proves that for a rectangular plot with given conditions, the length is 20 meters more than the breadth. -/
theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 60 ∧ 
  length > breadth ∧ 
  2 * (length + breadth) * 26.5 = 5300 → 
  length - breadth = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l964_96462


namespace NUMINAMATH_CALUDE_half_day_division_count_l964_96417

/-- The number of seconds in a half-day -/
def half_day_seconds : ℕ := 43200

/-- The number of ways to divide a half-day into periods -/
def num_divisions : ℕ := 60

/-- Theorem: The number of ordered pairs of positive integers (n, m) 
    satisfying n * m = half_day_seconds is equal to num_divisions -/
theorem half_day_division_count :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = half_day_seconds ∧ 
                                   p.1 > 0 ∧ p.2 > 0) 
                 (Finset.product (Finset.range (half_day_seconds + 1)) 
                                 (Finset.range (half_day_seconds + 1)))).card = num_divisions :=
sorry

end NUMINAMATH_CALUDE_half_day_division_count_l964_96417


namespace NUMINAMATH_CALUDE_number_tower_pattern_l964_96483

theorem number_tower_pattern (n : ℕ) : (10^n - 1) * 9 + (n + 1) = 10^(n+1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_number_tower_pattern_l964_96483


namespace NUMINAMATH_CALUDE_field_trip_buses_l964_96485

/-- The number of classrooms in the school -/
def num_classrooms : ℕ := 67

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 66

/-- The number of seats in each bus -/
def seats_per_bus : ℕ := 6

/-- The function to calculate the minimum number of buses needed -/
def min_buses_needed (classrooms : ℕ) (students : ℕ) (seats : ℕ) : ℕ :=
  (classrooms * students + seats - 1) / seats

/-- Theorem stating the minimum number of buses needed for the field trip -/
theorem field_trip_buses :
  min_buses_needed num_classrooms students_per_classroom seats_per_bus = 738 := by
  sorry


end NUMINAMATH_CALUDE_field_trip_buses_l964_96485


namespace NUMINAMATH_CALUDE_sum_of_abc_equals_45_l964_96401

-- Define a triangle with side lengths 3, 7, and x
structure Triangle where
  x : ℝ
  side1 : ℝ := 3
  side2 : ℝ := 7
  side3 : ℝ := x

-- Define the property of angles in arithmetic progression
def anglesInArithmeticProgression (t : Triangle) : Prop := sorry

-- Define the sum of possible values of x
def sumOfPossibleX (t : Triangle) : ℝ := sorry

-- Define a, b, and c as positive integers
def a : ℕ+ := sorry
def b : ℕ+ := sorry
def c : ℕ+ := sorry

-- Theorem statement
theorem sum_of_abc_equals_45 (t : Triangle) 
  (h1 : anglesInArithmeticProgression t) 
  (h2 : sumOfPossibleX t = a + Real.sqrt b + Real.sqrt c) : 
  a + b + c = 45 := by sorry

end NUMINAMATH_CALUDE_sum_of_abc_equals_45_l964_96401


namespace NUMINAMATH_CALUDE_rob_baseball_cards_l964_96467

theorem rob_baseball_cards (rob_total : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) :
  rob_doubles * 3 = rob_total →
  jess_doubles = rob_doubles * 5 →
  jess_doubles = 40 →
  rob_total = 24 := by
sorry

end NUMINAMATH_CALUDE_rob_baseball_cards_l964_96467


namespace NUMINAMATH_CALUDE_function_property_l964_96412

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f)
    (h_sym : ∀ x, f (3/2 + x) = -f (3/2 - x))
    (h_f1 : f 1 = 2) : 
  f 2 + f 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l964_96412


namespace NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l964_96410

theorem factory_temporary_workers_percentage 
  (total_workers : ℕ) 
  (technician_percentage : ℚ) 
  (non_technician_percentage : ℚ) 
  (permanent_technician_percentage : ℚ) 
  (permanent_non_technician_percentage : ℚ) 
  (h1 : technician_percentage = 80 / 100)
  (h2 : non_technician_percentage = 20 / 100)
  (h3 : permanent_technician_percentage = 80 / 100)
  (h4 : permanent_non_technician_percentage = 20 / 100)
  (h5 : technician_percentage + non_technician_percentage = 1) :
  let permanent_workers := (technician_percentage * permanent_technician_percentage + 
                            non_technician_percentage * permanent_non_technician_percentage) * total_workers
  let temporary_workers := total_workers - permanent_workers
  temporary_workers / total_workers = 32 / 100 := by
sorry

end NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l964_96410


namespace NUMINAMATH_CALUDE_f_range_and_triangle_area_l964_96460

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * (Real.cos x)^2 + Real.sin x * Real.cos x

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ A < Real.pi ∧
  B > 0 ∧ B < Real.pi ∧
  C > 0 ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem f_range_and_triangle_area 
  (h1 : ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ∈ Set.Icc 0 (1 + Real.sqrt 3 / 2))
  (h2 : ∃ A B C a b c, 
    triangle_ABC a b c A B C ∧ 
    f (A / 2) = Real.sqrt 3 ∧
    a = 4 ∧
    b + c = 5) :
  ∃ A B C a b c, 
    triangle_ABC a b c A B C ∧
    f (A / 2) = Real.sqrt 3 ∧
    a = 4 ∧
    b + c = 5 ∧
    (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_and_triangle_area_l964_96460


namespace NUMINAMATH_CALUDE_honor_roll_fraction_l964_96423

theorem honor_roll_fraction (total_students : ℝ) (female_students : ℝ) (male_students : ℝ) 
  (female_honor : ℝ) (male_honor : ℝ) :
  female_students = (2 / 5) * total_students →
  male_students = (3 / 5) * total_students →
  female_honor = (5 / 6) * female_students →
  male_honor = (2 / 3) * male_students →
  (female_honor + male_honor) / total_students = 11 / 15 := by
sorry

end NUMINAMATH_CALUDE_honor_roll_fraction_l964_96423


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l964_96498

theorem isosceles_triangle_height (s : ℝ) (h : ℝ) : 
  (1/2 : ℝ) * s * h = 2 * s^2 → h = 4 * s := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l964_96498


namespace NUMINAMATH_CALUDE_stratified_sample_class_size_l964_96490

/-- Given two classes with a total of 100 students, if a stratified random sample
    of 10 students contains 4 from one class, then the other class has 60 students. -/
theorem stratified_sample_class_size (total : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total = 100 →
  sample_size = 10 →
  class_a_sample = 4 →
  (total - (class_a_sample * total / sample_size) : ℕ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_class_size_l964_96490


namespace NUMINAMATH_CALUDE_students_left_unassigned_l964_96427

/-- The number of students left unassigned to groups in a school with specific classroom distributions -/
theorem students_left_unassigned (total_students : ℕ) (num_classrooms : ℕ) 
  (classroom_A : ℕ) (classroom_B : ℕ) (classroom_C : ℕ) (classroom_D : ℕ) 
  (num_groups : ℕ) : 
  total_students = 128 →
  num_classrooms = 4 →
  classroom_A = 37 →
  classroom_B = 31 →
  classroom_C = 25 →
  classroom_D = 35 →
  num_groups = 9 →
  classroom_A + classroom_B + classroom_C + classroom_D = total_students →
  total_students - (num_groups * (total_students / num_groups)) = 2 := by
  sorry

#eval 128 - (9 * (128 / 9))  -- This should evaluate to 2

end NUMINAMATH_CALUDE_students_left_unassigned_l964_96427


namespace NUMINAMATH_CALUDE_total_pencils_l964_96429

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l964_96429


namespace NUMINAMATH_CALUDE_inequality_proof_l964_96437

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l964_96437


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_congruence_l964_96488

/-- Given two congruent isosceles right triangles sharing a common base,
    if one leg of one triangle is 12, then the corresponding leg of the other triangle is 6√2 -/
theorem isosceles_right_triangle_congruence (a b c d : ℝ) :
  a = b ∧                    -- Triangle 1 is isosceles
  c = d ∧                    -- Triangle 2 is isosceles
  a^2 + a^2 = b^2 ∧          -- Triangle 1 is right-angled (Pythagorean theorem)
  c^2 + c^2 = d^2 ∧          -- Triangle 2 is right-angled (Pythagorean theorem)
  b = d ∧                    -- Triangles share a common base
  a = 12                     -- Given leg length in Triangle 1
  → c = 6 * Real.sqrt 2      -- To prove: corresponding leg in Triangle 2
:= by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_congruence_l964_96488


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l964_96416

theorem decimal_to_fraction : (2.75 : ℚ) = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l964_96416


namespace NUMINAMATH_CALUDE_subtraction_multiplication_addition_l964_96474

theorem subtraction_multiplication_addition (x : ℤ) : 
  423 - x = 421 → (x * 423) + 421 = 1267 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_addition_l964_96474


namespace NUMINAMATH_CALUDE_chairs_to_remove_l964_96425

/-- Given a conference hall setup with the following conditions:
  - Each row has 15 chairs
  - Initially, there are 195 chairs
  - 120 attendees are expected
  - All rows must be complete
  - The number of remaining chairs must be the smallest multiple of 15 that is greater than or equal to 120
  
  This theorem proves that the number of chairs to be removed is 60. -/
theorem chairs_to_remove (chairs_per_row : ℕ) (initial_chairs : ℕ) (expected_attendees : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : initial_chairs = 195)
  (h3 : expected_attendees = 120)
  (h4 : ∃ (n : ℕ), n * chairs_per_row ≥ expected_attendees ∧
        ∀ (m : ℕ), m * chairs_per_row ≥ expected_attendees → n ≤ m) :
  initial_chairs - (chairs_per_row * (initial_chairs / chairs_per_row)) = 60 :=
sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l964_96425


namespace NUMINAMATH_CALUDE_solve_email_problem_l964_96484

def email_problem (initial_delete : ℕ) (first_receive : ℕ) (second_delete : ℕ) (final_receive : ℕ) (final_count : ℕ) : Prop :=
  ∃ (x : ℕ), 
    initial_delete = 50 ∧
    first_receive = 15 ∧
    second_delete = 20 ∧
    final_receive = 10 ∧
    final_count = 30 ∧
    first_receive + x + final_receive = final_count ∧
    x = 5

theorem solve_email_problem :
  ∃ (initial_delete first_receive second_delete final_receive final_count : ℕ),
    email_problem initial_delete first_receive second_delete final_receive final_count :=
by
  sorry

end NUMINAMATH_CALUDE_solve_email_problem_l964_96484


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l964_96434

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → min a (min b c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l964_96434


namespace NUMINAMATH_CALUDE_expression_value_l964_96411

theorem expression_value : (35 + 12)^2 - (12^2 + 35^2 - 2 * 12 * 35) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l964_96411


namespace NUMINAMATH_CALUDE_g_512_minus_g_256_eq_zero_l964_96441

-- Define σ(n) as the sum of all positive divisors of n
def σ (n : ℕ+) : ℕ := sorry

-- Define g(n) = 2σ(n)/n
def g (n : ℕ+) : ℚ := 2 * (σ n) / n

-- Theorem statement
theorem g_512_minus_g_256_eq_zero : g 512 - g 256 = 0 := by sorry

end NUMINAMATH_CALUDE_g_512_minus_g_256_eq_zero_l964_96441


namespace NUMINAMATH_CALUDE_vector_relation_l964_96494

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variable (P A B C : V)

/-- Given that PA + 2PB + 3PC = 0, prove that AP = (1/3)AB + (1/2)AC -/
theorem vector_relation (h : (A - P) + 2 • (B - P) + 3 • (C - P) = 0) :
  P - A = (1/3) • (B - A) + (1/2) • (C - A) := by sorry

end NUMINAMATH_CALUDE_vector_relation_l964_96494


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l964_96448

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that the hexagon is equilateral
  is_equilateral : True
  -- Three nonadjacent acute interior angles measure 45°
  has_three_45deg_angles : True
  -- The enclosed area of the hexagon is 12√3
  area_eq_12root3 : side^2 * (3 * Real.sqrt 2 / 4 + Real.sqrt 3 / 2 - Real.sqrt 6 / 4) = 12 * Real.sqrt 3

/-- The perimeter of a SpecialHexagon is 24 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : h.side * 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l964_96448


namespace NUMINAMATH_CALUDE_power_product_squared_l964_96463

theorem power_product_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l964_96463


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l964_96407

theorem smallest_staircase_steps (n : ℕ) : 
  (n > 15) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l964_96407


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l964_96430

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a + 2) * (a - 3) = 20 ∧ (b + 2) * (b - 3) = 20 ∧ a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l964_96430


namespace NUMINAMATH_CALUDE_max_product_of_three_integers_l964_96402

theorem max_product_of_three_integers (a b c : ℕ+) : 
  (a * b * c = 8 * (a + b + c)) → (c = a + b) → (a * b * c ≤ 272) := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_three_integers_l964_96402


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l964_96472

theorem no_infinite_prime_sequence :
  ¬ ∃ (p : ℕ → ℕ), (∀ k, p (k + 1) = 5 * p k + 4) ∧ (∀ n, Nat.Prime (p n)) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l964_96472


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l964_96431

theorem smallest_number_divisible (n : ℕ) : 
  (∃ (k : ℕ), n - k = 44 ∧ 
   9 ∣ (n - k) ∧ 
   6 ∣ (n - k) ∧ 
   12 ∣ (n - k) ∧ 
   18 ∣ (n - k)) →
  (∀ (m : ℕ), m < n → 
    ¬(∃ (k : ℕ), m - k = 44 ∧ 
      9 ∣ (m - k) ∧ 
      6 ∣ (m - k) ∧ 
      12 ∣ (m - k) ∧ 
      18 ∣ (m - k))) →
  n = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l964_96431


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l964_96486

/-- Hyperbola struct -/
structure Hyperbola where
  F₁ : ℝ × ℝ  -- First focus
  F₂ : ℝ × ℝ  -- Second focus
  e : ℝ        -- Eccentricity

/-- Point on the hyperbola -/
def Point : Type := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Length of the real axis of a hyperbola -/
def realAxisLength (h : Hyperbola) : ℝ := sorry

/-- Theorem statement -/
theorem hyperbola_real_axis_length 
  (C : Hyperbola) 
  (P : Point) 
  (h_eccentricity : C.e = Real.sqrt 5)
  (h_point_on_hyperbola : P ∈ {p : Point | distance p C.F₁ - distance p C.F₂ = realAxisLength C})
  (h_distance_ratio : 2 * distance P C.F₁ = 3 * distance P C.F₂)
  (h_triangle_area : triangleArea P C.F₁ C.F₂ = 2 * Real.sqrt 5) :
  realAxisLength C = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l964_96486


namespace NUMINAMATH_CALUDE_thabo_hardcover_books_l964_96478

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (books : BookCollection) : Prop :=
  books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 280 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction

theorem thabo_hardcover_books (books : BookCollection) 
  (h : is_valid_collection books) : books.hardcover_nonfiction = 55 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_books_l964_96478


namespace NUMINAMATH_CALUDE_car_selection_problem_l964_96447

theorem car_selection_problem (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ)
  (h_num_cars : num_cars = 15)
  (h_selections_per_car : selections_per_car = 3)
  (h_cars_per_client : cars_per_client = 3) :
  (num_cars * selections_per_car) / cars_per_client = 15 := by
  sorry

end NUMINAMATH_CALUDE_car_selection_problem_l964_96447


namespace NUMINAMATH_CALUDE_younger_brother_height_l964_96471

def father_height : ℕ := 172
def height_diff_father_minkyung : ℕ := 35
def height_diff_minkyung_brother : ℕ := 28

theorem younger_brother_height :
  father_height - height_diff_father_minkyung - height_diff_minkyung_brother = 109 :=
by sorry

end NUMINAMATH_CALUDE_younger_brother_height_l964_96471


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l964_96408

/-- The distance from the center of the circle (x+4)^2 + (y-3)^2 = 9 to the line 4x + 3y - 1 = 0 is 8/5 -/
theorem distance_circle_center_to_line : 
  let circle := fun (x y : ℝ) => (x + 4)^2 + (y - 3)^2 = 9
  let line := fun (x y : ℝ) => 4*x + 3*y - 1 = 0
  let center := (-4, 3)
  abs (4 * center.1 + 3 * center.2 - 1) / Real.sqrt (4^2 + 3^2) = 8/5 := by
sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l964_96408


namespace NUMINAMATH_CALUDE_x_y_existence_l964_96493

theorem x_y_existence : ∃ (x y : ℝ), x / 7 = 5 / 14 ∧ x / 7 + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_y_existence_l964_96493


namespace NUMINAMATH_CALUDE_divisibility_in_chosen_numbers_l964_96470

theorem divisibility_in_chosen_numbers (n : ℕ+) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range (2*n + 1) → S.card = n + 1 →
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ b % a = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_in_chosen_numbers_l964_96470


namespace NUMINAMATH_CALUDE_parabola_equation_l964_96456

/-- The equation of a parabola given its parametric form -/
theorem parabola_equation (t : ℝ) :
  let x : ℝ := 3 * t + 6
  let y : ℝ := 5 * t^2 - 7
  y = (5/9) * x^2 - (20/3) * x + 13 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l964_96456


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l964_96468

theorem sqrt_equation_solution :
  ∃ y : ℚ, (40 : ℚ) / 60 = Real.sqrt (y / 60) → y = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l964_96468


namespace NUMINAMATH_CALUDE_seven_couples_handshakes_l964_96479

/-- The number of handshakes in a gathering of couples -/
def handshakes (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 7 couples, where each person shakes hands with everyone
    except their spouse and one other person, the total number of handshakes is 77. -/
theorem seven_couples_handshakes :
  handshakes 7 = 77 := by
  sorry

end NUMINAMATH_CALUDE_seven_couples_handshakes_l964_96479


namespace NUMINAMATH_CALUDE_correct_equation_proof_l964_96476

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def has_roots (a b c : ℝ) (r₁ r₂ : ℝ) : Prop :=
  quadratic_equation a b c r₁ = 0 ∧ quadratic_equation a b c r₂ = 0

theorem correct_equation_proof :
  ∃ (a₁ b₁ c₁ : ℝ) (a₂ b₂ c₂ : ℝ),
    has_roots a₁ b₁ c₁ 8 2 ∧
    has_roots a₂ b₂ c₂ (-9) (-1) ∧
    (a₁ = 1 ∧ b₁ = -10 ∧ c₁ ≠ 9) ∧
    (a₂ = 1 ∧ b₂ ≠ -10 ∧ c₂ = 9) ∧
    quadratic_equation 1 (-10) 9 = quadratic_equation a₁ b₁ c₁ ∧
    quadratic_equation 1 (-10) 9 = quadratic_equation a₂ b₂ c₂ :=
by sorry

end NUMINAMATH_CALUDE_correct_equation_proof_l964_96476


namespace NUMINAMATH_CALUDE_line_through_intersection_l964_96419

/-- The line l: ax - y + b = 0 passes through the intersection point of 
    lines l₁: 2x - 2y - 3 = 0 and l₂: 3x - 5y + 1 = 0 
    if and only if 17a + 4b = 11 -/
theorem line_through_intersection (a b : ℝ) : 
  (∃ x y : ℝ, 2*x - 2*y - 3 = 0 ∧ 3*x - 5*y + 1 = 0 ∧ a*x - y + b = 0) ↔ 
  17*a + 4*b = 11 := by
sorry

end NUMINAMATH_CALUDE_line_through_intersection_l964_96419


namespace NUMINAMATH_CALUDE_train_length_l964_96428

/-- The length of a train given its crossing times over a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ) 
  (h1 : bridge_length = 1500)
  (h2 : bridge_time = 70)
  (h3 : lamp_time = 20) :
  ∃ (train_length : ℝ), 
    train_length / lamp_time = (train_length + bridge_length) / bridge_time ∧ 
    train_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l964_96428


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l964_96443

/-- Given a quadratic equation x^2 + 8x - 1 = 0, when written in the form (x + a)^2 = b, b equals 17 -/
theorem complete_square_quadratic : 
  ∃ a : ℝ, ∀ x : ℝ, (x^2 + 8*x - 1 = 0) ↔ ((x + a)^2 = 17) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l964_96443


namespace NUMINAMATH_CALUDE_sum_inequality_l964_96477

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l964_96477


namespace NUMINAMATH_CALUDE_mangoes_rate_per_kg_l964_96442

/-- Given Tom's fruit purchase details, prove the rate per kg for mangoes -/
theorem mangoes_rate_per_kg 
  (apple_quantity : ℕ) 
  (apple_rate : ℕ) 
  (mango_quantity : ℕ) 
  (total_paid : ℕ) 
  (h1 : apple_quantity = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_quantity = 9)
  (h4 : total_paid = 965) :
  (total_paid - apple_quantity * apple_rate) / mango_quantity = 45 :=
by sorry

end NUMINAMATH_CALUDE_mangoes_rate_per_kg_l964_96442


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l964_96496

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l964_96496


namespace NUMINAMATH_CALUDE_log_ratio_evaluation_l964_96492

theorem log_ratio_evaluation : (Real.log 4 / Real.log 3) / (Real.log 8 / Real.log 9) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_evaluation_l964_96492


namespace NUMINAMATH_CALUDE_geometric_sequence_log_sum_l964_96438

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_log_sum
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 10 * a 11 = Real.exp 5) :
  (Finset.range 20).sum (λ i => Real.log (a (i + 1))) = 50 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_log_sum_l964_96438


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l964_96487

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l964_96487


namespace NUMINAMATH_CALUDE_max_clocks_in_workshop_l964_96435

/-- Represents a digital clock with hours and minutes -/
structure DigitalClock where
  hours : Nat
  minutes : Nat

/-- Represents the state of all clocks in the workshop -/
structure ClockWorkshop where
  clocks : List DigitalClock

/-- Checks if all clocks in the workshop show different times -/
def allDifferentTimes (workshop : ClockWorkshop) : Prop :=
  ∀ c1 c2 : DigitalClock, c1 ∈ workshop.clocks → c2 ∈ workshop.clocks → c1 ≠ c2 →
    (c1.hours ≠ c2.hours ∨ c1.minutes ≠ c2.minutes)

/-- Calculates the sum of hours displayed on all clocks -/
def sumHours (workshop : ClockWorkshop) : Nat :=
  workshop.clocks.foldl (fun sum clock => sum + clock.hours) 0

/-- Calculates the sum of minutes displayed on all clocks -/
def sumMinutes (workshop : ClockWorkshop) : Nat :=
  workshop.clocks.foldl (fun sum clock => sum + clock.minutes) 0

/-- Represents the state of the workshop after some time has passed -/
def advanceTime (workshop : ClockWorkshop) : ClockWorkshop := sorry

theorem max_clocks_in_workshop :
  ∀ (workshop : ClockWorkshop),
    workshop.clocks.length > 1 →
    (∀ clock ∈ workshop.clocks, clock.hours ≥ 1 ∧ clock.hours ≤ 12) →
    (∀ clock ∈ workshop.clocks, clock.minutes ≥ 0 ∧ clock.minutes < 60) →
    allDifferentTimes workshop →
    sumHours (advanceTime workshop) + 1 = sumHours workshop →
    sumMinutes (advanceTime workshop) + 1 = sumMinutes workshop →
    workshop.clocks.length ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_clocks_in_workshop_l964_96435


namespace NUMINAMATH_CALUDE_linear_equation_not_proportional_l964_96450

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0

/-- Direct proportionality between x and y -/
def DirectlyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, y t = k * x t

/-- Inverse proportionality between x and y -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, x t * y t = k

/-- 
For a linear equation ax + by = c, where a ≠ 0 and b ≠ 0,
y is neither directly nor inversely proportional to x
-/
theorem linear_equation_not_proportional (eq : LinearEquation) :
  let x : ℝ → ℝ := λ t => t
  let y : ℝ → ℝ := λ t => (eq.c - eq.a * t) / eq.b
  ¬(DirectlyProportional x y ∨ InverselyProportional x y) := by
  sorry


end NUMINAMATH_CALUDE_linear_equation_not_proportional_l964_96450


namespace NUMINAMATH_CALUDE_angle_of_inclination_cosine_l964_96465

theorem angle_of_inclination_cosine (θ : Real) :
  (∃ (m : Real), m = 2 ∧ θ = Real.arctan m) →
  Real.cos θ = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_cosine_l964_96465


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l964_96433

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 220 * p - 7 = 0) →
  (3 * q^3 - 4 * q^2 + 220 * q - 7 = 0) →
  (3 * r^3 - 4 * r^2 + 220 * r - 7 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l964_96433


namespace NUMINAMATH_CALUDE_class_size_l964_96452

/-- Given a class of children where:
  * 19 play tennis
  * 21 play squash
  * 10 play neither sport
  * 12 play both sports
  This theorem proves that there are 38 children in the class. -/
theorem class_size (T S N B : ℕ) (h1 : T = 19) (h2 : S = 21) (h3 : N = 10) (h4 : B = 12) :
  T + S - B + N = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l964_96452


namespace NUMINAMATH_CALUDE_tom_bikes_11860_miles_l964_96497

/-- The number of miles Tom bikes in a year -/
def total_miles : ℕ :=
  let miles_per_day_first_period : ℕ := 30
  let days_first_period : ℕ := 183
  let miles_per_day_second_period : ℕ := 35
  let days_in_year : ℕ := 365
  let days_second_period : ℕ := days_in_year - days_first_period
  miles_per_day_first_period * days_first_period + miles_per_day_second_period * days_second_period

/-- Theorem stating that Tom bikes 11860 miles in a year -/
theorem tom_bikes_11860_miles : total_miles = 11860 := by
  sorry

end NUMINAMATH_CALUDE_tom_bikes_11860_miles_l964_96497


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l964_96464

theorem quadratic_equation_solutions (a : ℝ) : a^2 + 10 = a + 10^2 ↔ a = 10 ∨ a = -9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l964_96464


namespace NUMINAMATH_CALUDE_right_triangle_area_l964_96400

/-- The area of a right triangle with hypotenuse 5√2 and one leg 5 is 12.5 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) 
  (h2 : c = 5 * Real.sqrt 2) (h3 : a = 5) : (1/2) * a * b = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l964_96400


namespace NUMINAMATH_CALUDE_sequence_equation_proof_l964_96475

/-- Given a sequence of equations, prove the value of (b+1)/a^2 -/
theorem sequence_equation_proof (a b : ℕ) (h : ∀ (n : ℕ), 32 ≤ n → n ≤ 32016 → 
  ∃ (m : ℕ), n + m / n = (n - 32 + 3) * (3 + m / n)) 
  (h_last : 32016 + a / b = 2016 * 3 * (a / b)) : 
  (b + 1) / (a^2 : ℚ) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equation_proof_l964_96475


namespace NUMINAMATH_CALUDE_mrs_hilt_travel_l964_96489

/-- Calculates the total miles traveled given the number of books read and miles per book -/
def total_miles (books_read : ℕ) (miles_per_book : ℕ) : ℕ :=
  books_read * miles_per_book

/-- Proves that Mrs. Hilt traveled 6750 miles to Japan -/
theorem mrs_hilt_travel : total_miles 15 450 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_travel_l964_96489


namespace NUMINAMATH_CALUDE_hcf_problem_l964_96405

theorem hcf_problem (a b : ℕ) (h1 : a = 391) (h2 : ∃ (hcf : ℕ), Nat.lcm a b = hcf * 13 * 17) : Nat.gcd a b = 23 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l964_96405


namespace NUMINAMATH_CALUDE_playground_route_combinations_l964_96482

theorem playground_route_combinations : 
  ∀ (n : ℕ) (k : ℕ), n = 2 ∧ k = 3 → n ^ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_playground_route_combinations_l964_96482


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l964_96406

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 30) → (n * exterior_angle = 360) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l964_96406


namespace NUMINAMATH_CALUDE_angle_measure_in_regular_octagon_l964_96481

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Measure of an angle in radians -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_in_regular_octagon 
  (ABCDEFGH : RegularOctagon) 
  (A E C : ℝ × ℝ) 
  (hA : A = ABCDEFGH.vertices 0)
  (hE : E = ABCDEFGH.vertices 4)
  (hC : C = ABCDEFGH.vertices 2) :
  angle_measure A E C = 112.5 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_regular_octagon_l964_96481


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l964_96418

/-- Calculates the sampling interval for systematic sampling -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for a population of 2000 and sample size of 50 is 40 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 2000 50 = 40 := by
  sorry

#eval systematicSamplingInterval 2000 50

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l964_96418


namespace NUMINAMATH_CALUDE_average_weight_increase_l964_96446

/-- Proves that the increase in average weight is 2.5 kg when a person weighing 65 kg
    in a group of 6 is replaced by a person weighing 80 kg. -/
theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 6 →
  old_weight = 65 →
  new_weight = 80 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l964_96446


namespace NUMINAMATH_CALUDE_matthew_friends_count_l964_96403

def total_crackers : ℝ := 36
def crackers_per_friend : ℝ := 6.5

theorem matthew_friends_count :
  ⌊total_crackers / crackers_per_friend⌋ = 5 :=
by sorry

end NUMINAMATH_CALUDE_matthew_friends_count_l964_96403


namespace NUMINAMATH_CALUDE_ellipse_equation_l964_96444

/-- 
Given an ellipse with center at the origin, one focus at (0, √50), 
and a chord intersecting the line y = 3x - 2 with midpoint x-coordinate 1/2, 
prove that the standard equation of the ellipse is x²/25 + y²/75 = 1.
-/
theorem ellipse_equation (F : ℝ × ℝ) (midpoint_x : ℝ) : 
  F = (0, Real.sqrt 50) →
  midpoint_x = 1/2 →
  ∃ (x y : ℝ), x^2/25 + y^2/75 = 1 ∧
    ∃ (x1 y1 x2 y2 : ℝ), 
      (x1^2/25 + y1^2/75 = 1) ∧
      (x2^2/25 + y2^2/75 = 1) ∧
      (y1 = 3*x1 - 2) ∧
      (y2 = 3*x2 - 2) ∧
      ((x1 + x2)/2 = midpoint_x) ∧
      ((y1 + y2)/2 = 3*midpoint_x - 2) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_equation_l964_96444


namespace NUMINAMATH_CALUDE_remaining_distance_is_546_point_5_l964_96480

-- Define the total distance
def total_distance : ℝ := 1045

-- Define Amoli's driving
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3

-- Define Anayet's driving
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2.5

-- Define Bimal's driving
def bimal_speed : ℝ := 55
def bimal_time : ℝ := 4

-- Theorem statement
theorem remaining_distance_is_546_point_5 :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time + bimal_speed * bimal_time) = 546.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_is_546_point_5_l964_96480


namespace NUMINAMATH_CALUDE_cafe_pricing_l964_96461

theorem cafe_pricing (s c p : ℝ) 
  (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
  (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
  (eq3 : 4 * s + 8 * c + p = 5.20) :
  s + c + p = 1.30 := by
  sorry

end NUMINAMATH_CALUDE_cafe_pricing_l964_96461


namespace NUMINAMATH_CALUDE_max_value_of_sine_function_l964_96499

theorem max_value_of_sine_function (x : ℝ) (h : -π/2 ≤ x ∧ x ≤ 0) :
  ∃ (y : ℝ), y = 3 * Real.sin x + 2 ∧ y ≤ 2 ∧ ∃ (x₀ : ℝ), -π/2 ≤ x₀ ∧ x₀ ≤ 0 ∧ 3 * Real.sin x₀ + 2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sine_function_l964_96499


namespace NUMINAMATH_CALUDE_valerie_laptop_savings_l964_96469

/-- Proves that Valerie needs 30 weeks to save for a laptop -/
theorem valerie_laptop_savings :
  let laptop_price : ℕ := 800
  let parents_money : ℕ := 100
  let uncle_money : ℕ := 60
  let siblings_money : ℕ := 40
  let weekly_tutoring_income : ℕ := 20
  let total_graduation_money : ℕ := parents_money + uncle_money + siblings_money
  let remaining_amount : ℕ := laptop_price - total_graduation_money
  let weeks_needed : ℕ := remaining_amount / weekly_tutoring_income
  weeks_needed = 30 := by
sorry


end NUMINAMATH_CALUDE_valerie_laptop_savings_l964_96469


namespace NUMINAMATH_CALUDE_cubic_floor_equation_solution_l964_96445

theorem cubic_floor_equation_solution :
  ∃! x : ℝ, 3 * x^3 - ⌊x⌋ = 3 :=
by
  -- The unique solution is x = ∛(4/3)
  use Real.rpow (4/3) (1/3)
  sorry

end NUMINAMATH_CALUDE_cubic_floor_equation_solution_l964_96445


namespace NUMINAMATH_CALUDE_min_value_expression_l964_96457

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 5/4)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l964_96457


namespace NUMINAMATH_CALUDE_distance_between_last_two_points_l964_96459

def cube_vertices : List (Fin 3 → ℝ) := [
  ![0, 0, 0], ![0, 0, 6], ![0, 6, 0], ![0, 6, 6],
  ![6, 0, 0], ![6, 0, 6], ![6, 6, 0], ![6, 6, 6]
]

def plane_intersections : List (Fin 3 → ℝ) := [
  ![0, 3, 0], ![2, 0, 0], ![2, 6, 6], ![4, 0, 6], ![0, 6, 6]
]

theorem distance_between_last_two_points :
  let S := plane_intersections[3]
  let T := plane_intersections[4]
  Real.sqrt ((S 0 - T 0)^2 + (S 1 - T 1)^2 + (S 2 - T 2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_last_two_points_l964_96459


namespace NUMINAMATH_CALUDE_profit_decrease_l964_96451

theorem profit_decrease (march_profit : ℝ) (h1 : march_profit > 0) : 
  let april_profit := march_profit * 1.4
  let june_profit := march_profit * 1.68
  ∃ (may_profit : ℝ), 
    may_profit = april_profit * 0.8 ∧ 
    june_profit = may_profit * 1.5 := by
  sorry

end NUMINAMATH_CALUDE_profit_decrease_l964_96451


namespace NUMINAMATH_CALUDE_negation_equivalence_l964_96420

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x < x ^ 2) ↔ (∀ x : ℝ, (2 : ℝ) ^ x ≥ x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l964_96420


namespace NUMINAMATH_CALUDE_prime_fourth_powers_sum_l964_96413

theorem prime_fourth_powers_sum (p q r s : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s →
  p ≤ q ∧ q ≤ r →
  p^4 + q^4 + r^4 + 119 = s^2 →
  p = 2 ∧ q = 3 ∧ r = 5 ∧ s = 29 := by
  sorry

end NUMINAMATH_CALUDE_prime_fourth_powers_sum_l964_96413


namespace NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l964_96409

theorem cos_squared_pi_fourth_minus_alpha (α : Real) 
  (h : Real.tan (α + π/4) = 3/4) : 
  Real.cos (π/4 - α)^2 = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l964_96409


namespace NUMINAMATH_CALUDE_min_problems_solved_l964_96404

/-- Represents the possible point values for each problem -/
inductive PointValue
  | three : PointValue
  | eight : PointValue
  | ten : PointValue

/-- Calculates the point value of a given PointValue -/
def pointValueToInt (pv : PointValue) : Nat :=
  match pv with
  | PointValue.three => 3
  | PointValue.eight => 8
  | PointValue.ten => 10

/-- Represents a solution to the problem -/
structure Solution :=
  (problems : List PointValue)

/-- Calculates the total score of a solution -/
def totalScore (sol : Solution) : Nat :=
  sol.problems.map pointValueToInt |>.sum

/-- Checks if a solution is valid (i.e., has a total score of 45) -/
def isValidSolution (sol : Solution) : Prop :=
  totalScore sol = 45

/-- The main theorem to prove -/
theorem min_problems_solved :
  ∀ (sol : Solution), isValidSolution sol → sol.problems.length ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_problems_solved_l964_96404


namespace NUMINAMATH_CALUDE_max_digits_product_4digit_3digit_l964_96455

theorem max_digits_product_4digit_3digit : 
  ∀ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 → 
    100 ≤ b ∧ b < 1000 → 
    a * b < 10000000 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_4digit_3digit_l964_96455


namespace NUMINAMATH_CALUDE_no_solution_condition_l964_96440

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 5 * |x - 4*a| + |x - a^2| + 4*x - 3*a ≠ 0) ↔ (a < -9 ∨ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l964_96440


namespace NUMINAMATH_CALUDE_club_officer_selection_count_l964_96466

/-- Represents the number of ways to choose club officers under specific conditions -/
def choose_club_officers (total_members boys girls : ℕ) : ℕ :=
  2 * boys * girls * (boys - 1)

/-- Theorem stating the number of ways to choose club officers -/
theorem club_officer_selection_count :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  (total_members = boys + girls) →
  (choose_club_officers total_members boys girls = 6300) := by
  sorry

#check club_officer_selection_count

end NUMINAMATH_CALUDE_club_officer_selection_count_l964_96466


namespace NUMINAMATH_CALUDE_counterexample_21_l964_96449

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem counterexample_21 :
  ¬(is_prime 21) ∧ ¬(is_prime (21 + 3)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_21_l964_96449


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l964_96432

theorem inscribed_square_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b * c) / (a^2 + b^2)
  s = 525 / 96 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l964_96432


namespace NUMINAMATH_CALUDE_system_solution_l964_96458

theorem system_solution : ∃ (s t : ℝ), 
  (7 * s + 3 * t = 102) ∧ 
  (s = (t - 3)^2) ∧ 
  (abs (t - 6.44) < 0.01) ∧ 
  (abs (s - 11.83) < 0.01) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l964_96458


namespace NUMINAMATH_CALUDE_unique_solution_l964_96491

/-- The set of solutions for the system of equations x + y = 2 and x - y = 0 -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 2 ∧ p.1 - p.2 = 0}

/-- Theorem stating that the solution set contains only the point (1,1) -/
theorem unique_solution :
  solution_set = {(1, 1)} := by sorry

end NUMINAMATH_CALUDE_unique_solution_l964_96491


namespace NUMINAMATH_CALUDE_profit_with_discount_theorem_l964_96421

/-- Calculates the profit percentage with discount given the discount rate and profit percentage without discount -/
def profit_percentage_with_discount (discount_rate : ℝ) (profit_no_discount : ℝ) : ℝ :=
  ((1 - discount_rate) * (1 + profit_no_discount) - 1) * 100

/-- Theorem stating that given a 5% discount and 28% profit without discount, the profit percentage with discount is 21.6% -/
theorem profit_with_discount_theorem :
  profit_percentage_with_discount 0.05 0.28 = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_profit_with_discount_theorem_l964_96421


namespace NUMINAMATH_CALUDE_z_pure_imaginary_iff_m_eq_2013_l964_96426

/-- A complex number z is pure imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m - 2013) (m - 1)

/-- Theorem stating that z is pure imaginary if and only if m = 2013. -/
theorem z_pure_imaginary_iff_m_eq_2013 :
    ∀ m : ℝ, is_pure_imaginary (z m) ↔ m = 2013 := by
  sorry

end NUMINAMATH_CALUDE_z_pure_imaginary_iff_m_eq_2013_l964_96426


namespace NUMINAMATH_CALUDE_pavan_travel_distance_l964_96415

theorem pavan_travel_distance :
  ∀ (total_distance : ℝ),
  (total_distance / 2 / 30 + total_distance / 2 / 25 = 11) →
  total_distance = 150 := by
sorry

end NUMINAMATH_CALUDE_pavan_travel_distance_l964_96415
