import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1958_195865

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℚ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_first : a 1 = 7/9)
  (h_thirteenth : a 13 = 4/5) :
  a 7 = 71/90 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1958_195865


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1958_195896

/-- The maximum marks of an exam, given the conditions from the problem -/
def maximum_marks : ℕ :=
  let required_percentage : ℚ := 80 / 100
  let marks_obtained : ℕ := 200
  let marks_short : ℕ := 200
  500

/-- Theorem stating that the maximum marks of the exam is 500 -/
theorem exam_maximum_marks :
  let required_percentage : ℚ := 80 / 100
  let marks_obtained : ℕ := 200
  let marks_short : ℕ := 200
  maximum_marks = 500 := by
  sorry

#check exam_maximum_marks

end NUMINAMATH_CALUDE_exam_maximum_marks_l1958_195896


namespace NUMINAMATH_CALUDE_train_B_speed_l1958_195812

-- Define the problem parameters
def distance_between_cities : ℝ := 330
def speed_train_A : ℝ := 60
def time_train_A : ℝ := 3
def time_train_B : ℝ := 2

-- Theorem statement
theorem train_B_speed : 
  ∃ (speed_train_B : ℝ),
    speed_train_B * time_train_B + speed_train_A * time_train_A = distance_between_cities ∧
    speed_train_B = 75 := by
  sorry

end NUMINAMATH_CALUDE_train_B_speed_l1958_195812


namespace NUMINAMATH_CALUDE_equation_solution_l1958_195848

theorem equation_solution : 
  ∃! x : ℚ, (x - 27) / 3 = (3 * x + 6) / 8 ∧ x = -234 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1958_195848


namespace NUMINAMATH_CALUDE_infinite_dividing_planes_l1958_195803

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields here

/-- A plane that intersects a regular tetrahedron -/
structure IntersectingPlane where
  -- Add necessary fields here

/-- Predicate to check if a plane divides a tetrahedron into two equal parts -/
def divides_equally (t : RegularTetrahedron) (p : IntersectingPlane) : Prop :=
  sorry

/-- The set of planes that divide a regular tetrahedron into two equal parts -/
def dividing_planes (t : RegularTetrahedron) : Set IntersectingPlane :=
  {p : IntersectingPlane | divides_equally t p}

/-- Theorem stating that there are infinitely many planes that divide a regular tetrahedron equally -/
theorem infinite_dividing_planes (t : RegularTetrahedron) :
  Set.Infinite (dividing_planes t) :=
sorry

end NUMINAMATH_CALUDE_infinite_dividing_planes_l1958_195803


namespace NUMINAMATH_CALUDE_min_value_of_E_l1958_195820

/-- Given that the minimum value of |x - 4| + |E| + |x - 5| is 11,
    prove that the minimum value of |E| is 10. -/
theorem min_value_of_E (E : ℝ) :
  (∃ (c : ℝ), ∀ (x : ℝ), c ≤ |x - 4| + |E| + |x - 5| ∧ 
   ∃ (x : ℝ), c = |x - 4| + |E| + |x - 5|) →
  (c = 11) →
  (∃ (d : ℝ), ∀ (y : ℝ), d ≤ |y| ∧ 
   ∃ (y : ℝ), d = |y|) →
  (d = 10) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_E_l1958_195820


namespace NUMINAMATH_CALUDE_quadratic_function_equivalence_l1958_195849

theorem quadratic_function_equivalence :
  ∀ x : ℝ, 2 * x^2 - 8 * x - 1 = 2 * (x - 2)^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_equivalence_l1958_195849


namespace NUMINAMATH_CALUDE_Q_proper_subset_of_P_l1958_195806

def P : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def Q : Set ℕ := {2, 3, 5, 6}

theorem Q_proper_subset_of_P : Q ⊂ P := by
  sorry

end NUMINAMATH_CALUDE_Q_proper_subset_of_P_l1958_195806


namespace NUMINAMATH_CALUDE_cookies_baked_l1958_195816

/-- The number of cookies Maria made -/
def total_cookies : ℕ := 144

/-- The fraction of cookies with nuts -/
def fraction_with_nuts : ℚ := 1/4

/-- The number of nuts used per cookie with nuts -/
def nuts_per_cookie : ℕ := 2

/-- The total number of nuts used -/
def total_nuts_used : ℕ := 72

theorem cookies_baked :
  (total_cookies : ℚ) * fraction_with_nuts * nuts_per_cookie = total_nuts_used := by
  sorry

end NUMINAMATH_CALUDE_cookies_baked_l1958_195816


namespace NUMINAMATH_CALUDE_dodecahedron_diagonals_l1958_195831

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Finset (Fin 5))
  vertex_face_incidence : vertices → Finset faces
  diag : Fin 20 → Fin 20 → Prop

/-- Properties of a dodecahedron -/
axiom dodecahedron_properties (D : Dodecahedron) :
  (D.vertices.card = 20) ∧
  (D.faces.card = 12) ∧
  (∀ v : D.vertices, (D.vertex_face_incidence v).card = 3) ∧
  (∀ v w : D.vertices, D.diag v w ↔ v ≠ w ∧ (D.vertex_face_incidence v ∩ D.vertex_face_incidence w).card = 0)

/-- The number of interior diagonals in a dodecahedron -/
def interior_diagonals (D : Dodecahedron) : ℕ :=
  (D.vertices.card * (D.vertices.card - 4)) / 2

/-- Theorem: A dodecahedron has 160 interior diagonals -/
theorem dodecahedron_diagonals (D : Dodecahedron) : interior_diagonals D = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_diagonals_l1958_195831


namespace NUMINAMATH_CALUDE_range_of_f_l1958_195870

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the domain
def domain : Set ℝ := Set.Icc 1 5

-- Theorem statement
theorem range_of_f :
  Set.range (fun x => f x) ∩ (Set.image f domain) = Set.Ico 2 11 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1958_195870


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l1958_195897

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l1958_195897


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_46_l1958_195860

theorem units_digit_of_27_times_46 : (27 * 46) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_46_l1958_195860


namespace NUMINAMATH_CALUDE_cleaning_time_with_doubled_speed_l1958_195855

-- Define the cleaning rates
def anne_rate : ℚ := 1 / 12
def bruce_rate : ℚ := 1 / 4 - anne_rate

-- Define the time it takes for both to clean at normal speed
def normal_time : ℚ := 4

-- Define Anne's doubled rate
def anne_doubled_rate : ℚ := 2 * anne_rate

-- Theorem statement
theorem cleaning_time_with_doubled_speed :
  (bruce_rate + anne_doubled_rate)⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_time_with_doubled_speed_l1958_195855


namespace NUMINAMATH_CALUDE_factorization_equality_l1958_195833

theorem factorization_equality (x : ℝ) : 
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1958_195833


namespace NUMINAMATH_CALUDE_angle_q_measure_l1958_195800

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  -- Angle measures in degrees
  angle_p : ℝ
  angle_q : ℝ
  angle_r : ℝ
  -- Triangle conditions
  sum_of_angles : angle_p + angle_q + angle_r = 180
  isosceles : angle_q = angle_r
  angle_r_five_times_p : angle_r = 5 * angle_p

/-- The measure of angle Q in the specified isosceles triangle is 900/11 degrees -/
theorem angle_q_measure (t : IsoscelesTriangle) : t.angle_q = 900 / 11 := by
  sorry

#check angle_q_measure

end NUMINAMATH_CALUDE_angle_q_measure_l1958_195800


namespace NUMINAMATH_CALUDE_unique_base_for_256_four_digits_l1958_195830

/-- A number n has exactly d digits in base b if and only if b^(d-1) ≤ n < b^d -/
def has_exactly_d_digits (n : ℕ) (b : ℕ) (d : ℕ) : Prop :=
  b^(d-1) ≤ n ∧ n < b^d

/-- The theorem statement -/
theorem unique_base_for_256_four_digits :
  ∃! b : ℕ, b ≥ 2 ∧ has_exactly_d_digits 256 b 4 :=
sorry

end NUMINAMATH_CALUDE_unique_base_for_256_four_digits_l1958_195830


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l1958_195859

theorem smallest_prime_dividing_sum : ∃ p : Nat, 
  Prime p ∧ p > 7 ∧ p ∣ (2^14 + 7^8) ∧ 
  ∀ q : Nat, Prime q → q ∣ (2^14 + 7^8) → q ≥ p :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l1958_195859


namespace NUMINAMATH_CALUDE_jerry_fireworks_l1958_195862

theorem jerry_fireworks (firecrackers sparklers : ℕ) 
  (h1 : firecrackers = 48)
  (h2 : sparklers = 30)
  (confiscated_firecrackers : ℕ := firecrackers / 4)
  (confiscated_sparklers : ℕ := sparklers / 10)
  (remaining_firecrackers : ℕ := firecrackers - confiscated_firecrackers)
  (remaining_sparklers : ℕ := sparklers - confiscated_sparklers)
  (defective_firecrackers : ℕ := remaining_firecrackers / 6)
  (defective_sparklers : ℕ := remaining_sparklers / 4)
  (good_firecrackers : ℕ := remaining_firecrackers - defective_firecrackers)
  (good_sparklers : ℕ := remaining_sparklers - defective_sparklers)
  (set_off_firecrackers : ℕ := good_firecrackers / 2)
  (set_off_sparklers : ℕ := good_sparklers * 2 / 3) :
  set_off_firecrackers + set_off_sparklers = 29 :=
by sorry

end NUMINAMATH_CALUDE_jerry_fireworks_l1958_195862


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_half_l1958_195881

/-- Given an ellipse and a hyperbola with shared foci, prove the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity_half 
  (a b m n c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hab : a > b)
  (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hyperbola_eq : ∀ (x y : ℝ), x^2 / m^2 - y^2 / n^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 / m^2 - p.2^2 / n^2 = 1})
  (foci : c > 0 ∧ {(-c, 0), (c, 0)} ⊆ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} ∩ {p : ℝ × ℝ | p.1^2 / m^2 - p.2^2 / n^2 = 1})
  (geom_mean : c^2 = a * m)
  (arith_mean : n^2 = m^2 + c^2 / 2) :
  c / a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_half_l1958_195881


namespace NUMINAMATH_CALUDE_percentage_of_burpees_l1958_195877

/-- Calculate the percentage of burpees in Emmett's workout routine -/
theorem percentage_of_burpees (jumping_jacks pushups situps burpees lunges : ℕ) :
  jumping_jacks = 25 →
  pushups = 15 →
  situps = 30 →
  burpees = 10 →
  lunges = 20 →
  (burpees : ℚ) / (jumping_jacks + pushups + situps + burpees + lunges) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_burpees_l1958_195877


namespace NUMINAMATH_CALUDE_fashion_show_duration_l1958_195808

/-- The total time for a fashion show runway -/
def fashion_show_time (num_models : ℕ) (bathing_suits_per_model : ℕ) (evening_wear_per_model : ℕ) (time_per_trip : ℕ) : ℕ :=
  (num_models * (bathing_suits_per_model + evening_wear_per_model)) * time_per_trip

/-- Theorem: The fashion show with 6 models, 2 bathing suits and 3 evening wear per model, and 2 minutes per trip takes 60 minutes -/
theorem fashion_show_duration :
  fashion_show_time 6 2 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fashion_show_duration_l1958_195808


namespace NUMINAMATH_CALUDE_unique_solution_l1958_195829

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (2 * x - 2 * y) + x = f (3 * x) - f (2 * y) + k * y

/-- The theorem stating the unique solution to the functional equation -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, ∃ k : ℝ, SatisfiesEquation f k ∧ f = id ∧ k = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1958_195829


namespace NUMINAMATH_CALUDE_count_is_nine_l1958_195837

/-- A function that returns the count of valid 4-digit numbers greater than 1000 
    that can be formed using the digits of 2012 -/
def count_valid_numbers : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating that the count of valid numbers is 9 -/
theorem count_is_nine : count_valid_numbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_is_nine_l1958_195837


namespace NUMINAMATH_CALUDE_sticker_cost_l1958_195835

theorem sticker_cost (num_packs : ℕ) (stickers_per_pack : ℕ) (james_payment : ℚ) :
  num_packs = 4 →
  stickers_per_pack = 30 →
  james_payment = 6 →
  (2 * james_payment) / (num_packs * stickers_per_pack : ℚ) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_sticker_cost_l1958_195835


namespace NUMINAMATH_CALUDE_streamer_profit_formula_l1958_195853

/-- Streamer's daily profit function -/
def daily_profit (x : ℝ) : ℝ :=
  (x - 50) * (300 + 3 * (99 - x))

/-- Initial selling price -/
def initial_price : ℝ := 99

/-- Initial daily sales volume -/
def initial_sales : ℝ := 300

/-- Sales volume increase per yuan price decrease -/
def sales_increase_rate : ℝ := 3

/-- Cost and expenses per item -/
def cost_per_item : ℝ := 50

theorem streamer_profit_formula (x : ℝ) :
  daily_profit x = (x - cost_per_item) * (initial_sales + sales_increase_rate * (initial_price - x)) :=
by sorry

end NUMINAMATH_CALUDE_streamer_profit_formula_l1958_195853


namespace NUMINAMATH_CALUDE_derivative_symmetry_l1958_195890

/-- Given a function f(x) = ax^4 + bx^2 + c, if f'(1) = 2, then f'(-1) = -2 -/
theorem derivative_symmetry (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^4 + b * x^2 + c
  let f' := fun (x : ℝ) => 4 * a * x^3 + 2 * b * x
  f' 1 = 2 → f' (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l1958_195890


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l1958_195861

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l1958_195861


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l1958_195847

/-- Proves that given a boat with speed 18 kmph in still water and a stream with speed 6 kmph,
    if the boat can cover 48 km downstream or a certain distance upstream in the same time,
    then the distance the boat can cover upstream is 24 km. -/
theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) :
  boat_speed = 18 →
  stream_speed = 6 →
  downstream_distance = 48 →
  (downstream_distance / (boat_speed + stream_speed) = 
   (boat_speed - stream_speed) * (downstream_distance / (boat_speed + stream_speed))) →
  (boat_speed - stream_speed) * (downstream_distance / (boat_speed + stream_speed)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l1958_195847


namespace NUMINAMATH_CALUDE_largest_c_value_l1958_195868

theorem largest_c_value (c d e : ℤ) 
  (eq : 5 * c + (d - 12)^2 + e^3 = 235)
  (c_lt_d : c < d) : 
  c ≤ 22 ∧ ∃ (c' d' e' : ℤ), c' = 22 ∧ c' < d' ∧ 5 * c' + (d' - 12)^2 + e'^3 = 235 :=
sorry

end NUMINAMATH_CALUDE_largest_c_value_l1958_195868


namespace NUMINAMATH_CALUDE_parabola_vertex_l1958_195811

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 2 is at the point (1, 2) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1958_195811


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l1958_195822

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x + 1

def xValues : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_composite_function :
  (xValues.map (λ x => q (p x))).sum = -13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l1958_195822


namespace NUMINAMATH_CALUDE_max_retained_pits_problem_l1958_195879

/-- Calculates the maximum number of pits that can be retained on a road -/
def max_retained_pits (road_length : ℕ) (initial_spacing : ℕ) (revised_spacing : ℕ) : ℕ :=
  let interval := Nat.lcm initial_spacing revised_spacing
  let num_intervals := road_length / interval
  2 * (num_intervals + 1)

/-- Theorem stating the maximum number of retained pits for the given problem -/
theorem max_retained_pits_problem :
  max_retained_pits 120 3 5 = 18 := by
  sorry

#eval max_retained_pits 120 3 5

end NUMINAMATH_CALUDE_max_retained_pits_problem_l1958_195879


namespace NUMINAMATH_CALUDE_circle_condition_l1958_195842

/-- The equation of a potential circle with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 5*m = 0

/-- Theorem stating the necessary and sufficient condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) :
  (∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ), circle_equation x y m ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔ m < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1958_195842


namespace NUMINAMATH_CALUDE_max_value_xy_over_z_l1958_195814

theorem max_value_xy_over_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 4 * x^2 - 3 * x * y + y^2 - z = 0) :
  ∃ (M : ℝ), M = 1 ∧ ∀ (w : ℝ), w = x * y / z → w ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_xy_over_z_l1958_195814


namespace NUMINAMATH_CALUDE_incomplete_factor_multiple_statement_l1958_195844

theorem incomplete_factor_multiple_statement : ¬(56 / 7 = 8 → (∃n : ℕ, 56 = n * 7) ∧ (∃m : ℕ, 7 * m = 56)) := by
  sorry

end NUMINAMATH_CALUDE_incomplete_factor_multiple_statement_l1958_195844


namespace NUMINAMATH_CALUDE_decimal_place_150_of_5_over_8_l1958_195823

theorem decimal_place_150_of_5_over_8 : 
  let decimal_expansion := (5 : ℚ) / 8
  let digit_at_n (q : ℚ) (n : ℕ) := (q * 10^n).floor % 10
  digit_at_n decimal_expansion 150 = 0 := by
  sorry

end NUMINAMATH_CALUDE_decimal_place_150_of_5_over_8_l1958_195823


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l1958_195832

theorem x_plus_y_equals_negative_one 
  (x y : ℝ) 
  (h1 : x + |x| + y = 5) 
  (h2 : x + |y| - y = 6) : 
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l1958_195832


namespace NUMINAMATH_CALUDE_sum_of_x_y_z_l1958_195878

theorem sum_of_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_y_z_l1958_195878


namespace NUMINAMATH_CALUDE_first_divisible_correct_l1958_195856

/-- The first 4-digit number divisible by 25, 40, and 75 -/
def first_divisible : ℕ := 1200

/-- The greatest 4-digit number divisible by 25, 40, and 75 -/
def greatest_divisible : ℕ := 9600

/-- Theorem stating that first_divisible is the first 4-digit number divisible by 25, 40, and 75 -/
theorem first_divisible_correct :
  (first_divisible ≥ 1000) ∧
  (first_divisible ≤ 9999) ∧
  (first_divisible % 25 = 0) ∧
  (first_divisible % 40 = 0) ∧
  (first_divisible % 75 = 0) ∧
  (∀ n : ℕ, 1000 ≤ n ∧ n < first_divisible →
    ¬(n % 25 = 0 ∧ n % 40 = 0 ∧ n % 75 = 0)) ∧
  (greatest_divisible = 9600) ∧
  (greatest_divisible % 25 = 0) ∧
  (greatest_divisible % 40 = 0) ∧
  (greatest_divisible % 75 = 0) ∧
  (∀ m : ℕ, m > greatest_divisible → m > 9999 ∨ ¬(m % 25 = 0 ∧ m % 40 = 0 ∧ m % 75 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_first_divisible_correct_l1958_195856


namespace NUMINAMATH_CALUDE_parabola_vertex_l1958_195834

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 6*y + 4*x - 7 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ x' y', parabola_equation x' y' → y' ≥ y

theorem parabola_vertex :
  is_vertex 4 (-3) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1958_195834


namespace NUMINAMATH_CALUDE_lines_intersect_at_one_point_l1958_195892

-- Define the basic geometric objects
variable (A B C D E F P Q M O : Point)

-- Define the convex quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the extension relationships
def lies_on_extension (P X Y Z : Point) : Prop := sorry

-- Define the midpoint relationship
def is_midpoint (M X Y : Point) : Prop := sorry

-- Define when a point lies on a line
def point_on_line (P X Y : Point) : Prop := sorry

-- Main theorem
theorem lines_intersect_at_one_point 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_E_ext : lies_on_extension E A B B)
  (h_F_ext : lies_on_extension F C D D)
  (h_M_mid_AD : is_midpoint M A D)
  (h_P_on_BE : point_on_line P B E)
  (h_Q_on_DF : point_on_line Q D F)
  (h_M_mid_PQ : is_midpoint M P Q) :
  ∃ O, point_on_line O A B ∧ point_on_line O C D ∧ point_on_line O P Q :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_one_point_l1958_195892


namespace NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l1958_195888

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | (x + 2*m) * (x - m + 4) < 0}
def B : Set ℝ := {x | (1 - x) / (x + 2) > 0}

-- Theorem for Question 1
theorem subset_condition (m : ℝ) : B ⊆ A m → m ≥ 5 ∨ m ≤ -1/2 := by
  sorry

-- Theorem for Question 2
theorem intersection_empty_condition (m : ℝ) : A m ∩ B = ∅ → 1 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l1958_195888


namespace NUMINAMATH_CALUDE_polynomial_parity_l1958_195850

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Multiplies two polynomials -/
def polyMult (p q : IntPolynomial) : IntPolynomial := sorry

/-- Checks if all elements in a list are even -/
def allEven (l : List Int) : Prop := ∀ x ∈ l, Even x

/-- Checks if all elements in a list are multiples of 4 -/
def allMultiplesOf4 (l : List Int) : Prop := ∀ x ∈ l, ∃ k, x = 4 * k

/-- Checks if at least one element in a list is odd -/
def hasOdd (l : List Int) : Prop := ∃ x ∈ l, Odd x

theorem polynomial_parity (P Q : IntPolynomial) :
  (allEven (polyMult P Q)) ∧ ¬(allMultiplesOf4 (polyMult P Q)) →
  ((allEven P ∧ hasOdd Q) ∨ (allEven Q ∧ hasOdd P)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_parity_l1958_195850


namespace NUMINAMATH_CALUDE_at_least_one_closed_under_mul_l1958_195809

/-- A set of real numbers closed under multiplication -/
structure ClosedMultSet (α : Type) [Mul α] := 
  (S : Set α)
  (hclosed : ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S)

theorem at_least_one_closed_under_mul 
  {α : Type} [Mul α] (X : ClosedMultSet α) 
  (A B : Set α) 
  (hS : X.S = A ∪ B) 
  (hAB : A ∩ B = ∅) 
  (hA : ∀ a b c, a ∈ A → b ∈ A → c ∈ A → (a * b * c) ∈ A) 
  (hB : ∀ a b c, a ∈ B → b ∈ B → c ∈ B → (a * b * c) ∈ B) : 
  (∀ x y, x ∈ A → y ∈ A → (x * y) ∈ A) ∨ 
  (∀ x y, x ∈ B → y ∈ B → (x * y) ∈ B) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_closed_under_mul_l1958_195809


namespace NUMINAMATH_CALUDE_no_integer_solution_l1958_195883

theorem no_integer_solution : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1958_195883


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1958_195873

theorem tangent_line_to_circle (r : ℝ) (h1 : r > 0) : 
  (∃ (x y : ℝ), x + y = 2*r ∧ (x - 1)^2 + (y - 1)^2 = r^2 ∧ 
   ∀ (x' y' : ℝ), x' + y' = 2*r → (x' - 1)^2 + (y' - 1)^2 ≥ r^2) →
  r = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1958_195873


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l1958_195880

theorem interest_equality_implies_second_sum (total : ℚ) 
  (h1 : total = 2665) 
  (h2 : ∃ x : ℚ, x * (3/100) * 8 = (total - x) * (5/100) * 3) : 
  ∃ second : ℚ, second = total - 2460 :=
sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l1958_195880


namespace NUMINAMATH_CALUDE_clara_pill_cost_l1958_195872

/-- The cost of pills for Clara's treatment --/
def pill_cost (blue_cost : ℚ) : Prop :=
  let days : ℕ := 10
  let red_cost : ℚ := blue_cost - 2
  let daily_cost : ℚ := blue_cost + red_cost
  let total_cost : ℚ := 480
  (days : ℚ) * daily_cost = total_cost ∧ blue_cost = 25

theorem clara_pill_cost : ∃ (blue_cost : ℚ), pill_cost blue_cost := by
  sorry

end NUMINAMATH_CALUDE_clara_pill_cost_l1958_195872


namespace NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l1958_195841

theorem largest_k_for_distinct_roots : 
  ∃ k : ℤ, k = 8 ∧ 
  (∀ m : ℤ, m > k → ¬(∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + m = 0 ∧ y^2 - 6*y + m = 0)) ∧
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l1958_195841


namespace NUMINAMATH_CALUDE_function_product_l1958_195891

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem function_product (y z : ℝ) 
  (h1 : -1 < y ∧ y < 1) 
  (h2 : -1 < z ∧ z < 1) 
  (h3 : f ((y + z) / (1 + y * z)) = 1) 
  (h4 : f ((y - z) / (1 - y * z)) = 2) : 
  f y * f z = -3/4 := by
sorry

end NUMINAMATH_CALUDE_function_product_l1958_195891


namespace NUMINAMATH_CALUDE_taxi_driver_theorem_l1958_195836

def taxi_trips : List Int := [5, -3, 6, -7, 6, -2, -5, 4, 6, -8]

theorem taxi_driver_theorem :
  (taxi_trips.take 7).sum = 0 ∧ taxi_trips.sum = 2 := by sorry

end NUMINAMATH_CALUDE_taxi_driver_theorem_l1958_195836


namespace NUMINAMATH_CALUDE_power_calculation_l1958_195857

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1958_195857


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l1958_195887

/-- Given a triangle PQR where ∠P is thrice ∠R and ∠Q is equal to ∠R, 
    the measure of ∠Q is 36°. -/
theorem angle_measure_in_special_triangle (P Q R : ℝ) : 
  P + Q + R = 180 →  -- sum of angles in a triangle
  P = 3 * R →        -- ∠P is thrice ∠R
  Q = R →            -- ∠Q is equal to ∠R
  Q = 36 :=          -- measure of ∠Q is 36°
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l1958_195887


namespace NUMINAMATH_CALUDE_caramel_candies_count_l1958_195867

theorem caramel_candies_count (total : ℕ) (lemon : ℕ) (caramel : ℕ) : 
  lemon = 4 →
  (caramel : ℚ) / (total : ℚ) = 3 / 7 →
  total = lemon + caramel →
  caramel = 3 := by
sorry

end NUMINAMATH_CALUDE_caramel_candies_count_l1958_195867


namespace NUMINAMATH_CALUDE_fried_green_tomatoes_l1958_195866

/-- Given that each tomato is cut into 8 slices and 20 tomatoes are needed to feed a family of 8 for a single meal, 
    prove that 20 slices are needed for a single person's meal. -/
theorem fried_green_tomatoes (slices_per_tomato : ℕ) (tomatoes_for_family : ℕ) (family_size : ℕ) 
  (h1 : slices_per_tomato = 8)
  (h2 : tomatoes_for_family = 20)
  (h3 : family_size = 8) :
  (slices_per_tomato * tomatoes_for_family) / family_size = 20 := by
  sorry

#check fried_green_tomatoes

end NUMINAMATH_CALUDE_fried_green_tomatoes_l1958_195866


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l1958_195839

/-- A type representing the balls -/
inductive Ball : Type
| one
| two
| three
| four

/-- A type representing the boxes -/
inductive Box : Type
| one
| two
| three
| four

/-- A function representing the placement of balls into boxes -/
def Placement := Ball → Box

/-- The event "ball number 1 is placed into box number 1" -/
def event1 (p : Placement) : Prop := p Ball.one = Box.one

/-- The event "ball number 1 is placed into box number 2" -/
def event2 (p : Placement) : Prop := p Ball.one = Box.two

/-- The sample space of all possible placements -/
def Ω : Set Placement := {p | ∀ b : Box, ∃! ball : Ball, p ball = b}

theorem events_mutually_exclusive_but_not_opposite :
  (∀ p ∈ Ω, ¬(event1 p ∧ event2 p)) ∧
  ¬(∀ p ∈ Ω, event1 p ↔ ¬event2 p) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l1958_195839


namespace NUMINAMATH_CALUDE_equation_solution_and_difference_l1958_195864

theorem equation_solution_and_difference :
  (∃ x : ℚ, 11 * x + 4 = 7) ∧
  (let x : ℚ := 3 / 11; 11 * x + 4 = 7) ∧
  (12 / 11 - 3 / 11 = 9 / 11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_difference_l1958_195864


namespace NUMINAMATH_CALUDE_amethyst_bead_count_l1958_195825

/-- Proves the number of amethyst beads in a specific necklace configuration -/
theorem amethyst_bead_count (total : ℕ) (turquoise : ℕ) (amethyst : ℕ) : 
  total = 40 → 
  turquoise = 19 → 
  total = amethyst + 2 * amethyst + turquoise → 
  amethyst = 7 := by
  sorry

end NUMINAMATH_CALUDE_amethyst_bead_count_l1958_195825


namespace NUMINAMATH_CALUDE_expression_evaluation_l1958_195801

theorem expression_evaluation : 5 + 7 * (2 + 1/4) = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1958_195801


namespace NUMINAMATH_CALUDE_second_train_speed_l1958_195828

/-- Calculates the speed of the second train given the parameters of two trains passing each other --/
theorem second_train_speed 
  (train1_length : ℝ)
  (train1_speed : ℝ)
  (train2_length : ℝ)
  (time_to_cross : ℝ)
  (h1 : train1_length = 420)
  (h2 : train1_speed = 72)
  (h3 : train2_length = 640)
  (h4 : time_to_cross = 105.99152067834574)
  : ∃ (train2_speed : ℝ), train2_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l1958_195828


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1958_195824

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1958_195824


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1958_195885

/-- Proves that a train of length 60 m crossing an electric pole in 1.4998800095992322 seconds has a speed of approximately 11.112 km/hr -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) 
  (h1 : train_length = 60) 
  (h2 : crossing_time = 1.4998800095992322) : 
  ∃ (speed : Real), abs (speed - 11.112) < 0.001 ∧ 
  speed = (train_length / crossing_time) * (3600 / 1000) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1958_195885


namespace NUMINAMATH_CALUDE_count_nonincreasing_7digit_integers_l1958_195843

/-- The number of 7-digit positive integers with nonincreasing digits -/
def nonincreasing_7digit_integers : ℕ :=
  Nat.choose 16 7 - 1

/-- Proposition: The number of 7-digit positive integers with nonincreasing digits is 11439 -/
theorem count_nonincreasing_7digit_integers :
  nonincreasing_7digit_integers = 11439 := by
  sorry

end NUMINAMATH_CALUDE_count_nonincreasing_7digit_integers_l1958_195843


namespace NUMINAMATH_CALUDE_equation_solution_l1958_195827

theorem equation_solution : ∃ x : ℚ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1958_195827


namespace NUMINAMATH_CALUDE_shaded_area_is_45_l1958_195813

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  bottomLeft : Point
  side : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  bottomRight : Point
  base : ℝ
  height : ℝ

/-- Calculates the area of the shaded region formed by the intersection of a square and a right triangle -/
def shadedArea (square : Square) (triangle : RightTriangle) : ℝ :=
  sorry

/-- Theorem stating that the shaded area is 45 square units given the specified conditions -/
theorem shaded_area_is_45 :
  ∀ (square : Square) (triangle : RightTriangle),
    square.bottomLeft = Point.mk 12 0 →
    square.side = 12 →
    triangle.bottomRight = Point.mk 12 0 →
    triangle.base = 12 →
    triangle.height = 9 →
    shadedArea square triangle = 45 :=
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_45_l1958_195813


namespace NUMINAMATH_CALUDE_system_solution_range_l1958_195851

theorem system_solution_range (x y m : ℝ) : 
  (x + 2*y = m + 4) →
  (2*x + y = 2*m - 1) →
  (x + y < 2) →
  (x - y < 4) →
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_range_l1958_195851


namespace NUMINAMATH_CALUDE_circle_symmetric_l1958_195871

-- Define a circle
def Circle : Type := Unit

-- Define axisymmetric property
def isAxisymmetric (shape : Type) : Prop := sorry

-- Define centrally symmetric property
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Theorem stating that a circle is both axisymmetric and centrally symmetric
theorem circle_symmetric : isAxisymmetric Circle ∧ isCentrallySymmetric Circle := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_l1958_195871


namespace NUMINAMATH_CALUDE_park_short_trees_l1958_195840

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem: The park will have 217 short trees after planting -/
theorem park_short_trees : 
  total_short_trees 112 105 = 217 := by
  sorry

end NUMINAMATH_CALUDE_park_short_trees_l1958_195840


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1958_195810

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1958_195810


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1958_195815

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1958_195815


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_five_l1958_195875

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a+1)*x + 2

-- State the theorem
theorem decreasing_f_implies_a_leq_neg_five (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -5 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_five_l1958_195875


namespace NUMINAMATH_CALUDE_alberts_earnings_l1958_195893

-- Define Albert's original earnings
def original_earnings : ℝ := 660

-- Theorem statement
theorem alberts_earnings :
  let scenario1 := original_earnings * 1.14 * 0.9
  let scenario2 := original_earnings * 1.15 * 1.2 * 0.9
  (scenario1 = 678) → (scenario2 = 819.72) := by
  sorry

end NUMINAMATH_CALUDE_alberts_earnings_l1958_195893


namespace NUMINAMATH_CALUDE_brown_mushrooms_count_l1958_195802

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := sorry

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The total number of white-spotted mushrooms -/
def total_white_spotted : ℕ := 17

theorem brown_mushrooms_count :
  brown_mushrooms = 6 :=
by
  have h1 : (blue_mushrooms / 2 : ℕ) + (2 * red_mushrooms / 3 : ℕ) + brown_mushrooms = total_white_spotted :=
    sorry
  sorry

end NUMINAMATH_CALUDE_brown_mushrooms_count_l1958_195802


namespace NUMINAMATH_CALUDE_paper_folding_ratio_l1958_195898

theorem paper_folding_ratio : 
  let square_side : ℝ := 1
  let large_rect_length : ℝ := square_side
  let large_rect_width : ℝ := square_side / 2
  let small_rect_length : ℝ := square_side
  let small_rect_width : ℝ := square_side / 4
  let large_rect_perimeter : ℝ := 2 * (large_rect_length + large_rect_width)
  let small_rect_perimeter : ℝ := 2 * (small_rect_length + small_rect_width)
  small_rect_perimeter / large_rect_perimeter = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_paper_folding_ratio_l1958_195898


namespace NUMINAMATH_CALUDE_cupcakes_needed_l1958_195826

theorem cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade : ℕ)
  (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_club : ℕ) :
  fourth_grade_classes = 8 →
  students_per_fourth_grade = 40 →
  pe_class_students = 80 →
  afterschool_clubs = 2 →
  students_per_club = 35 →
  fourth_grade_classes * students_per_fourth_grade +
  pe_class_students +
  afterschool_clubs * students_per_club = 470 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_needed_l1958_195826


namespace NUMINAMATH_CALUDE_parallelogram_inscribed_in_circle_is_rectangle_l1958_195874

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a parallelogram
def isParallelogram (q : Quadrilateral) : Prop := sorry

-- Define an inscribed quadrilateral
def isInscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define a rectangle
def isRectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem parallelogram_inscribed_in_circle_is_rectangle 
  (q : Quadrilateral) (c : Circle) : 
  isParallelogram q → isInscribed q c → isRectangle q := by sorry

end NUMINAMATH_CALUDE_parallelogram_inscribed_in_circle_is_rectangle_l1958_195874


namespace NUMINAMATH_CALUDE_food_drive_ratio_l1958_195895

theorem food_drive_ratio (total_students : ℕ) (no_cans_students : ℕ) (four_cans_students : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  no_cans_students = 2 →
  four_cans_students = 13 →
  total_cans = 232 →
  let twelve_cans_students := total_students - no_cans_students - four_cans_students
  (twelve_cans_students : ℚ) / total_students = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_food_drive_ratio_l1958_195895


namespace NUMINAMATH_CALUDE_max_product_distances_l1958_195852

/-- Given two perpendicular lines passing through fixed points A and B,
    prove that the maximum value of the product of distances from their
    intersection point P to A and B is |AB|²/2 -/
theorem max_product_distances (m : ℝ) : ∃ (P : ℝ × ℝ),
  (P.1 + m * P.2 = 0) ∧
  (m * P.1 - P.2 - m + 3 = 0) →
  ∀ (Q : ℝ × ℝ),
    (Q.1 + m * Q.2 = 0) ∧
    (m * Q.1 - Q.2 - m + 3 = 0) →
    (Q.1 - 0)^2 + (Q.2 - 0)^2 * ((Q.1 - 1)^2 + (Q.2 - 3)^2) ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_max_product_distances_l1958_195852


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1958_195805

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) < x + 3) ∧ ((x + 1) / 3 - x < 3) → -4 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1958_195805


namespace NUMINAMATH_CALUDE_train_speed_problem_l1958_195884

theorem train_speed_problem (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 44 →
  passing_time = 36 →
  train_length = 40 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (5/18) * passing_time = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1958_195884


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_l1958_195846

/-- Represents an n-digit number with all digits equal to d -/
def digits_number (n : ℕ+) (d : ℕ) : ℕ :=
  d * (10^n.val - 1) / 9

/-- The equation C_n - A_n = B_n^2 holds for at least two distinct values of n -/
def equation_holds (a b c : ℕ) : Prop :=
  ∃ n m : ℕ+, n ≠ m ∧
    digits_number (2*n) c - digits_number n a = (digits_number n b)^2 ∧
    digits_number (2*m) c - digits_number m a = (digits_number m b)^2

theorem smallest_sum_of_digits :
  ∀ a b c : ℕ,
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    equation_holds a b c →
    ∀ x y z : ℕ,
      0 < x ∧ x < 10 →
      0 < y ∧ y < 10 →
      0 < z ∧ z < 10 →
      equation_holds x y z →
      5 ≤ x + y + z :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_l1958_195846


namespace NUMINAMATH_CALUDE_inequality_holds_l1958_195858

theorem inequality_holds (x y : ℝ) : y^3 - x < |x^3| ↔ 
  (x ≥ 0 ∧ y^3 < x^3 + x) ∨ (x < 0 ∧ y^3 < -x^3 - x) := by sorry

end NUMINAMATH_CALUDE_inequality_holds_l1958_195858


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l1958_195845

/-- Given two lines with the same non-zero y-intercept, where the first line has
    a slope of 8 and an x-intercept of (u, 0), and the second line has a slope
    of 4 and an x-intercept of (v, 0), prove that the ratio of u to v is 1/2. -/
theorem ratio_of_x_intercepts (b : ℝ) (u v : ℝ) 
    (h1 : b ≠ 0)
    (h2 : 0 = 8 * u + b)
    (h3 : 0 = 4 * v + b) :
    u / v = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l1958_195845


namespace NUMINAMATH_CALUDE_permutations_mod_1000_l1958_195882

/-- The number of characters in the string -/
def n : ℕ := 16

/-- The number of A's in the string -/
def num_a : ℕ := 4

/-- The number of B's in the string -/
def num_b : ℕ := 5

/-- The number of C's in the string -/
def num_c : ℕ := 4

/-- The number of D's in the string -/
def num_d : ℕ := 3

/-- The number of positions where A's cannot be placed -/
def no_a_positions : ℕ := 5

/-- The number of positions where B's cannot be placed -/
def no_b_positions : ℕ := 5

/-- The number of positions where C's and D's cannot be placed -/
def no_cd_positions : ℕ := 6

/-- The function that calculates the number of permutations satisfying the conditions -/
def permutations : ℕ :=
  (Nat.choose no_cd_positions num_d) *
  (Nat.choose (no_cd_positions - num_d) (num_c - (no_cd_positions - num_d))) *
  (Nat.choose no_a_positions num_b) *
  (Nat.choose no_b_positions num_a)

theorem permutations_mod_1000 :
  permutations ≡ 75 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutations_mod_1000_l1958_195882


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1958_195854

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1958_195854


namespace NUMINAMATH_CALUDE_rectangle_area_l1958_195899

/-- A rectangle with length four times its width and perimeter 200 cm has an area of 1600 cm² --/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 → l * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1958_195899


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_two_l1958_195821

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x + 2) / (x - 3) = (x - k) / (x - 7)

-- Define the domain restriction
def valid_domain (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 7

-- Theorem statement
theorem no_solution_iff_k_eq_two :
  ∀ k : ℝ, (∀ x : ℝ, valid_domain x → ¬equation x k) ↔ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_two_l1958_195821


namespace NUMINAMATH_CALUDE_vitamin_a_content_l1958_195886

/-- The amount of Vitamin A in a single pill, in mg -/
def vitamin_a_per_pill : ℝ := 50

/-- The recommended daily serving of Vitamin A, in mg -/
def daily_recommended : ℝ := 200

/-- The number of pills needed for the weekly recommended amount -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem vitamin_a_content :
  vitamin_a_per_pill = daily_recommended * (days_per_week : ℝ) / (pills_per_week : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_content_l1958_195886


namespace NUMINAMATH_CALUDE_quartic_equation_sum_l1958_195817

theorem quartic_equation_sum (a b c : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℕ+, 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (∀ x : ℝ, x^4 - 10*x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
  a + b + c = 109 := by
sorry

end NUMINAMATH_CALUDE_quartic_equation_sum_l1958_195817


namespace NUMINAMATH_CALUDE_frog_jump_theorem_l1958_195838

/-- A regular polygon with 2n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) :=
  (n_ge_two : n ≥ 2)

/-- A configuration of frogs on the vertices of a regular polygon -/
structure FrogConfiguration (n : ℕ) :=
  (polygon : RegularPolygon n)
  (frogs : Fin (2*n) → Bool)

/-- A jumping method for the frogs -/
def JumpingMethod (n : ℕ) := Fin (2*n) → Bool

/-- Check if a line segment passes through the center of the circle -/
def passes_through_center (n : ℕ) (v1 v2 : Fin (2*n)) : Prop :=
  ∃ k : ℕ, v2 = v1 + n ∨ v1 = v2 + n

/-- The main theorem -/
theorem frog_jump_theorem (n : ℕ) :
  (∃ (config : FrogConfiguration n) (jump : JumpingMethod n),
    ∀ v1 v2 : Fin (2*n),
      v1 ≠ v2 →
      config.frogs v1 = true →
      config.frogs v2 = true →
      ¬passes_through_center n v1 v2) ↔
  n % 4 = 2 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_theorem_l1958_195838


namespace NUMINAMATH_CALUDE_solve_bus_problem_l1958_195869

def bus_problem (initial : ℕ) 
                (first_off : ℕ) 
                (second_off second_on : ℕ) 
                (third_off third_on : ℕ) : Prop :=
  let after_first := initial - first_off
  let after_second := after_first - second_off + second_on
  let after_third := after_second - third_off + third_on
  after_third = 28

theorem solve_bus_problem : 
  bus_problem 50 15 8 2 4 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_bus_problem_l1958_195869


namespace NUMINAMATH_CALUDE_curve_equation_l1958_195889

/-- Given a curve ax² + by² = 2 passing through (0, 5/3) and (1, 1), with a + b = 2,
    prove that the equation of the curve is 16/25 * x² + 9/25 * y² = 1 -/
theorem curve_equation (a b : ℝ) :
  (∀ x y : ℝ, a * x^2 + b * y^2 = 2) →
  (a * 0^2 + b * (5/3)^2 = 2) →
  (a * 1^2 + b * 1^2 = 2) →
  (a + b = 2) →
  (∀ x y : ℝ, 16/25 * x^2 + 9/25 * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_l1958_195889


namespace NUMINAMATH_CALUDE_base6_subtraction_addition_l1958_195819

-- Define a function to convert base-6 to decimal
def base6ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert decimal to base-6
def decimalToBase6 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base6_subtraction_addition :
  decimalToBase6 (base6ToDecimal 655 - base6ToDecimal 222 + base6ToDecimal 111) = 544 := by
  sorry

end NUMINAMATH_CALUDE_base6_subtraction_addition_l1958_195819


namespace NUMINAMATH_CALUDE_candy_bar_sales_l1958_195876

/-- The number of candy bars Marvin sold -/
def marvins_candy_bars : ℕ := 35

theorem candy_bar_sales : 
  let candy_bar_price : ℕ := 2
  let tinas_candy_bars : ℕ := 3 * marvins_candy_bars
  let marvins_revenue : ℕ := candy_bar_price * marvins_candy_bars
  let tinas_revenue : ℕ := candy_bar_price * tinas_candy_bars
  tinas_revenue = marvins_revenue + 140 → marvins_candy_bars = 35 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l1958_195876


namespace NUMINAMATH_CALUDE_initial_apples_in_pile_l1958_195863

def apple_pile (initial : ℕ) (added : ℕ) (final : ℕ) : Prop :=
  initial + added = final

def package_size : ℕ := 11

theorem initial_apples_in_pile : 
  ∃ (initial : ℕ), apple_pile initial 5 13 ∧ initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_in_pile_l1958_195863


namespace NUMINAMATH_CALUDE_candidate_count_l1958_195818

theorem candidate_count (total : ℕ) (selected_A selected_B : ℕ) : 
  selected_A = (6 * total) / 100 →
  selected_B = (7 * total) / 100 →
  selected_B = selected_A + 81 →
  total = 8100 := by
sorry

end NUMINAMATH_CALUDE_candidate_count_l1958_195818


namespace NUMINAMATH_CALUDE_last_two_digits_of_product_l1958_195807

theorem last_two_digits_of_product (n : ℕ) : 
  (33 * 92025^1989) % 100 = 25 := by
  sorry

#eval (33 * 92025^1989) % 100

end NUMINAMATH_CALUDE_last_two_digits_of_product_l1958_195807


namespace NUMINAMATH_CALUDE_special_trapezoid_base_difference_l1958_195804

/-- A trapezoid with specific angle and side length properties -/
structure SpecialTrapezoid where
  /-- The measure of one angle at the larger base in degrees -/
  angle1 : ℝ
  /-- The measure of the other angle at the larger base in degrees -/
  angle2 : ℝ
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the larger base -/
  larger_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- Condition: One angle at the larger base is 60° -/
  angle1_is_60 : angle1 = 60
  /-- Condition: The other angle at the larger base is 30° -/
  angle2_is_30 : angle2 = 30
  /-- Condition: The shorter leg is 5 units long -/
  shorter_leg_is_5 : shorter_leg = 5

/-- Theorem: The difference between the bases of the special trapezoid is 10 units -/
theorem special_trapezoid_base_difference (t : SpecialTrapezoid) :
  t.larger_base - t.shorter_base = 10 := by
  sorry


end NUMINAMATH_CALUDE_special_trapezoid_base_difference_l1958_195804


namespace NUMINAMATH_CALUDE_unfair_coin_expected_worth_l1958_195894

/-- An unfair coin with given probabilities for heads and tails, and corresponding gains and losses -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ
  prob_sum_one : prob_heads + prob_tails = 1
  prob_nonneg : prob_heads ≥ 0 ∧ prob_tails ≥ 0

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails)

/-- Theorem stating the expected worth of the specific unfair coin -/
theorem unfair_coin_expected_worth :
  ∃ (c : UnfairCoin),
    c.prob_heads = 2/3 ∧
    c.prob_tails = 1/3 ∧
    c.gain_heads = 5 ∧
    c.loss_tails = 6 ∧
    expected_worth c = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_worth_l1958_195894
