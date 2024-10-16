import Mathlib

namespace NUMINAMATH_CALUDE_integer_triplet_sum_product_l1727_172793

theorem integer_triplet_sum_product (a b c : ℤ) : 
  a < 4 ∧ b < 4 ∧ c < 4 ∧ 
  a < b ∧ b < c ∧ 
  a + b + c = a * b * c →
  ((a, b, c) = (1, 2, 3) ∨ 
   (a, b, c) = (-3, -2, -1) ∨ 
   (a, b, c) = (-1, 0, 1) ∨ 
   (a, b, c) = (-2, 0, 2) ∨ 
   (a, b, c) = (-3, 0, 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triplet_sum_product_l1727_172793


namespace NUMINAMATH_CALUDE_pizza_buffet_theorem_l1727_172744

theorem pizza_buffet_theorem (A B C : ℕ+) :
  (A : ℚ) = 1.8 * B ∧
  (B : ℚ) = C / 8 ∧
  A ≥ 2 ∧ B ≥ 1 ∧ C ≥ 8 →
  A = 2 ∧ B = 1 ∧ C = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_buffet_theorem_l1727_172744


namespace NUMINAMATH_CALUDE_simplify_expression_l1727_172790

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1727_172790


namespace NUMINAMATH_CALUDE_sculpture_exposed_area_l1727_172730

/-- Represents the sculpture with its properties --/
structure Sculpture where
  cubeEdge : Real
  bottomLayerCubes : Nat
  middleLayerCubes : Nat
  topLayerCubes : Nat
  submersionRatio : Real

/-- Calculates the exposed surface area of the sculpture --/
def exposedSurfaceArea (s : Sculpture) : Real :=
  sorry

/-- Theorem stating that the exposed surface area of the given sculpture is 12.75 square meters --/
theorem sculpture_exposed_area :
  let s : Sculpture := {
    cubeEdge := 0.5,
    bottomLayerCubes := 16,
    middleLayerCubes := 9,
    topLayerCubes := 1,
    submersionRatio := 0.5
  }
  exposedSurfaceArea s = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_exposed_area_l1727_172730


namespace NUMINAMATH_CALUDE_seller_total_loss_l1727_172766

/-- Represents the total loss of a seller in a transaction with a counterfeit banknote -/
def seller_loss (item_cost change_given fake_note_value real_note_value : ℕ) : ℕ :=
  item_cost + change_given + real_note_value

/-- Theorem stating the total loss of the seller in the given scenario -/
theorem seller_total_loss :
  let item_cost : ℕ := 20
  let customer_payment : ℕ := 100
  let change_given : ℕ := customer_payment - item_cost
  let fake_note_value : ℕ := 100
  let real_note_value : ℕ := 100
  seller_loss item_cost change_given fake_note_value real_note_value = 200 := by
  sorry

#eval seller_loss 20 80 100 100

end NUMINAMATH_CALUDE_seller_total_loss_l1727_172766


namespace NUMINAMATH_CALUDE_essays_total_pages_l1727_172733

def words_per_page : ℕ := 235

def johnny_words : ℕ := 195
def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 50
def samantha_words : ℕ := 3 * madeline_words
def ryan_words : ℕ := johnny_words + 100

def pages_needed (words : ℕ) : ℕ :=
  (words + words_per_page - 1) / words_per_page

def total_pages : ℕ :=
  pages_needed johnny_words +
  pages_needed madeline_words +
  pages_needed timothy_words +
  pages_needed samantha_words +
  pages_needed ryan_words

theorem essays_total_pages : total_pages = 12 := by
  sorry

end NUMINAMATH_CALUDE_essays_total_pages_l1727_172733


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1727_172762

/-- Given a quadratic function f(x) = ax² + bx + 1 with two distinct points
    (m, 2023) and (n, 2023) on its graph, prove that f(m + n) = 1 -/
theorem quadratic_function_property
  (a b m n : ℝ)
  (hm : a * m^2 + b * m + 1 = 2023)
  (hn : a * n^2 + b * n + 1 = 2023)
  (hd : m ≠ n) :
  a * (m + n)^2 + b * (m + n) + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1727_172762


namespace NUMINAMATH_CALUDE_overlap_ratio_l1727_172741

theorem overlap_ratio (circle_area square_area overlap_area : ℝ) 
  (h1 : overlap_area = 0.5 * circle_area)
  (h2 : overlap_area = 0.25 * square_area) :
  (square_area - overlap_area) / (circle_area + square_area - overlap_area) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_overlap_ratio_l1727_172741


namespace NUMINAMATH_CALUDE_batting_cage_pitches_per_token_l1727_172757

/-- The number of pitches per token at a batting cage -/
def pitches_per_token : ℕ := 15

/-- Macy's number of tokens -/
def macy_tokens : ℕ := 11

/-- Piper's number of tokens -/
def piper_tokens : ℕ := 17

/-- Macy's number of hits -/
def macy_hits : ℕ := 50

/-- Piper's number of hits -/
def piper_hits : ℕ := 55

/-- Total number of missed pitches -/
def total_misses : ℕ := 315

theorem batting_cage_pitches_per_token :
  (macy_tokens + piper_tokens) * pitches_per_token =
  macy_hits + piper_hits + total_misses :=
by sorry

end NUMINAMATH_CALUDE_batting_cage_pitches_per_token_l1727_172757


namespace NUMINAMATH_CALUDE_miriam_pushups_l1727_172723

/-- Calculates the number of push-ups Miriam does on Friday given her schedule for the week. -/
theorem miriam_pushups (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) 
  (h1 : monday = 5)
  (h2 : tuesday = 7)
  (h3 : wednesday = 2 * tuesday)
  (h4 : thursday = (monday + tuesday + wednesday) / 2)
  : monday + tuesday + wednesday + thursday = 39 := by
  sorry

end NUMINAMATH_CALUDE_miriam_pushups_l1727_172723


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1727_172736

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1727_172736


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1727_172721

/-- Given three equally spaced parallel lines intersecting a circle and creating
    chords of lengths 42, 36, and 36, prove that the distance between two
    adjacent parallel lines is 2√2006. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (x y z : ℝ), x = 42 ∧ y = 36 ∧ z = 36 ∧
   21 * x * 21 + (d/2)^2 * x = 21 * r^2 + 21 * r^2 ∧
   18 * y * 18 + (d/2)^2 * y = 18 * r^2 + 18 * r^2) →
  d = 2 * Real.sqrt 2006 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1727_172721


namespace NUMINAMATH_CALUDE_mikes_purchase_cost_l1727_172764

/-- The total cost of a camera and lens purchase -/
def total_cost (old_camera_cost lens_price lens_discount : ℚ) : ℚ :=
  let new_camera_cost := old_camera_cost * (1 + 0.3)
  let discounted_lens_price := lens_price - lens_discount
  new_camera_cost + discounted_lens_price

/-- Theorem stating the total cost of Mike's camera and lens purchase -/
theorem mikes_purchase_cost :
  total_cost 4000 400 200 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_mikes_purchase_cost_l1727_172764


namespace NUMINAMATH_CALUDE_hire_year_proof_l1727_172758

/-- Rule of 70 provision: An employee can retire when their age plus years of employment total at least 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1966

/-- The age at which the employee was hired -/
def hire_age : ℕ := 30

/-- The year the employee becomes eligible to retire -/
def retirement_eligibility_year : ℕ := 2006

/-- Theorem stating that an employee hired at age 30, who becomes eligible to retire under the rule of 70 provision in 2006, was hired in 1966 -/
theorem hire_year_proof :
  rule_of_70 (hire_age + (retirement_eligibility_year - hire_year)) (retirement_eligibility_year - hire_year) ∧
  hire_year = 1966 :=
sorry

end NUMINAMATH_CALUDE_hire_year_proof_l1727_172758


namespace NUMINAMATH_CALUDE_q_gt_one_not_sufficient_nor_necessary_l1727_172722

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem: "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∃ a q, GeometricSequence a q ∧ q > 1 ∧ ¬IncreasingSequence a) ∧
  (∃ a q, GeometricSequence a q ∧ IncreasingSequence a ∧ ¬(q > 1)) :=
sorry

end NUMINAMATH_CALUDE_q_gt_one_not_sufficient_nor_necessary_l1727_172722


namespace NUMINAMATH_CALUDE_degree_of_minus_5xy_squared_l1727_172726

/-- The type of monomials with integer coefficients in two variables -/
structure Monomial :=
  (coeff : ℤ)
  (x_exp : ℕ)
  (y_exp : ℕ)

/-- The degree of a monomial is the sum of its exponents -/
def degree (m : Monomial) : ℕ := m.x_exp + m.y_exp

/-- The monomial -5xy^2 -/
def m : Monomial := ⟨-5, 1, 2⟩

theorem degree_of_minus_5xy_squared :
  degree m = 3 := by sorry

end NUMINAMATH_CALUDE_degree_of_minus_5xy_squared_l1727_172726


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_pie_l1727_172745

theorem longest_segment_in_quarter_pie (d : ℝ) (h : d = 20) : 
  let r := d / 2
  let l := 2 * r * Real.sin (π / 4)
  l^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_pie_l1727_172745


namespace NUMINAMATH_CALUDE_writer_average_speed_l1727_172719

/-- Calculates the average writing speed given the words and hours for two writing sessions -/
def average_writing_speed (words1 : ℕ) (hours1 : ℕ) (words2 : ℕ) (hours2 : ℕ) : ℚ :=
  (words1 + words2 : ℚ) / (hours1 + hours2 : ℚ)

/-- Theorem stating that the average writing speed for the given sessions is 500 words per hour -/
theorem writer_average_speed :
  average_writing_speed 30000 60 50000 100 = 500 := by
  sorry

end NUMINAMATH_CALUDE_writer_average_speed_l1727_172719


namespace NUMINAMATH_CALUDE_triangle_isosceles_l1727_172788

/-- Given a triangle ABC with angles α, β, γ and sides a, b,
    if a + b = tan(γ/2) * (a * tan(α) + b * tan(β)),
    then the triangle ABC is isosceles. -/
theorem triangle_isosceles (α β γ a b : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ 
  α + β + γ = Real.pi ∧
  0 < a ∧ 0 < b ∧
  a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β) →
  a = b ∨ α = β := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l1727_172788


namespace NUMINAMATH_CALUDE_product_of_decimals_l1727_172761

theorem product_of_decimals : (0.5 : ℝ) * 0.8 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1727_172761


namespace NUMINAMATH_CALUDE_triangle_side_perp_distance_relation_l1727_172748

/-- Represents a triangle with side lengths and perpendicular distances -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ

/-- Theorem stating the relationship between side lengths and perpendicular distances -/
theorem triangle_side_perp_distance_relation (t : Triangle) 
  (h_side : t.a < t.b ∧ t.b < t.c) : 
  t.h_a > t.h_b ∧ t.h_b > t.h_c := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_perp_distance_relation_l1727_172748


namespace NUMINAMATH_CALUDE_root_condition_implies_a_range_l1727_172703

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1)*x + (a - 2)

-- State the theorem
theorem root_condition_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ f a x = 0 ∧ f a y = 0) →
  -2 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_a_range_l1727_172703


namespace NUMINAMATH_CALUDE_map_age_conversion_l1727_172781

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem map_age_conversion :
  octal_to_decimal 7324 = 2004 := by
  sorry

end NUMINAMATH_CALUDE_map_age_conversion_l1727_172781


namespace NUMINAMATH_CALUDE_problem_l1727_172780

theorem problem (a b : ℝ) (h : a - |b| > 0) : a^2 - b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_l1727_172780


namespace NUMINAMATH_CALUDE_paint_for_snake_2016_l1727_172700

/-- Amount of paint needed for a snake of cubes -/
def paint_for_snake (num_cubes : ℕ) (paint_per_cube : ℕ) : ℕ :=
  let periodic_fragment := 6
  let paint_per_fragment := periodic_fragment * paint_per_cube
  let num_fragments := num_cubes / periodic_fragment
  let paint_for_fragments := num_fragments * paint_per_fragment
  let paint_for_ends := 2 * (paint_per_cube / 3)
  paint_for_fragments + paint_for_ends

theorem paint_for_snake_2016 :
  paint_for_snake 2016 60 = 121000 := by
  sorry

end NUMINAMATH_CALUDE_paint_for_snake_2016_l1727_172700


namespace NUMINAMATH_CALUDE_initial_outlay_is_10000_l1727_172715

/-- Calculates the profit for a horseshoe manufacturing company --/
def horseshoe_profit (initial_outlay : ℝ) (sets_produced : ℕ) : ℝ :=
  let manufacturing_cost := initial_outlay + 20 * sets_produced
  let revenue := 50 * sets_produced
  revenue - manufacturing_cost

/-- Proves that the initial outlay is $10,000 given the conditions --/
theorem initial_outlay_is_10000 :
  ∃ (initial_outlay : ℝ),
    horseshoe_profit initial_outlay 500 = 5000 ∧
    initial_outlay = 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_outlay_is_10000_l1727_172715


namespace NUMINAMATH_CALUDE_kindergarten_class_size_l1727_172729

theorem kindergarten_class_size 
  (num_groups : ℕ) 
  (time_per_student : ℕ) 
  (time_per_group : ℕ) 
  (h1 : num_groups = 3)
  (h2 : time_per_student = 4)
  (h3 : time_per_group = 24) :
  num_groups * (time_per_group / time_per_student) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_kindergarten_class_size_l1727_172729


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1727_172705

theorem race_speed_ratio (va vb : ℝ) (h : va > 0 ∧ vb > 0) :
  (1 / va = (1 - 0.09523809523809523) / vb) → (va / vb = 21 / 19) := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1727_172705


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1727_172787

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive sequence
  a 1 = 1 →  -- a_1 = 1
  a 7 = a 6 + 2 * a 5 →  -- a_7 = a_6 + 2a_5
  (∃ q : ℝ, q > 0 ∧ ∀ k, a (k + 1) = q * a k) →  -- geometric sequence
  a m * a n = 16 →  -- a_m * a_n = 16
  m > 0 ∧ n > 0 →  -- m and n are positive
  1 / m + 4 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1727_172787


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_side_length_l1727_172738

theorem rectangle_area_ratio_side_length (area_ratio : ℚ) (p q r : ℕ) : 
  area_ratio = 500 / 125 →
  (p * Real.sqrt q : ℝ) / r = Real.sqrt (area_ratio) →
  p + q + r = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_side_length_l1727_172738


namespace NUMINAMATH_CALUDE_photo_arrangement_count_total_arrangement_count_l1727_172708

/-- The number of ways to arrange 6 people with specific conditions -/
def arrangement_count : ℕ := 480

/-- The number of people in the group -/
def total_people : ℕ := 6

/-- The number of people with specific positions (A, B, and C) -/
def specific_people : ℕ := 3

/-- The number of ways A and B can be arranged on one side of C -/
def ab_arrangements : ℕ := 4

/-- The number of ways to arrange the remaining people -/
def remaining_arrangements : ℕ := 120

theorem photo_arrangement_count :
  arrangement_count = ab_arrangements * remaining_arrangements :=
sorry

theorem total_arrangement_count :
  arrangement_count = 480 :=
sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_total_arrangement_count_l1727_172708


namespace NUMINAMATH_CALUDE_train_passing_time_specific_train_problem_l1727_172765

/-- The time taken for a faster train to completely pass a slower train -/
theorem train_passing_time (train_length : ℝ) (fast_speed slow_speed : ℝ) : ℝ :=
  let relative_speed := fast_speed - slow_speed
  let relative_speed_mps := relative_speed * (5 / 18)
  let total_distance := 2 * train_length
  total_distance / relative_speed_mps

/-- Proof of the specific train problem -/
theorem specific_train_problem :
  ∃ (t : ℝ), abs (t - train_passing_time 75 46 36) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_specific_train_problem_l1727_172765


namespace NUMINAMATH_CALUDE_surface_area_of_drilled_cube_l1727_172768

-- Define the cube
def cube_side_length : ℝ := 10

-- Define points on the cube
def point_A : ℝ × ℝ × ℝ := (0, 0, 0)
def point_G : ℝ × ℝ × ℝ := (cube_side_length, cube_side_length, cube_side_length)

-- Define the distance of H, I, J from A
def distance_from_A : ℝ := 3

-- Define the solid T
def solid_T (cube_side_length : ℝ) (distance_from_A : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

-- Calculate the surface area of the solid T
def surface_area_T (t : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem surface_area_of_drilled_cube :
  surface_area_T (solid_T cube_side_length distance_from_A) = 526.5 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_drilled_cube_l1727_172768


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1727_172709

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has no real roots -/
def has_no_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq < 0

theorem quadratic_no_real_roots :
  let eq_A : QuadraticEquation := ⟨1, 1, -2⟩
  let eq_B : QuadraticEquation := ⟨1, -2, 0⟩
  let eq_C : QuadraticEquation := ⟨1, 1, 5⟩
  let eq_D : QuadraticEquation := ⟨1, -2, 1⟩
  has_no_real_roots eq_C ∧
  ¬(has_no_real_roots eq_A ∨ has_no_real_roots eq_B ∨ has_no_real_roots eq_D) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1727_172709


namespace NUMINAMATH_CALUDE_lowest_possible_score_l1727_172775

theorem lowest_possible_score (num_tests : Nat) (max_score : Nat) (avg_score : Nat) :
  num_tests = 4 →
  max_score = 100 →
  avg_score = 88 →
  ∃ (scores : Fin num_tests → Nat),
    (∀ i, scores i ≤ max_score) ∧
    (Finset.sum Finset.univ (λ i => scores i) = num_tests * avg_score) ∧
    (∃ i, scores i = 52) ∧
    (∀ i, scores i ≥ 52) :=
by
  sorry

#check lowest_possible_score

end NUMINAMATH_CALUDE_lowest_possible_score_l1727_172775


namespace NUMINAMATH_CALUDE_heartsuit_three_five_l1727_172774

-- Define the ♥ operation
def heartsuit (x y : ℤ) : ℤ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_five_l1727_172774


namespace NUMINAMATH_CALUDE_percent_excess_l1727_172724

theorem percent_excess (M N k : ℝ) (hM : M > 0) (hN : N > 0) (hk : k > 0) :
  (M - k * N) / (k * N) * 100 = 100 * (M - k * N) / (k * N) := by
  sorry

end NUMINAMATH_CALUDE_percent_excess_l1727_172724


namespace NUMINAMATH_CALUDE_binary_calculation_l1727_172778

theorem binary_calculation : 
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end NUMINAMATH_CALUDE_binary_calculation_l1727_172778


namespace NUMINAMATH_CALUDE_cubic_factorization_l1727_172759

theorem cubic_factorization (a : ℝ) : a^3 - 16*a = a*(a+4)*(a-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1727_172759


namespace NUMINAMATH_CALUDE_calculation_proof_l1727_172710

theorem calculation_proof : (2014 * 2014 + 2012) - 2013 * 2013 = 6039 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1727_172710


namespace NUMINAMATH_CALUDE_c_minus_d_eq_neg_two_l1727_172771

-- Define g as an invertible function from ℝ to ℝ
noncomputable def g : ℝ → ℝ := sorry

-- Assume g is invertible
axiom g_invertible : Function.Injective g

-- Define c and d
def c : ℝ := sorry
def d : ℝ := sorry

-- State the conditions
axiom g_c_eq_d : g c = d
axiom g_d_eq_5 : g d = 5

-- Theorem to prove
theorem c_minus_d_eq_neg_two : c - d = -2 := by sorry

end NUMINAMATH_CALUDE_c_minus_d_eq_neg_two_l1727_172771


namespace NUMINAMATH_CALUDE_solution_set_a_eq_one_solution_set_is_real_l1727_172718

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x - 2

-- Part 1: Solution set for a = 1
theorem solution_set_a_eq_one :
  {x : ℝ | f 1 x ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Part 2: Conditions for solution set to be ℝ
theorem solution_set_is_real :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_one_solution_set_is_real_l1727_172718


namespace NUMINAMATH_CALUDE_equation_solution_l1727_172735

theorem equation_solution : ∃ x : ℝ, 90 + 5 * x / (180 / 3) = 91 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1727_172735


namespace NUMINAMATH_CALUDE_find_incorrect_value_l1727_172747

/-- Given a set of observations with an initial mean and a corrected mean after fixing one misrecorded value,
    this theorem proves that the original incorrect value can be determined. -/
theorem find_incorrect_value (n : ℕ) (m1 m2 x : ℚ) (hn : n > 0) :
  let y := n * m1 + x - n * m2
  y = 23 :=
by sorry

end NUMINAMATH_CALUDE_find_incorrect_value_l1727_172747


namespace NUMINAMATH_CALUDE_james_travel_time_l1727_172786

-- Define the parameters
def driving_speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

-- Define the theorem
theorem james_travel_time :
  (distance / driving_speed) + stop_time = 7 :=
by sorry

end NUMINAMATH_CALUDE_james_travel_time_l1727_172786


namespace NUMINAMATH_CALUDE_cube_intersection_figures_l1727_172754

-- Define the set of possible plane figures
inductive PlaneFigure
| EquilateralTriangle
| Trapezoid
| RightAngledTriangle
| Rectangle

-- Define the set of plane figures that can be obtained from cube intersection
def CubeIntersectionFigures : Set PlaneFigure :=
  {PlaneFigure.EquilateralTriangle, PlaneFigure.Trapezoid, PlaneFigure.Rectangle}

-- Theorem statement
theorem cube_intersection_figures :
  CubeIntersectionFigures = {PlaneFigure.EquilateralTriangle, PlaneFigure.Trapezoid, PlaneFigure.Rectangle} :=
by sorry

end NUMINAMATH_CALUDE_cube_intersection_figures_l1727_172754


namespace NUMINAMATH_CALUDE_rainfall_rate_l1727_172797

/-- Rainfall problem statement -/
theorem rainfall_rate (monday_hours monday_rate tuesday_hours wednesday_hours total_rainfall : ℝ) 
  (h1 : monday_hours = 7)
  (h2 : monday_rate = 1)
  (h3 : tuesday_hours = 4)
  (h4 : wednesday_hours = 2)
  (h5 : total_rainfall = 23)
  : ∃ tuesday_rate : ℝ, 
    monday_hours * monday_rate + tuesday_hours * tuesday_rate + wednesday_hours * (2 * tuesday_rate) = total_rainfall ∧ 
    tuesday_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_rate_l1727_172797


namespace NUMINAMATH_CALUDE_bag_probability_l1727_172779

/-- Given a bag of 5 balls where the probability of picking a red ball is 0.4,
    prove that the probability of picking exactly one red ball and one white ball
    when two balls are picked is 3/5 -/
theorem bag_probability (total_balls : ℕ) (prob_red : ℝ) :
  total_balls = 5 →
  prob_red = 0.4 →
  (2 : ℝ) * prob_red * (1 - prob_red) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_bag_probability_l1727_172779


namespace NUMINAMATH_CALUDE_function_identity_l1727_172785

def IsNonDegenerateTriangle (a b c : ℕ+) : Prop :=
  a.val + b.val > c.val ∧ b.val + c.val > a.val ∧ c.val + a.val > b.val

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ (a b : ℕ+), IsNonDegenerateTriangle a (f b) (f (b + f a - 1))) :
  ∀ (a : ℕ+), f a = a := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l1727_172785


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1727_172791

theorem fraction_equivalence : ∃ n : ℚ, (4 + n) / (7 + n) = 2 / 3 :=
by
  use 2
  sorry

#check fraction_equivalence

end NUMINAMATH_CALUDE_fraction_equivalence_l1727_172791


namespace NUMINAMATH_CALUDE_sin_cos_relation_l1727_172702

theorem sin_cos_relation (α : ℝ) : 
  2 * Real.sin (α - π/3) = (2 - Real.sqrt 3) * Real.cos α → 
  Real.sin (2*α) + 3 * (Real.cos α)^2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l1727_172702


namespace NUMINAMATH_CALUDE_inequality_proof_l1727_172701

theorem inequality_proof (a b c : ℝ) :
  a = 31/32 →
  b = Real.cos (1/4) →
  c = 4 * Real.sin (1/4) →
  c > b ∧ b > a := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1727_172701


namespace NUMINAMATH_CALUDE_lcm_problem_l1727_172763

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1727_172763


namespace NUMINAMATH_CALUDE_golden_ratio_and_relations_l1727_172782

theorem golden_ratio_and_relations :
  -- Part 1: Golden Ratio
  (∃ x : ℝ, x > 0 ∧ x^2 + x - 1 = 0 ∧ x = (-1 + Real.sqrt 5) / 2) ∧
  -- Part 2: Relation between a and b
  (∀ m a b : ℝ, a^2 + m*a = 1 → b^2 - 2*m*b = 4 → b ≠ -2*a → a*b = 2) ∧
  -- Part 3: Relation between p, q, and n
  (∀ n p q : ℝ, p ≠ q → p^2 + n*p - 1 = q → q^2 + n*q - 1 = p → p*q - n = 0) :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_and_relations_l1727_172782


namespace NUMINAMATH_CALUDE_sum_six_to_thousand_l1727_172749

/-- Count of digit 6 occurrences in a number -/
def count_six (n : ℕ) : ℕ := sorry

/-- Sum of digit 6 occurrences from 1 to n -/
def sum_six_occurrences (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digit 6 occurrences from 1 to 1000 is 300 -/
theorem sum_six_to_thousand :
  sum_six_occurrences 1000 = 300 := by sorry

end NUMINAMATH_CALUDE_sum_six_to_thousand_l1727_172749


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l1727_172776

/-- The height of a tree after a given number of years, given that it triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem stating that a tree tripling its height yearly reaches 9 feet after 2 years if it's 81 feet after 4 years -/
theorem tree_height_after_two_years 
  (h : tree_height (tree_height h₀ 2) 2 = 81) : 
  tree_height h₀ 2 = 9 :=
by
  sorry

#check tree_height_after_two_years

end NUMINAMATH_CALUDE_tree_height_after_two_years_l1727_172776


namespace NUMINAMATH_CALUDE_number_of_correct_propositions_is_zero_l1727_172706

/-- Definition of a Frustum -/
structure Frustum where
  -- A frustum has two parallel faces (base and top)
  has_parallel_faces : Bool
  -- The extensions of lateral edges intersect at a point
  lateral_edges_intersect : Bool
  -- The extensions of waists of lateral faces intersect at a point
  waists_intersect : Bool

/-- Definition of a proposition about frustums -/
structure FrustumProposition where
  statement : String
  is_correct : Bool

/-- The three given propositions -/
def propositions : List FrustumProposition := [
  { statement := "Cutting a pyramid with a plane, the part between the base of the pyramid and the section is a frustum",
    is_correct := false },
  { statement := "A polyhedron with two parallel and similar bases, and all other faces being trapezoids, is a frustum",
    is_correct := false },
  { statement := "A hexahedron with two parallel faces and the other four faces being isosceles trapezoids is a frustum",
    is_correct := false }
]

/-- Theorem: The number of correct propositions is zero -/
theorem number_of_correct_propositions_is_zero :
  (propositions.filter (λ p => p.is_correct)).length = 0 := by
  sorry

end NUMINAMATH_CALUDE_number_of_correct_propositions_is_zero_l1727_172706


namespace NUMINAMATH_CALUDE_valid_closed_broken_line_segments_l1727_172789

/-- A closed broken line where each segment intersects exactly once and no three segments share a common point. -/
structure ClosedBrokenLine where
  segments : ℕ
  is_closed : Bool
  each_segment_intersects_once : Bool
  no_three_segments_share_point : Bool

/-- Predicate to check if a ClosedBrokenLine is valid -/
def is_valid_closed_broken_line (line : ClosedBrokenLine) : Prop :=
  line.is_closed ∧ line.each_segment_intersects_once ∧ line.no_three_segments_share_point

/-- Theorem stating that a valid ClosedBrokenLine can have 1996 segments but not 1997 -/
theorem valid_closed_broken_line_segments :
  (∃ (line : ClosedBrokenLine), line.segments = 1996 ∧ is_valid_closed_broken_line line) ∧
  (¬ ∃ (line : ClosedBrokenLine), line.segments = 1997 ∧ is_valid_closed_broken_line line) := by
  sorry

end NUMINAMATH_CALUDE_valid_closed_broken_line_segments_l1727_172789


namespace NUMINAMATH_CALUDE_problem_solution_l1727_172756

theorem problem_solution (x : ℚ) : 5 * x - 8 = 12 * x + 15 → 5 * (x + 4) = 25 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1727_172756


namespace NUMINAMATH_CALUDE_max_equal_distribution_l1727_172752

theorem max_equal_distribution (bags eyeliners scarves hairbands : ℕ) 
  (h1 : bags = 2923)
  (h2 : eyeliners = 3239)
  (h3 : scarves = 1785)
  (h4 : hairbands = 1379) :
  Nat.gcd bags (Nat.gcd eyeliners (Nat.gcd scarves hairbands)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_equal_distribution_l1727_172752


namespace NUMINAMATH_CALUDE_triangle_inequality_l1727_172704

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^3 + b^3 + c^3 + 4*a*b*c ≤ 9/32 * (a + b + c)^3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1727_172704


namespace NUMINAMATH_CALUDE_simplify_expression_l1727_172750

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1727_172750


namespace NUMINAMATH_CALUDE_inequality_proof_l1727_172732

theorem inequality_proof (n : ℕ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1727_172732


namespace NUMINAMATH_CALUDE_sine_cosine_shift_l1727_172773

theorem sine_cosine_shift (ω : ℝ) (h_ω : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 8)
  let g : ℝ → ℝ := λ x ↦ Real.cos (ω * x)
  (∀ x : ℝ, f (x + π / ω) = f x) →
  ∃ k : ℝ, k = 3 * π / 16 ∧ ∀ x : ℝ, g x = f (x + k) :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_shift_l1727_172773


namespace NUMINAMATH_CALUDE_percentage_of_women_l1727_172713

theorem percentage_of_women (initial_workers : ℕ) (initial_men_fraction : ℚ) 
  (new_hires : ℕ) : 
  initial_workers = 90 → 
  initial_men_fraction = 2/3 → 
  new_hires = 10 → 
  let total_workers := initial_workers + new_hires
  let initial_women := initial_workers * (1 - initial_men_fraction)
  let total_women := initial_women + new_hires
  (total_women / total_workers : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_l1727_172713


namespace NUMINAMATH_CALUDE_equation_solution_l1727_172770

theorem equation_solution :
  ∃ x : ℚ, (7 + 3.5 * x = 2.1 * x - 30 * 1.5) ∧ (x = -520 / 14) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1727_172770


namespace NUMINAMATH_CALUDE_clock_correction_l1727_172772

/-- The daily gain of the clock in minutes -/
def daily_gain : ℚ := 13 / 4

/-- The number of hours between 8 A.M. on April 10 and 3 P.M. on April 19 -/
def total_hours : ℕ := 223

/-- The negative correction in minutes to be subtracted from the clock -/
def m : ℚ := (daily_gain * total_hours) / 24

theorem clock_correction : m = 30 + 13 / 96 := by
  sorry

end NUMINAMATH_CALUDE_clock_correction_l1727_172772


namespace NUMINAMATH_CALUDE_exam_scores_l1727_172783

theorem exam_scores (total_students : Nat) (high_scorers : Nat) (high_score : Nat) 
  (rest_average : Nat) (class_average : Nat) 
  (h1 : total_students = 25)
  (h2 : high_scorers = 3)
  (h3 : high_score = 95)
  (h4 : rest_average = 45)
  (h5 : class_average = 42) : 
  ∃ zero_scorers : Nat, 
    (zero_scorers + high_scorers + (total_students - zero_scorers - high_scorers)) = total_students ∧
    (high_scorers * high_score + (total_students - zero_scorers - high_scorers) * rest_average) 
      = (total_students * class_average) ∧
    zero_scorers = 5 := by
  sorry

end NUMINAMATH_CALUDE_exam_scores_l1727_172783


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1727_172725

/-- 
Given an arithmetic sequence {a_n} with common difference -2,
if a_1, a_4, and a_5 form a geometric sequence, then a_3 = 5.
-/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n - 2) →  -- arithmetic sequence with common difference -2
  (a 4)^2 = a 1 * a 5 →         -- a_1, a_4, a_5 form a geometric sequence
  a 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1727_172725


namespace NUMINAMATH_CALUDE_binomial_10_4_l1727_172737

theorem binomial_10_4 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_4_l1727_172737


namespace NUMINAMATH_CALUDE_R3_sequence_arithmetic_l1727_172769

def is_R3_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 3 → a (n - 3) + a (n + 3) = 2 * a n) ∧
  (∀ n : ℕ, a (n + 1) ≥ a n)

def is_arithmetic_subsequence (b : ℕ → ℝ) (start : ℕ) (step : ℕ) (count : ℕ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, i < count → b (start + i * step) - b (start + (i + 1) * step) = d

theorem R3_sequence_arithmetic (a : ℕ → ℝ) (h1 : is_R3_sequence a) 
  (h2 : ∃ p : ℕ, p > 1 ∧ is_arithmetic_subsequence a (3 * p - 3) 2 4) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d := by
  sorry

end NUMINAMATH_CALUDE_R3_sequence_arithmetic_l1727_172769


namespace NUMINAMATH_CALUDE_one_statement_implies_negation_l1727_172798

theorem one_statement_implies_negation (p q r : Prop) : 
  let statement1 := p ∧ q ∧ ¬r
  let statement2 := p ∧ ¬q ∧ r
  let statement3 := ¬p ∧ q ∧ ¬r
  let statement4 := ¬p ∧ ¬q ∧ r
  let negation := ¬((p ∧ q) ∨ r)
  ∃! x : Fin 4, match x with
    | 0 => statement1 → negation
    | 1 => statement2 → negation
    | 2 => statement3 → negation
    | 3 => statement4 → negation
  := by sorry

end NUMINAMATH_CALUDE_one_statement_implies_negation_l1727_172798


namespace NUMINAMATH_CALUDE_evaluate_power_of_power_l1727_172760

theorem evaluate_power_of_power : (3^3)^2 = 729 := by sorry

end NUMINAMATH_CALUDE_evaluate_power_of_power_l1727_172760


namespace NUMINAMATH_CALUDE_identical_solutions_l1727_172711

theorem identical_solutions (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 ∧ y = 3*x + k) ↔ k = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_identical_solutions_l1727_172711


namespace NUMINAMATH_CALUDE_min_value_theorem_l1727_172796

theorem min_value_theorem (x y k : ℝ) 
  (hx : x > k) (hy : y > k) (hk : k > 1) :
  ∃ (m : ℝ), m = 8 * k ∧ 
  ∀ (a b : ℝ), a > k → b > k → 
  (a^2 / (b - k) + b^2 / (a - k)) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1727_172796


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1727_172734

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∀ x : ℝ, x^2 - x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1727_172734


namespace NUMINAMATH_CALUDE_probability_three_fives_out_of_five_dice_probability_exactly_three_fives_l1727_172727

/-- The probability of exactly 3 out of 5 fair 10-sided dice showing the number 5 -/
theorem probability_three_fives_out_of_five_dice : ℚ :=
  81 / 10000

/-- A fair 10-sided die -/
def fair_10_sided_die : Finset ℕ := Finset.range 10

/-- The probability of rolling a 5 on a fair 10-sided die -/
def prob_roll_5 : ℚ := 1 / 10

/-- The probability of not rolling a 5 on a fair 10-sided die -/
def prob_not_roll_5 : ℚ := 9 / 10

/-- The number of ways to choose 3 dice out of 5 -/
def ways_to_choose_3_out_of_5 : ℕ := 10

theorem probability_exactly_three_fives (n : ℕ) (k : ℕ) 
  (h1 : n = 5) (h2 : k = 3) : 
  probability_three_fives_out_of_five_dice = 
    (ways_to_choose_3_out_of_5 : ℚ) * (prob_roll_5 ^ k) * (prob_not_roll_5 ^ (n - k)) :=
sorry

end NUMINAMATH_CALUDE_probability_three_fives_out_of_five_dice_probability_exactly_three_fives_l1727_172727


namespace NUMINAMATH_CALUDE_total_worth_of_cloth_sold_l1727_172717

/-- Calculates the total worth of cloth sold through two agents given their commission rates and amounts -/
theorem total_worth_of_cloth_sold 
  (rate_A rate_B : ℝ) 
  (commission_A commission_B : ℝ) 
  (h1 : rate_A = 0.025) 
  (h2 : rate_B = 0.03) 
  (h3 : commission_A = 21) 
  (h4 : commission_B = 27) : 
  ∃ (total_worth : ℝ), total_worth = commission_A / rate_A + commission_B / rate_B :=
sorry

end NUMINAMATH_CALUDE_total_worth_of_cloth_sold_l1727_172717


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1727_172742

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + y^2 + 1 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1727_172742


namespace NUMINAMATH_CALUDE_amusement_park_admission_fee_l1727_172751

theorem amusement_park_admission_fee (child_fee : ℝ) (total_people : ℕ) (total_fee : ℝ) (num_children : ℕ) :
  child_fee = 1.5 →
  total_people = 315 →
  total_fee = 810 →
  num_children = 180 →
  ∃ (adult_fee : ℝ), adult_fee = 4 ∧ 
    child_fee * num_children + adult_fee * (total_people - num_children) = total_fee :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_fee_l1727_172751


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l1727_172728

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    if the rectangle's area is three times the circle's area,
    then the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (circle_radius : ℝ) (rectangle_shorter_side rectangle_longer_side : ℝ) :
  circle_radius = 6 →
  rectangle_shorter_side = 2 * circle_radius →
  rectangle_shorter_side * rectangle_longer_side = 3 * Real.pi * circle_radius^2 →
  rectangle_longer_side = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l1727_172728


namespace NUMINAMATH_CALUDE_nell_remaining_cards_nell_remaining_cards_proof_l1727_172767

/-- 
Given that Nell initially had 304 baseball cards and gave 28 cards to Jeff,
this theorem proves that Nell now has 276 cards.
-/
theorem nell_remaining_cards : ℕ → ℕ → ℕ → Prop :=
  fun initial_cards cards_given remaining_cards =>
    initial_cards = 304 →
    cards_given = 28 →
    remaining_cards = initial_cards - cards_given →
    remaining_cards = 276

-- The proof is omitted
theorem nell_remaining_cards_proof : nell_remaining_cards 304 28 276 := by
  sorry

end NUMINAMATH_CALUDE_nell_remaining_cards_nell_remaining_cards_proof_l1727_172767


namespace NUMINAMATH_CALUDE_unique_functional_equation_solution_l1727_172799

theorem unique_functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) = x * f x + y * f y) →
  (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_solution_l1727_172799


namespace NUMINAMATH_CALUDE_downstream_speed_l1727_172712

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Theorem stating the relationship between upstream, stillwater, and downstream speeds -/
theorem downstream_speed (s : RowingSpeed) 
  (h1 : s.upstream = 15) 
  (h2 : s.stillWater = 25) : 
  s.downstream = 35 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_l1727_172712


namespace NUMINAMATH_CALUDE_weekly_calorie_allowance_is_11700_l1727_172755

/-- Represents the weekly calorie allowance calculation for a person in their 60's --/
def weekly_calorie_allowance : ℕ :=
  let average_daily_allowance : ℕ := 2000
  let daily_reduction : ℕ := 500
  let reduced_daily_allowance : ℕ := average_daily_allowance - daily_reduction
  let intense_workout_days : ℕ := 2
  let moderate_exercise_days : ℕ := 3
  let rest_days : ℕ := 2
  let intense_workout_extra_calories : ℕ := 300
  let moderate_exercise_extra_calories : ℕ := 200
  
  (reduced_daily_allowance + intense_workout_extra_calories) * intense_workout_days +
  (reduced_daily_allowance + moderate_exercise_extra_calories) * moderate_exercise_days +
  reduced_daily_allowance * rest_days

/-- Theorem stating that the weekly calorie allowance is 11700 calories --/
theorem weekly_calorie_allowance_is_11700 : 
  weekly_calorie_allowance = 11700 := by
  sorry

end NUMINAMATH_CALUDE_weekly_calorie_allowance_is_11700_l1727_172755


namespace NUMINAMATH_CALUDE_smallest_student_count_l1727_172746

/-- Represents the number of students in each grade --/
structure GradeCount where
  eighth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- Checks if the given grade counts satisfy the required ratios --/
def satisfiesRatios (gc : GradeCount) : Prop :=
  gc.eighth * 4 = gc.seventh * 7 ∧ gc.seventh * 9 = gc.sixth * 10

/-- Theorem stating the smallest possible total number of students --/
theorem smallest_student_count :
  ∃ (gc : GradeCount), satisfiesRatios gc ∧
    gc.eighth + gc.seventh + gc.sixth = 73 ∧
    (∀ (gc' : GradeCount), satisfiesRatios gc' →
      gc'.eighth + gc'.seventh + gc'.sixth ≥ 73) :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_count_l1727_172746


namespace NUMINAMATH_CALUDE_exchange_three_cows_to_chickens_l1727_172714

/-- Exchange rates between animals -/
structure ExchangeRates where
  cows_to_sheep : ℚ      -- Rate of cows to sheep
  sheep_to_rabbits : ℚ   -- Rate of sheep to rabbits
  rabbits_to_chickens : ℚ -- Rate of rabbits to chickens

/-- Given the exchange rates, calculate how many chickens can be exchanged for a given number of cows -/
def cows_to_chickens (rates : ExchangeRates) (num_cows : ℚ) : ℚ :=
  num_cows * rates.cows_to_sheep * rates.sheep_to_rabbits * rates.rabbits_to_chickens

/-- Theorem stating that 3 cows can be exchanged for 819 chickens given the specified exchange rates -/
theorem exchange_three_cows_to_chickens :
  let rates : ExchangeRates := {
    cows_to_sheep := 42 / 2,
    sheep_to_rabbits := 26 / 3,
    rabbits_to_chickens := 3 / 2
  }
  cows_to_chickens rates 3 = 819 := by
  sorry


end NUMINAMATH_CALUDE_exchange_three_cows_to_chickens_l1727_172714


namespace NUMINAMATH_CALUDE_parent_payment_calculation_l1727_172794

/-- Calculates the amount each parent has to pay in different currencies --/
theorem parent_payment_calculation
  (former_salary : ℝ)
  (raise_percentage : ℝ)
  (tax_rate : ℝ)
  (num_kids : ℕ)
  (usd_to_eur : ℝ)
  (usd_to_gbp : ℝ)
  (usd_to_jpy : ℝ)
  (h1 : former_salary = 60000)
  (h2 : raise_percentage = 0.25)
  (h3 : tax_rate = 0.10)
  (h4 : num_kids = 15)
  (h5 : usd_to_eur = 0.85)
  (h6 : usd_to_gbp = 0.75)
  (h7 : usd_to_jpy = 110) :
  let new_salary := former_salary * (1 + raise_percentage)
  let after_tax_salary := new_salary * (1 - tax_rate)
  let amount_per_parent := after_tax_salary / num_kids
  (amount_per_parent / usd_to_eur = 5294.12) ∧
  (amount_per_parent / usd_to_gbp = 6000) ∧
  (amount_per_parent * usd_to_jpy = 495000) :=
by sorry

end NUMINAMATH_CALUDE_parent_payment_calculation_l1727_172794


namespace NUMINAMATH_CALUDE_roots_sum_zero_l1727_172731

theorem roots_sum_zero (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : Real.log (abs x₁) = m) 
  (h₂ : Real.log (abs x₂) = m) 
  (h₃ : x₁ ≠ x₂) : 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_zero_l1727_172731


namespace NUMINAMATH_CALUDE_a_cubed_congruent_implies_a_sixth_congruent_l1727_172753

theorem a_cubed_congruent_implies_a_sixth_congruent (n : ℕ+) (a : ℤ) 
  (h : a^3 ≡ 1 [ZMOD n]) : a^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_a_cubed_congruent_implies_a_sixth_congruent_l1727_172753


namespace NUMINAMATH_CALUDE_point_A_in_second_quadrant_l1727_172777

/-- A point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: The point A(-2, 3) is in the second quadrant -/
theorem point_A_in_second_quadrant :
  let A : Point2D := ⟨-2, 3⟩
  isInSecondQuadrant A := by
  sorry


end NUMINAMATH_CALUDE_point_A_in_second_quadrant_l1727_172777


namespace NUMINAMATH_CALUDE_positive_number_has_square_root_l1727_172716

theorem positive_number_has_square_root :
  ∀ x : ℝ, x > 0 → ∃ y : ℝ, y * y = x :=
by sorry

end NUMINAMATH_CALUDE_positive_number_has_square_root_l1727_172716


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1727_172740

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (containedIn : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : containedIn m α)
  (h4 : containedIn n β)
  (h5 : perpendicular n α) :
  planePerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1727_172740


namespace NUMINAMATH_CALUDE_x_value_proof_l1727_172707

theorem x_value_proof :
  let equation := (2021 / 2022 - 2022 / 2021) + x = 0
  ∃ x, equation ∧ x = 2022 / 2021 - 2021 / 2022 :=
by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1727_172707


namespace NUMINAMATH_CALUDE_parallel_intersection_false_l1727_172792

-- Define the types for planes and lines
variable (α β : Plane) (m n : Line)

-- Define the parallel and intersection relations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem parallel_intersection_false :
  ¬(∀ α β m n,
    (parallel m α ∧ intersection α β = n) → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_intersection_false_l1727_172792


namespace NUMINAMATH_CALUDE_height_difference_pablo_charlene_l1727_172795

/-- Given the heights of various people, prove the height difference between Pablo and Charlene. -/
theorem height_difference_pablo_charlene :
  ∀ (height_janet height_ruby height_pablo height_charlene : ℕ),
  height_janet = 62 →
  height_charlene = 2 * height_janet →
  height_ruby = 192 →
  height_pablo = height_ruby + 2 →
  height_pablo - height_charlene = 70 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_pablo_charlene_l1727_172795


namespace NUMINAMATH_CALUDE_lauren_change_calculation_l1727_172739

/-- Calculates the change Lauren receives after grocery shopping --/
theorem lauren_change_calculation : 
  let hamburger_meat_price : ℝ := 3.50
  let hamburger_meat_weight : ℝ := 2
  let buns_price : ℝ := 1.50
  let lettuce_price : ℝ := 1.00
  let tomato_price_per_pound : ℝ := 2.00
  let tomato_weight : ℝ := 1.5
  let pickles_price : ℝ := 2.50
  let coupon_value : ℝ := 1.00
  let paid_amount : ℝ := 20.00

  let total_cost : ℝ := 
    hamburger_meat_price * hamburger_meat_weight +
    buns_price + 
    lettuce_price + 
    tomato_price_per_pound * tomato_weight + 
    pickles_price - 
    coupon_value

  let change : ℝ := paid_amount - total_cost

  change = 6.00 := by sorry

end NUMINAMATH_CALUDE_lauren_change_calculation_l1727_172739


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1727_172743

theorem expression_simplification_and_evaluation (m : ℤ) 
  (h1 : -2 ≤ m ∧ m ≤ 2) 
  (h2 : m ≠ -2 ∧ m ≠ 0 ∧ m ≠ 1 ∧ m ≠ 2) :
  (m / (m - 2) - 4 / (m^2 - 2*m)) / ((m + 2) / (m^2 - m)) = 
  ((m - 4) * (m + 1) * (m - 1)) / (m * (m - 2) * (m + 2)) ∧
  ((m - 4) * (m + 1) * (m - 1)) / (m * (m - 2) * (m + 2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1727_172743


namespace NUMINAMATH_CALUDE_ellipse_m_value_l1727_172720

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop := x^2 / m + y^2 / 16 = 1

-- Define the distances from a point to the foci
def distance_to_foci (d1 d2 : ℝ) : Prop := d1 = 3 ∧ d2 = 7

-- Theorem statement
theorem ellipse_m_value (x y m : ℝ) :
  ellipse_equation x y m →
  ∃ (d1 d2 : ℝ), distance_to_foci d1 d2 →
  m = 25 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l1727_172720


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1727_172784

/-- Represents a parabola in the form y² = -4px where p is the focal length -/
structure Parabola where
  p : ℝ

/-- The focus of a parabola -/
def Parabola.focus (par : Parabola) : ℝ × ℝ := (-par.p, 0)

/-- The x-coordinate of the directrix of a parabola -/
def Parabola.directrix (par : Parabola) : ℝ := par.p

theorem parabola_focus_and_directrix (par : Parabola) 
  (h : par.p = 2) : 
  (par.focus = (-2, 0)) ∧ (par.directrix = 2) := by
  sorry

#check parabola_focus_and_directrix

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1727_172784
