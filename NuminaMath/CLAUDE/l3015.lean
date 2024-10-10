import Mathlib

namespace cube_labeling_impossible_l3015_301569

/-- Represents a cube with vertices labeled by natural numbers -/
structure LabeledCube :=
  (vertices : Fin 8 → ℕ)
  (is_permutation : Function.Bijective vertices)

/-- The set of edges in a cube -/
def cube_edges : Finset (Fin 8 × Fin 8) := sorry

/-- The sum of labels at the ends of an edge -/
def edge_sum (c : LabeledCube) (e : Fin 8 × Fin 8) : ℕ :=
  c.vertices e.1 + c.vertices e.2

/-- Theorem: It's impossible to label a cube's vertices with 1 to 8 such that all edge sums are different -/
theorem cube_labeling_impossible : 
  ¬ ∃ (c : LabeledCube), (∀ v : Fin 8, c.vertices v ∈ Finset.range 9 \ {0}) ∧ 
    (∀ e₁ e₂ : Fin 8 × Fin 8, e₁ ∈ cube_edges → e₂ ∈ cube_edges → e₁ ≠ e₂ → 
      edge_sum c e₁ ≠ edge_sum c e₂) :=
sorry

end cube_labeling_impossible_l3015_301569


namespace ellipse_point_position_l3015_301516

theorem ellipse_point_position 
  (a b c : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_a_gt_b : a > b)
  (h_roots : x₁ + x₂ = -b/a ∧ x₁ * x₂ = -c/a) :
  1 < x₁^2 + x₂^2 ∧ x₁^2 + x₂^2 < 2 := by
sorry

end ellipse_point_position_l3015_301516


namespace sequence_general_term_l3015_301511

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 4 * a n - 3) →
  (∀ n : ℕ, n ≥ 1 → a n = (4/3)^(n-1)) :=
by sorry

end sequence_general_term_l3015_301511


namespace inverse_proportion_point_relation_l3015_301553

/-- Given two points A(x₁, 2) and B(x₂, 4) on the graph of y = k/x where k > 0,
    prove that x₁ > x₂ > 0 -/
theorem inverse_proportion_point_relation (k x₁ x₂ : ℝ) 
  (h_k : k > 0)
  (h_A : 2 = k / x₁)
  (h_B : 4 = k / x₂) :
  x₁ > x₂ ∧ x₂ > 0 :=
by sorry

end inverse_proportion_point_relation_l3015_301553


namespace only_345_is_right_triangle_l3015_301564

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- The theorem stating that among the given sets, only (3,4,5) forms a right triangle -/
theorem only_345_is_right_triangle :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) :=
by sorry

end only_345_is_right_triangle_l3015_301564


namespace shekars_math_marks_l3015_301566

def science_marks : ℝ := 65
def social_studies_marks : ℝ := 82
def english_marks : ℝ := 62
def biology_marks : ℝ := 85
def average_marks : ℝ := 74
def number_of_subjects : ℕ := 5

theorem shekars_math_marks :
  ∃ (math_marks : ℝ),
    (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / number_of_subjects = average_marks ∧
    math_marks = 76 := by
  sorry

end shekars_math_marks_l3015_301566


namespace determinant_equals_zy_l3015_301536

def matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  !![1, x, z; 1, x+z, z; 1, y, y+z]

theorem determinant_equals_zy (x y z : ℝ) : 
  Matrix.det (matrix x y z) = z * y := by sorry

end determinant_equals_zy_l3015_301536


namespace yellow_ball_count_l3015_301596

theorem yellow_ball_count (red yellow green : ℕ) : 
  red + yellow + green = 68 →
  yellow = 2 * red →
  3 * green = 4 * yellow →
  yellow = 24 := by
sorry

end yellow_ball_count_l3015_301596


namespace floor_minus_self_unique_solution_l3015_301521

theorem floor_minus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ - s = -10.3 :=
by
  -- The proof would go here
  sorry

end floor_minus_self_unique_solution_l3015_301521


namespace campbell_geometry_qualification_l3015_301547

/-- Represents the minimum score required in the 4th quarter to achieve a given average -/
def min_fourth_quarter_score (q1 q2 q3 : ℚ) (required_avg : ℚ) : ℚ :=
  4 * required_avg - (q1 + q2 + q3)

/-- Theorem: Given Campbell's scores and the required average, the minimum 4th quarter score is 107% -/
theorem campbell_geometry_qualification (campbell_q1 campbell_q2 campbell_q3 : ℚ)
  (h1 : campbell_q1 = 84/100)
  (h2 : campbell_q2 = 79/100)
  (h3 : campbell_q3 = 70/100)
  (required_avg : ℚ)
  (h4 : required_avg = 85/100) :
  min_fourth_quarter_score campbell_q1 campbell_q2 campbell_q3 required_avg = 107/100 := by
sorry

#eval min_fourth_quarter_score (84/100) (79/100) (70/100) (85/100)

end campbell_geometry_qualification_l3015_301547


namespace trigonometric_equation_solutions_l3015_301570

theorem trigonometric_equation_solutions :
  ∃ (S : Finset ℝ), 
    (∀ X ∈ S, 0 < X ∧ X < 2 * Real.pi) ∧
    (∀ X ∈ S, 1 + 2 * Real.sin X - 4 * (Real.sin X)^2 - 8 * (Real.sin X)^3 = 0) ∧
    S.card = 4 ∧
    (∀ Y, 0 < Y ∧ Y < 2 * Real.pi → 
      (1 + 2 * Real.sin Y - 4 * (Real.sin Y)^2 - 8 * (Real.sin Y)^3 = 0) → 
      Y ∈ S) := by
  sorry

end trigonometric_equation_solutions_l3015_301570


namespace angle_B_measure_l3015_301532

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 1, b = 2cos(C), and sin(C)cos(A) - sin(π/4 - B)sin(π/4 + B) = 0,
    then B = π/6. -/
theorem angle_B_measure (A B C : Real) (a b c : Real) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a = 1 →
  b = 2 * Real.cos C →
  Real.sin C * Real.cos A - Real.sin (π/4 - B) * Real.sin (π/4 + B) = 0 →
  B = π/6 := by
  sorry

end angle_B_measure_l3015_301532


namespace perpendicular_lines_sin_2alpha_l3015_301515

theorem perpendicular_lines_sin_2alpha (α : Real) :
  let l₁ : Real → Real → Real := λ x y => x * Real.sin α + y - 1
  let l₂ : Real → Real → Real := λ x y => x - 3 * y * Real.cos α + 1
  (∀ x y, l₁ x y = 0 → l₂ x y = 0 → (Real.sin α + 3 * Real.cos α) * (Real.sin α - 3 * Real.cos α) = 0) →
  Real.sin (2 * α) = 3 / 5 := by
sorry

end perpendicular_lines_sin_2alpha_l3015_301515


namespace price_change_after_markup_and_markdown_l3015_301518

theorem price_change_after_markup_and_markdown (original_price : ℝ) (markup_percent : ℝ) (markdown_percent : ℝ)
  (h_original_positive : original_price > 0)
  (h_markup : markup_percent = 10)
  (h_markdown : markdown_percent = 10) :
  original_price * (1 + markup_percent / 100) * (1 - markdown_percent / 100) < original_price :=
by sorry

end price_change_after_markup_and_markdown_l3015_301518


namespace tan_negative_405_degrees_l3015_301517

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by sorry

end tan_negative_405_degrees_l3015_301517


namespace multiple_decimals_between_7_5_and_9_5_l3015_301598

theorem multiple_decimals_between_7_5_and_9_5 : 
  ∃ (x y : ℝ), 7.5 < x ∧ x < y ∧ y < 9.5 :=
sorry

end multiple_decimals_between_7_5_and_9_5_l3015_301598


namespace compute_fraction_power_l3015_301528

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end compute_fraction_power_l3015_301528


namespace cos_seven_pi_sixths_l3015_301586

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seven_pi_sixths_l3015_301586


namespace logarithm_product_equality_logarithm_expression_equality_l3015_301527

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the lg function (log base 10)
noncomputable def lg (x : ℝ) : ℝ := log 10 x

theorem logarithm_product_equality : 
  log 2 25 * log 3 4 * log 5 9 = 8 := by sorry

theorem logarithm_expression_equality :
  1/2 * lg (32/49) - 4/3 * lg (Real.sqrt 8) + lg (Real.sqrt 245) = 1/2 := by sorry

end logarithm_product_equality_logarithm_expression_equality_l3015_301527


namespace tank_capacity_l3015_301507

theorem tank_capacity : ℝ → Prop :=
  fun capacity =>
    let initial_fraction : ℚ := 1/4
    let final_fraction : ℚ := 3/4
    let added_water : ℝ := 180
    initial_fraction * capacity + added_water = final_fraction * capacity →
    capacity = 360

-- Proof
example : tank_capacity 360 := by
  sorry

end tank_capacity_l3015_301507


namespace solve_gumball_problem_l3015_301568

def gumball_problem (alicia_gumballs : ℕ) (remaining_gumballs : ℕ) : Prop :=
  let pedro_gumballs := alicia_gumballs + 3 * alicia_gumballs
  let total_gumballs := alicia_gumballs + pedro_gumballs
  let taken_gumballs := total_gumballs - remaining_gumballs
  (taken_gumballs : ℚ) / (total_gumballs : ℚ) = 2/5

theorem solve_gumball_problem :
  gumball_problem 20 60 := by sorry

end solve_gumball_problem_l3015_301568


namespace division_in_ratio_l3015_301565

theorem division_in_ratio (total : ℕ) (ratio_b ratio_c : ℕ) (amount_c : ℕ) : 
  total = 2000 →
  ratio_b = 4 →
  ratio_c = 16 →
  amount_c = total * ratio_c / (ratio_b + ratio_c) →
  amount_c = 1600 := by
sorry

end division_in_ratio_l3015_301565


namespace final_movie_length_l3015_301522

def original_length : ℕ := 60
def removed_scenes : List ℕ := [8, 3, 4, 2, 6]

theorem final_movie_length :
  original_length - (removed_scenes.sum) = 37 := by
  sorry

end final_movie_length_l3015_301522


namespace lily_book_count_l3015_301556

/-- The number of books Lily read last month -/
def last_month_books : ℕ := 4

/-- The number of books Lily plans to read this month -/
def this_month_books : ℕ := 2 * last_month_books

/-- The total number of books Lily will read over two months -/
def total_books : ℕ := last_month_books + this_month_books

theorem lily_book_count : total_books = 12 := by
  sorry

end lily_book_count_l3015_301556


namespace unique_prime_solution_l3015_301589

theorem unique_prime_solution :
  ∀ (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end unique_prime_solution_l3015_301589


namespace largest_integer_solution_of_inequalities_l3015_301505

theorem largest_integer_solution_of_inequalities :
  ∀ x : ℤ, (x - 3 * (x - 2) ≥ 4 ∧ 2 * x + 1 < x - 1) → x ≤ -3 :=
by
  sorry

end largest_integer_solution_of_inequalities_l3015_301505


namespace total_packs_eq_243_l3015_301559

/-- The total number of packs sold in six villages -/
def total_packs : ℕ := 23 + 28 + 35 + 43 + 50 + 64

/-- Theorem stating that the total number of packs sold equals 243 -/
theorem total_packs_eq_243 : total_packs = 243 := by
  sorry

end total_packs_eq_243_l3015_301559


namespace parallel_segments_k_value_l3015_301501

/-- Given four points on a Cartesian plane, prove that if segment AB is parallel to segment XY, then k = -6 -/
theorem parallel_segments_k_value 
  (A B X Y : ℝ × ℝ) 
  (hA : A = (-4, 0)) 
  (hB : B = (0, -4)) 
  (hX : X = (0, 8)) 
  (hY : Y = (14, k))
  (h_parallel : (B.1 - A.1) * (Y.2 - X.2) = (B.2 - A.2) * (Y.1 - X.1)) : 
  k = -6 := by
  sorry

end parallel_segments_k_value_l3015_301501


namespace quadratic_roots_conditions_l3015_301523

/-- The quadratic equation (2m+1)x^2 + 4mx + 2m-3 = 0 has:
    1. Two distinct real roots iff m ∈ (-3/4, -1/2) ∪ (-1/2, ∞)
    2. Two equal real roots iff m = -3/4
    3. No real roots iff m ∈ (-∞, -3/4) -/
theorem quadratic_roots_conditions (m : ℝ) :
  let a := 2*m + 1
  let b := 4*m
  let c := 2*m - 3
  let discriminant := b^2 - 4*a*c
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) ↔ 
    (m > -3/4 ∧ m ≠ -1/2) ∧
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ discriminant = 0) ↔ 
    (m = -3/4) ∧
  (∀ x : ℝ, a*x^2 + b*x + c ≠ 0) ↔ 
    (m < -3/4) :=
by sorry

end quadratic_roots_conditions_l3015_301523


namespace lcm_of_ratio_and_hcf_l3015_301555

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → Nat.gcd a b = 4 → Nat.lcm a b = 48 := by
  sorry

end lcm_of_ratio_and_hcf_l3015_301555


namespace fraction_power_product_l3015_301573

theorem fraction_power_product :
  (3 / 5 : ℚ) ^ 4 * (2 / 9 : ℚ) = 162 / 5625 := by sorry

end fraction_power_product_l3015_301573


namespace unique_four_digit_number_with_geometric_property_l3015_301581

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ :=
  n / 1000

def second_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def third_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def fourth_digit (n : ℕ) : ℕ :=
  n % 10

def ab (n : ℕ) : ℕ :=
  10 * (first_digit n) + (second_digit n)

def bc (n : ℕ) : ℕ :=
  10 * (second_digit n) + (third_digit n)

def cd (n : ℕ) : ℕ :=
  10 * (third_digit n) + (fourth_digit n)

def is_increasing_geometric_sequence (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ y * y = x * z

theorem unique_four_digit_number_with_geometric_property :
  ∃! n : ℕ, is_valid_four_digit_number n ∧
             first_digit n ≠ 0 ∧
             is_increasing_geometric_sequence (ab n) (bc n) (cd n) :=
by sorry

end unique_four_digit_number_with_geometric_property_l3015_301581


namespace point_belongs_to_transformed_plane_l3015_301579

/-- Plane equation coefficients -/
structure PlaneCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Apply similarity transformation to plane equation -/
def transformPlane (p : PlaneCoefficients) (k : ℝ) : PlaneCoefficients :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Check if a point satisfies a plane equation -/
def satisfiesPlane (point : Point3D) (plane : PlaneCoefficients) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Main theorem: Point A belongs to the image of plane a after similarity transformation -/
theorem point_belongs_to_transformed_plane 
  (A : Point3D) 
  (a : PlaneCoefficients) 
  (k : ℝ) 
  (h1 : A.x = 1/2) 
  (h2 : A.y = 1/3) 
  (h3 : A.z = 1) 
  (h4 : a.a = 2) 
  (h5 : a.b = -3) 
  (h6 : a.c = 3) 
  (h7 : a.d = -2) 
  (h8 : k = 1.5) : 
  satisfiesPlane A (transformPlane a k) :=
sorry

end point_belongs_to_transformed_plane_l3015_301579


namespace barometric_pressure_proof_l3015_301546

/-- Represents the combined gas law equation -/
def combined_gas_law (p1 v1 T1 p2 v2 T2 : ℝ) : Prop :=
  p1 * v1 / T1 = p2 * v2 / T2

/-- Calculates the absolute temperature from Celsius -/
def absolute_temp (celsius : ℝ) : ℝ := celsius + 273

theorem barometric_pressure_proof 
  (well_functioning_pressure : ℝ) 
  (faulty_pressure_15C : ℝ) 
  (faulty_pressure_30C : ℝ) 
  (air_free_space : ℝ) :
  well_functioning_pressure = 762 →
  faulty_pressure_15C = 704 →
  faulty_pressure_30C = 692 →
  air_free_space = 143 →
  ∃ (true_pressure : ℝ),
    true_pressure = 748 ∧
    combined_gas_law 
      (well_functioning_pressure - faulty_pressure_15C) 
      air_free_space 
      (absolute_temp 15)
      (true_pressure - faulty_pressure_30C) 
      (air_free_space + (faulty_pressure_15C - faulty_pressure_30C)) 
      (absolute_temp 30) :=
by sorry

end barometric_pressure_proof_l3015_301546


namespace outfit_combinations_l3015_301534

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (hats : ℕ) : 
  shirts = 4 → pants = 5 → hats = 3 → shirts * pants * hats = 60 := by
  sorry

end outfit_combinations_l3015_301534


namespace quadratic_factorization_l3015_301533

theorem quadratic_factorization (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 14 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 30 = (x + b) * (x - c)) →
  a + b + c = 15 :=
by sorry

end quadratic_factorization_l3015_301533


namespace special_ellipse_properties_l3015_301580

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  minor_axis_length : b = Real.sqrt 3
  foci_triangle : ∃ (c : ℝ), a = 2 * c ∧ a^2 = b^2 + c^2

/-- The point P -/
def P : ℝ × ℝ := (0, 2)

/-- Theorem about the special ellipse and its properties -/
theorem special_ellipse_properties (E : SpecialEllipse) :
  -- 1. Standard equation
  E.a^2 = 4 ∧ E.b^2 = 3 ∧
  -- 2. Existence of line l
  ∃ (k : ℝ), 
    -- 3. Equation of line l
    (k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2) ∧
    -- Line passes through P and intersects the ellipse at two distinct points
    ∃ (M N : ℝ × ℝ), M ≠ N ∧
      M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 ∧
      N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 ∧
      M.2 = k * M.1 + P.2 ∧
      N.2 = k * N.1 + P.2 ∧
      -- Satisfying the dot product condition
      M.1 * N.1 + M.2 * N.2 = 2 := by
  sorry

end special_ellipse_properties_l3015_301580


namespace sqrt_four_equals_two_l3015_301576

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end sqrt_four_equals_two_l3015_301576


namespace paint_house_theorem_l3015_301590

/-- Represents the time (in hours) it takes to paint a house given the number of people working -/
def paintTime (people : ℕ) : ℚ :=
  24 / people

theorem paint_house_theorem :
  paintTime 4 = 6 →
  paintTime 3 = 8 :=
by
  sorry

end paint_house_theorem_l3015_301590


namespace football_kick_distance_l3015_301519

theorem football_kick_distance (longest_kick : ℝ) (average_kick : ℝ) (kick1 kick2 kick3 : ℝ) :
  longest_kick = 43 →
  average_kick = 37 →
  (kick1 + kick2 + kick3) / 3 = average_kick →
  kick1 = longest_kick →
  kick2 = kick3 →
  kick2 = 34 ∧ kick3 = 34 := by
  sorry

end football_kick_distance_l3015_301519


namespace intersection_area_of_specific_rectangles_l3015_301563

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ
  angle : ℝ  -- Angle of rotation in radians

/-- Calculates the area of intersection between two rectangles -/
noncomputable def intersectionArea (r1 r2 : Rectangle) : ℝ :=
  sorry

/-- Theorem stating the area of intersection between two specific rectangles -/
theorem intersection_area_of_specific_rectangles :
  let r1 : Rectangle := { width := 4, height := 12, angle := 0 }
  let r2 : Rectangle := { width := 5, height := 10, angle := π/6 }
  intersectionArea r1 r2 = 24 := by
  sorry

end intersection_area_of_specific_rectangles_l3015_301563


namespace division_remainder_l3015_301594

theorem division_remainder (n : ℕ) : 
  (n / 8 = 8 ∧ n % 8 = 0) → n % 5 = 4 := by
sorry

end division_remainder_l3015_301594


namespace cos_150_degrees_l3015_301599

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end cos_150_degrees_l3015_301599


namespace greater_fourteen_game_count_l3015_301545

/-- Represents a basketball league with two divisions -/
structure BasketballLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of scheduled games in the league -/
def total_games (league : BasketballLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games +
                        league.teams_per_division * league.inter_division_games
  total_teams * games_per_team / 2

/-- The Greater Fourteen Basketball League -/
def greater_fourteen : BasketballLeague :=
  { divisions := 2,
    teams_per_division := 7,
    intra_division_games := 2,
    inter_division_games := 2 }

theorem greater_fourteen_game_count :
  total_games greater_fourteen = 182 := by
  sorry

end greater_fourteen_game_count_l3015_301545


namespace maria_spent_60_dollars_l3015_301510

def flower_cost : ℕ := 6
def roses_bought : ℕ := 7
def daisies_bought : ℕ := 3

theorem maria_spent_60_dollars : 
  (roses_bought + daisies_bought) * flower_cost = 60 := by
  sorry

end maria_spent_60_dollars_l3015_301510


namespace range_of_a_l3015_301506

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3 - a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ 0) → a ∈ Set.Icc (-7 : ℝ) 2 := by
  sorry

end range_of_a_l3015_301506


namespace inequality_proof_l3015_301540

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  24 * x * y * z ≤ 3 * (x + y) * (y + z) * (z + x) ∧ 
  3 * (x + y) * (y + z) * (z + x) ≤ 8 * (x^3 + y^3 + z^3) := by
  sorry

end inequality_proof_l3015_301540


namespace integer_pair_solution_l3015_301514

theorem integer_pair_solution :
  ∀ x y : ℕ+,
  (2 * x.val * y.val = 2 * x.val + y.val + 21) →
  ((x.val = 1 ∧ y.val = 23) ∨ (x.val = 6 ∧ y.val = 3)) :=
by
  sorry

end integer_pair_solution_l3015_301514


namespace simplify_fraction_l3015_301538

theorem simplify_fraction (a b : ℝ) 
  (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end simplify_fraction_l3015_301538


namespace min_value_sum_of_reciprocals_l3015_301509

theorem min_value_sum_of_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (1 + a^n) + 1 / (1 + b^n) ≥ 1 ∧
  (1 / (1 + a^n) + 1 / (1 + b^n) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_value_sum_of_reciprocals_l3015_301509


namespace person_height_from_shadow_l3015_301585

/-- Given a tree and a person under the same light conditions, calculate the person's height -/
theorem person_height_from_shadow (tree_height tree_shadow person_shadow : ℝ) 
  (h1 : tree_height = 60)
  (h2 : tree_shadow = 18)
  (h3 : person_shadow = 3) :
  (tree_height / tree_shadow) * person_shadow = 10 := by
  sorry

end person_height_from_shadow_l3015_301585


namespace remainder_sum_l3015_301535

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end remainder_sum_l3015_301535


namespace shaded_area_is_24_5_l3015_301542

/-- Represents the structure of the grid --/
structure Grid :=
  (rect1 : Int × Int)
  (rect2 : Int × Int)
  (rect3 : Int × Int)

/-- Calculates the area of a rectangle --/
def rectangleArea (dims : Int × Int) : Int :=
  dims.1 * dims.2

/-- Calculates the total area of the grid --/
def totalGridArea (g : Grid) : Int :=
  rectangleArea g.rect1 + rectangleArea g.rect2 + rectangleArea g.rect3

/-- Calculates the area of a right-angled triangle --/
def triangleArea (base height : Int) : Rat :=
  (base * height) / 2

/-- The main theorem stating the area of the shaded region --/
theorem shaded_area_is_24_5 (g : Grid) 
    (h1 : g.rect1 = (3, 4))
    (h2 : g.rect2 = (4, 5))
    (h3 : g.rect3 = (5, 6))
    (h4 : totalGridArea g = 62)
    (h5 : triangleArea 15 5 = 37.5) :
  totalGridArea g - triangleArea 15 5 = 24.5 := by
  sorry


end shaded_area_is_24_5_l3015_301542


namespace rice_mixture_cost_l3015_301524

/-- Proves that mixing two varieties of rice in a given ratio results in the specified cost per kg -/
theorem rice_mixture_cost 
  (cost1 : ℝ) 
  (cost2 : ℝ) 
  (ratio : ℝ) 
  (mixture_cost : ℝ) 
  (h1 : cost1 = 7) 
  (h2 : cost2 = 8.75) 
  (h3 : ratio = 2.5) 
  (h4 : mixture_cost = 7.5) : 
  (ratio * cost1 + cost2) / (ratio + 1) = mixture_cost := by
  sorry

end rice_mixture_cost_l3015_301524


namespace rationalize_denominator_l3015_301552

theorem rationalize_denominator : 
  1 / (Real.sqrt 3 - 2) = -(Real.sqrt 3) - 2 := by sorry

end rationalize_denominator_l3015_301552


namespace suzy_final_book_count_l3015_301577

/-- Calculates the final number of books Suzy has after a series of transactions -/
def final_book_count (initial_books : ℕ) 
                     (wed_checkout : ℕ) 
                     (thu_return thu_checkout : ℕ) 
                     (fri_return : ℕ) : ℕ :=
  initial_books - wed_checkout + thu_return - thu_checkout + fri_return

/-- Theorem stating that Suzy ends up with 80 books given the specific transactions -/
theorem suzy_final_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

end suzy_final_book_count_l3015_301577


namespace max_sum_coordinates_l3015_301560

/-- Triangle DEF in the cartesian plane with the following properties:
  - Area of triangle DEF is 65
  - Coordinates of D are (10, 15)
  - Coordinates of F are (19, 18)
  - Coordinates of E are (r, s)
  - The line containing the median to side DF has slope -3
-/
def TriangleDEF (r s : ℝ) : Prop :=
  let d := (10, 15)
  let f := (19, 18)
  let e := (r, s)
  let area := 65
  let median_slope := -3
  -- Area condition
  area = (1/2) * abs (r * 15 + 10 * 18 + 19 * s - s * 10 - 15 * 19 - r * 18) ∧
  -- Median slope condition
  median_slope = (s - (33/2)) / (r - (29/2))

theorem max_sum_coordinates (r s : ℝ) :
  TriangleDEF r s → r + s ≤ 1454/15 := by
  sorry

end max_sum_coordinates_l3015_301560


namespace darla_total_cost_l3015_301500

/-- The total cost of electricity given the rate, usage, and late fee. -/
def total_cost (rate : ℝ) (usage : ℝ) (late_fee : ℝ) : ℝ :=
  rate * usage + late_fee

/-- Proof that Darla's total cost is $1350 -/
theorem darla_total_cost : 
  total_cost 4 300 150 = 1350 := by
  sorry

end darla_total_cost_l3015_301500


namespace binomial_identities_l3015_301597

theorem binomial_identities (n k : ℕ) : 
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) ∧ 
  (Finset.range (n + 1)).sum (λ k => k * (n.choose k)) = n * 2^(n - 1) := by
  sorry

end binomial_identities_l3015_301597


namespace inscribed_quadrilateral_fourth_side_l3015_301550

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ

/-- The theorem stating the property of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 100 * Real.sqrt 3)
  (h_side1 : q.sides 0 = 100)
  (h_side2 : q.sides 1 = 200)
  (h_side3 : q.sides 2 = 300) :
  q.sides 3 = 450 := by
  sorry

end inscribed_quadrilateral_fourth_side_l3015_301550


namespace jester_win_prob_constant_l3015_301584

/-- The probability of the Jester winning in a game with 2n-1 regular townspeople, one Jester, and one goon -/
def jester_win_probability (n : ℕ+) : ℚ :=
  1 / 3

/-- The game ends immediately if the Jester is sent to jail during the morning -/
axiom morning_jail_win (n : ℕ+) : 
  jester_win_probability n = 1 / (2 * n + 1) + 
    ((2 * n - 1) / (2 * n + 1)) * ((2 * n - 2) / (2 * n - 1)) * jester_win_probability (n - 1)

/-- The Jester does not win if sent to jail at night -/
axiom night_jail_no_win (n : ℕ+) :
  jester_win_probability n = 
    ((2 * n - 1) / (2 * n + 1)) * ((2 * n - 2) / (2 * n - 1)) * jester_win_probability (n - 1)

theorem jester_win_prob_constant (n : ℕ+) : 
  jester_win_probability n = 1 / 3 := by
  sorry

end jester_win_prob_constant_l3015_301584


namespace school_trip_theorem_l3015_301567

/-- The number of school buses -/
def num_buses : ℕ := 95

/-- The number of seats in each school bus -/
def seats_per_bus : ℕ := 118

/-- The number of students in the school -/
def num_students : ℕ := num_buses * seats_per_bus

theorem school_trip_theorem : num_students = 11210 := by
  sorry

end school_trip_theorem_l3015_301567


namespace center_of_specific_circle_l3015_301529

/-- The center coordinates of a circle given its equation -/
def circle_center (a b r : ℝ) : ℝ × ℝ := (a, -b)

/-- Theorem: The center coordinates of the circle (x-2)^2 + (y+1)^2 = 4 are (2, -1) -/
theorem center_of_specific_circle :
  circle_center 2 (-1) 2 = (2, -1) := by sorry

end center_of_specific_circle_l3015_301529


namespace polynomial_sum_of_coefficients_l3015_301593

theorem polynomial_sum_of_coefficients 
  (a b c d : ℝ) 
  (g : ℂ → ℂ) 
  (h₁ : ∀ x, g x = x^4 + a*x^3 + b*x^2 + c*x + d) 
  (h₂ : g (-3*I) = 0) 
  (h₃ : g (1 + I) = 0) : 
  a + b + c + d = 9 := by
sorry

end polynomial_sum_of_coefficients_l3015_301593


namespace sqrt_equation_solution_l3015_301561

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x ∧ 
  x = 729/144 := by sorry

end sqrt_equation_solution_l3015_301561


namespace cubic_equation_result_l3015_301578

theorem cubic_equation_result (x : ℝ) (h : x^3 + 4*x^2 = 8) :
  x^5 + 80*x^3 = -376*x^2 - 32*x + 768 := by
  sorry

end cubic_equation_result_l3015_301578


namespace no_hexagon_tiling_l3015_301539

-- Define a grid hexagon
structure GridHexagon where
  -- Add necessary fields to define the hexagon
  -- This is a placeholder and should be adjusted based on the specific hexagon properties
  side_length : ℝ
  diagonal_length : ℝ

-- Define a grid rectangle
structure GridRectangle where
  width : ℕ
  height : ℕ

-- Define the tiling property
def can_tile (r : GridRectangle) (h : GridHexagon) : Prop :=
  -- This is a placeholder for the actual tiling condition
  -- It should represent that the rectangle can be tiled with the hexagons
  sorry

-- The main theorem
theorem no_hexagon_tiling (r : GridRectangle) (h : GridHexagon) : 
  ¬(can_tile r h) := by
  sorry

end no_hexagon_tiling_l3015_301539


namespace first_number_proof_l3015_301574

theorem first_number_proof (N : ℕ) : 
  (∃ k m : ℕ, N = 170 * k + 10 ∧ 875 = 170 * m + 25) →
  N = 860 := by
  sorry

end first_number_proof_l3015_301574


namespace expression_evaluation_l3015_301588

theorem expression_evaluation : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end expression_evaluation_l3015_301588


namespace unique_n_satisfying_equation_l3015_301587

theorem unique_n_satisfying_equation : 
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 9⌋ - ⌊(n : ℚ) / 3⌋^2 = 5 ∧ n = 14 :=
by sorry

end unique_n_satisfying_equation_l3015_301587


namespace triangle_perimeter_l3015_301575

theorem triangle_perimeter : ∀ x : ℝ, 
  (x - 2) * (x - 4) = 0 →
  x + 3 > 6 →
  x + 3 + 6 = 13 :=
by
  sorry

end triangle_perimeter_l3015_301575


namespace problem_solution_l3015_301551

theorem problem_solution (x y : ℚ) : 
  x / y = 12 / 5 → y = 25 → x = 60 := by
  sorry

end problem_solution_l3015_301551


namespace collinearity_condition_l3015_301544

/-- Three points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Collinearity condition for three points in a 2D plane -/
def collinear (A B C : Point2D) : Prop :=
  A.x * B.y + B.x * C.y + C.x * A.y = A.y * B.x + B.y * C.x + C.y * A.x

/-- Theorem: Three points are collinear iff they satisfy the collinearity condition -/
theorem collinearity_condition (A B C : Point2D) :
  collinear A B C ↔ A.x * B.y + B.x * C.y + C.x * A.y = A.y * B.x + B.y * C.x + C.y * A.x :=
sorry

end collinearity_condition_l3015_301544


namespace sqrt_expression_simplification_l3015_301592

theorem sqrt_expression_simplification :
  let x := Real.sqrt 97
  let y := Real.sqrt 486
  let z := Real.sqrt 125
  let w := Real.sqrt 54
  let v := Real.sqrt 49
  (x + y + z) / (w + v) = (x + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7) := by
  sorry

end sqrt_expression_simplification_l3015_301592


namespace f_composition_half_f_composition_eq_one_solutions_l3015_301531

noncomputable section

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by sorry

theorem f_composition_eq_one_solutions : 
  {x : ℝ | f (f x) = 1} = {1, Real.exp (Real.exp 1)} := by sorry

end f_composition_half_f_composition_eq_one_solutions_l3015_301531


namespace intersection_M_N_l3015_301520

def M : Set ℝ := {x : ℝ | x^2 + 3*x = 0}
def N : Set ℝ := {3, 0}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end intersection_M_N_l3015_301520


namespace four_integers_product_2002_sum_less_40_l3015_301541

theorem four_integers_product_2002_sum_less_40 :
  ∀ (a b c d : ℕ+),
    a * b * c * d = 2002 →
    (a : ℕ) + b + c + d < 40 →
    ((a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨
     (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
     (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨
     (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
     (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨
     (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
     (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨
     (a = 1 ∧ b = 13 ∧ c = 14 ∧ d = 11) ∨
     (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨
     (a = 1 ∧ b = 13 ∧ c = 11 ∧ d = 14) ∨
     (a = 7 ∧ b = 2 ∧ c = 11 ∧ d = 13) ∨
     (a = 7 ∧ b = 11 ∧ c = 2 ∧ d = 13) ∨
     (a = 7 ∧ b = 11 ∧ c = 13 ∧ d = 2) ∨
     (a = 11 ∧ b = 2 ∧ c = 7 ∧ d = 13) ∨
     (a = 11 ∧ b = 7 ∧ c = 2 ∧ d = 13) ∨
     (a = 11 ∧ b = 7 ∧ c = 13 ∧ d = 2) ∨
     (a = 11 ∧ b = 13 ∧ c = 2 ∧ d = 7) ∨
     (a = 11 ∧ b = 13 ∧ c = 7 ∧ d = 2) ∨
     (a = 13 ∧ b = 2 ∧ c = 7 ∧ d = 11) ∨
     (a = 13 ∧ b = 7 ∧ c = 2 ∧ d = 11) ∨
     (a = 13 ∧ b = 7 ∧ c = 11 ∧ d = 2) ∨
     (a = 13 ∧ b = 11 ∧ c = 2 ∧ d = 7) ∨
     (a = 13 ∧ b = 11 ∧ c = 7 ∧ d = 2) ∨
     (a = 14 ∧ b = 1 ∧ c = 11 ∧ d = 13) ∨
     (a = 14 ∧ b = 11 ∧ c = 1 ∧ d = 13) ∨
     (a = 14 ∧ b = 11 ∧ c = 13 ∧ d = 1) ∨
     (a = 11 ∧ b = 1 ∧ c = 14 ∧ d = 13) ∨
     (a = 11 ∧ b = 14 ∧ c = 1 ∧ d = 13) ∨
     (a = 11 ∧ b = 14 ∧ c = 13 ∧ d = 1) ∨
     (a = 11 ∧ b = 13 ∧ c = 1 ∧ d = 14) ∨
     (a = 11 ∧ b = 13 ∧ c = 14 ∧ d = 1) ∨
     (a = 13 ∧ b = 1 ∧ c = 14 ∧ d = 11) ∨
     (a = 13 ∧ b = 14 ∧ c = 1 ∧ d = 11) ∨
     (a = 13 ∧ b = 14 ∧ c = 11 ∧ d = 1) ∨
     (a = 13 ∧ b = 11 ∧ c = 1 ∧ d = 14) ∨
     (a = 13 ∧ b = 11 ∧ c = 14 ∧ d = 1)) :=
by sorry

end four_integers_product_2002_sum_less_40_l3015_301541


namespace tuesday_rain_amount_l3015_301502

/-- The amount of rain on Monday in inches -/
def monday_rain : ℝ := 0.9

/-- The difference in rain between Monday and Tuesday in inches -/
def rain_difference : ℝ := 0.7

/-- The amount of rain on Tuesday in inches -/
def tuesday_rain : ℝ := monday_rain - rain_difference

theorem tuesday_rain_amount : tuesday_rain = 0.2 := by
  sorry

end tuesday_rain_amount_l3015_301502


namespace total_canoes_by_april_l3015_301582

def canoes_built (month : Nat) : Nat :=
  match month with
  | 0 => 5  -- February (0-indexed)
  | n + 1 => 3 * canoes_built n

theorem total_canoes_by_april : 
  canoes_built 0 + canoes_built 1 + canoes_built 2 = 65 := by
  sorry

end total_canoes_by_april_l3015_301582


namespace intersection_and_lines_l3015_301583

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l₃ (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

theorem intersection_and_lines :
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y) →
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) →
  (∀ (x y : ℝ), parallel_line x y ↔ (∃ (t : ℝ), x = P.1 + t ∧ y = P.2 + t * (-2))) ∧
  (∀ (x y : ℝ), perpendicular_line x y ↔ (∃ (t : ℝ), x = P.1 + t ∧ y = P.2 + t * (1/2))) :=
sorry

end intersection_and_lines_l3015_301583


namespace cake_slices_kept_l3015_301595

theorem cake_slices_kept (total_slices : ℕ) (eaten_fraction : ℚ) (extra_eaten : ℕ) : 
  total_slices = 35 →
  eaten_fraction = 2/5 →
  extra_eaten = 3 →
  total_slices - (eaten_fraction * total_slices + extra_eaten) = 18 :=
by sorry

end cake_slices_kept_l3015_301595


namespace rook_placement_l3015_301526

/-- The number of ways to place 3 rooks on a 6 × 2006 chessboard such that they don't attack each other -/
def placeRooks : ℕ := 20 * 2006 * 2005 * 2004

/-- The width of the chessboard -/
def boardWidth : ℕ := 6

/-- The height of the chessboard -/
def boardHeight : ℕ := 2006

/-- The number of rooks to be placed -/
def numRooks : ℕ := 3

theorem rook_placement :
  placeRooks = (Nat.choose boardWidth numRooks) * 
               boardHeight * (boardHeight - 1) * (boardHeight - 2) := by
  sorry

end rook_placement_l3015_301526


namespace not_all_countries_have_complete_systems_l3015_301530

/-- Represents a country's internet regulation system -/
structure InternetRegulation where
  country : String
  hasCompleteSystem : Bool

/-- Information about internet regulation systems in different countries -/
def countryRegulations : List InternetRegulation := [
  { country := "United States", hasCompleteSystem := false },
  { country := "United Kingdom", hasCompleteSystem := false },
  { country := "Russia", hasCompleteSystem := true }
]

/-- Theorem stating that not all countries (US, UK, and Russia) have complete internet regulation systems -/
theorem not_all_countries_have_complete_systems : 
  ¬ (∀ c ∈ countryRegulations, c.hasCompleteSystem = true) := by
  sorry

end not_all_countries_have_complete_systems_l3015_301530


namespace gain_percent_calculation_l3015_301503

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 40 * S) : 
  (S - C) / C * 100 = 25 := by
sorry

end gain_percent_calculation_l3015_301503


namespace sum_of_fractions_l3015_301549

theorem sum_of_fractions : 
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (-5 : ℚ) / 6 + (1 : ℚ) / 5 + (1 : ℚ) / 4 + (-9 : ℚ) / 20 + (-5 : ℚ) / 6 = (-5 : ℚ) / 6 := by
  sorry

end sum_of_fractions_l3015_301549


namespace michael_paid_594_l3015_301525

/-- The total amount Michael paid for his purchases after discounts -/
def total_paid (suit_price shoes_price shirt_price tie_price : ℕ) 
  (suit_discount shoes_discount shirt_tie_discount_percent : ℕ) : ℕ :=
  let suit_discounted := suit_price - suit_discount
  let shoes_discounted := shoes_price - shoes_discount
  let shirt_tie_total := shirt_price + tie_price
  let shirt_tie_discount := shirt_tie_total * shirt_tie_discount_percent / 100
  let shirt_tie_discounted := shirt_tie_total - shirt_tie_discount
  suit_discounted + shoes_discounted + shirt_tie_discounted

/-- Theorem stating that Michael paid $594 for his purchases -/
theorem michael_paid_594 :
  total_paid 430 190 80 50 100 30 20 = 594 := by sorry

end michael_paid_594_l3015_301525


namespace hyperbola_axis_ratio_l3015_301543

/-- Given a hyperbola with equation x² - my² = 1, where m is a real number,
    if the length of the conjugate axis is three times that of the transverse axis,
    then m = 1/9 -/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, x^2 - m*y^2 = 1) → 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*b = 3*(2*a) ∧ a^2 = 1 ∧ b^2 = 1/m) →
  m = 1/9 := by
sorry

end hyperbola_axis_ratio_l3015_301543


namespace discount_order_difference_discount_order_difference_proof_l3015_301571

/-- Proves that the difference between applying 25% off then $5 off, and applying $5 off then 25% off, on a $30 item, is 125 cents. -/
theorem discount_order_difference : ℝ → Prop :=
  fun original_price : ℝ =>
    let first_discount : ℝ := 5
    let second_discount_rate : ℝ := 0.25
    let price_25_then_5 := (original_price * (1 - second_discount_rate)) - first_discount
    let price_5_then_25 := (original_price - first_discount) * (1 - second_discount_rate)
    original_price = 30 →
    (price_25_then_5 - price_5_then_25) * 100 = 125

/-- The proof of the theorem. -/
theorem discount_order_difference_proof : discount_order_difference 30 := by
  sorry

end discount_order_difference_discount_order_difference_proof_l3015_301571


namespace fA_inter_fB_l3015_301548

def f (n : ℕ+) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def fA : Set ℕ+ := {n : ℕ+ | f n ∈ A}
def fB : Set ℕ+ := {m : ℕ+ | f m ∈ B}

theorem fA_inter_fB : fA ∩ fB = {1, 2} := by sorry

end fA_inter_fB_l3015_301548


namespace some_number_value_l3015_301537

theorem some_number_value (x : ℝ) : 65 + 5 * 12 / (180 / x) = 66 → x = 3 := by
  sorry

end some_number_value_l3015_301537


namespace hayden_ironing_days_l3015_301557

/-- Given that Hayden spends 8 minutes ironing clothes each day he does so,
    and over 4 weeks he spends 160 minutes ironing,
    prove that he irons his clothes 5 days per week. -/
theorem hayden_ironing_days (minutes_per_day : ℕ) (total_minutes : ℕ) (weeks : ℕ) :
  minutes_per_day = 8 →
  total_minutes = 160 →
  weeks = 4 →
  (total_minutes / weeks) / minutes_per_day = 5 :=
by sorry

end hayden_ironing_days_l3015_301557


namespace copy_pages_for_ten_dollars_l3015_301572

/-- The number of pages that can be copied for a given amount of money, 
    given the cost of copying 5 pages --/
def pages_copied (cost_5_pages : ℚ) (amount : ℚ) : ℚ :=
  (amount / cost_5_pages) * 5

/-- Theorem stating that given the cost of 10 cents for 5 pages, 
    the number of pages that can be copied for $10 is 500 --/
theorem copy_pages_for_ten_dollars :
  pages_copied (10 / 100) (10 : ℚ) = 500 := by
  sorry

#eval pages_copied (10 / 100) 10

end copy_pages_for_ten_dollars_l3015_301572


namespace pull_up_median_mode_l3015_301513

def pull_up_data : List ℕ := [6, 8, 7, 7, 8, 9, 8, 9]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem pull_up_median_mode :
  median pull_up_data = 8 ∧ mode pull_up_data = 8 := by sorry

end pull_up_median_mode_l3015_301513


namespace nested_expression_value_l3015_301591

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end nested_expression_value_l3015_301591


namespace inequality_relations_l3015_301558

theorem inequality_relations (a b : ℝ) (h : a > b) : 
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) := by
  sorry

end inequality_relations_l3015_301558


namespace unique_solution_l3015_301562

theorem unique_solution : ∃! x : ℤ, x^2 + 105 = (x - 20)^2 ∧ x = 7 := by
  sorry

end unique_solution_l3015_301562


namespace number_with_specific_totient_l3015_301508

theorem number_with_specific_totient (N : ℕ) (α β γ : ℕ) :
  N = 3^α * 5^β * 7^γ →
  Nat.totient N = 3600 →
  N = 7875 := by
sorry

end number_with_specific_totient_l3015_301508


namespace angle_sum_equality_l3015_301504

theorem angle_sum_equality (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (eq1 : 4 * (Real.cos a)^2 + 3 * (Real.sin b)^2 = 1)
  (eq2 : 4 * Real.sin (2*a) + 3 * Real.cos (2*b) = 0) :
  a + 2*b = π/2 := by
sorry

end angle_sum_equality_l3015_301504


namespace greatest_possible_average_speed_l3015_301554

/-- A number is a palindrome if it reads the same backward as forward -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The next palindrome after a given number -/
def nextPalindrome (n : ℕ) : ℕ := sorry

theorem greatest_possible_average_speed 
  (initial_reading : ℕ) 
  (drive_duration : ℝ) 
  (speed_limit : ℝ) 
  (h1 : isPalindrome initial_reading)
  (h2 : drive_duration = 2)
  (h3 : speed_limit = 65)
  (h4 : initial_reading = 12321) :
  let final_reading := nextPalindrome initial_reading
  let distance := final_reading - initial_reading
  let max_distance := drive_duration * speed_limit
  let average_speed := distance / drive_duration
  (distance ≤ max_distance ∧ isPalindrome final_reading) →
  average_speed ≤ 50 ∧ 
  ∃ (s : ℝ), s > 50 → 
    ¬∃ (d : ℕ), d > distance ∧ 
      d ≤ max_distance ∧ 
      isPalindrome (initial_reading + d) ∧
      s = d / drive_duration :=
by sorry

end greatest_possible_average_speed_l3015_301554


namespace unique_solution_of_equation_l3015_301512

theorem unique_solution_of_equation : 
  ∃! x : ℝ, (x^3 + 2*x^2) / (x^2 + 3*x + 2) + x = -6 ∧ x ≠ -2 :=
by sorry

end unique_solution_of_equation_l3015_301512
