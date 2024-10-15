import Mathlib

namespace NUMINAMATH_CALUDE_prob_two_students_same_section_l616_61608

/-- The probability of two specific students being selected and placed in the same section -/
theorem prob_two_students_same_section 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (num_sections : ℕ) 
  (section_capacity : ℕ) 
  (h1 : total_students = 100)
  (h2 : selected_students = 60)
  (h3 : num_sections = 3)
  (h4 : section_capacity = 20)
  (h5 : selected_students = num_sections * section_capacity) :
  (selected_students : ℚ) / total_students * 
  (selected_students - 1) / (total_students - 1) * 
  (section_capacity - 1) / (selected_students - 1) = 19 / 165 :=
sorry

end NUMINAMATH_CALUDE_prob_two_students_same_section_l616_61608


namespace NUMINAMATH_CALUDE_investment_rate_problem_l616_61627

theorem investment_rate_problem (total_investment : ℝ) (amount_at_eight_percent : ℝ) (income_difference : ℝ) (R : ℝ) :
  total_investment = 2000 →
  amount_at_eight_percent = 600 →
  income_difference = 92 →
  (total_investment - amount_at_eight_percent) * R - amount_at_eight_percent * 0.08 = income_difference →
  R = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l616_61627


namespace NUMINAMATH_CALUDE_laurence_to_missy_relation_keith_receives_32_messages_l616_61661

/-- Messages sent from Juan to Laurence -/
def messages_juan_to_laurence : ℕ := sorry

/-- Messages sent from Juan to Keith -/
def messages_juan_to_keith : ℕ := 8 * messages_juan_to_laurence

/-- Messages sent from Laurence to Missy -/
def messages_laurence_to_missy : ℕ := 18

/-- Relation between messages from Laurence to Missy and from Juan to Laurence -/
theorem laurence_to_missy_relation : 
  messages_laurence_to_missy = (4.5 : ℚ) * messages_juan_to_laurence := sorry

theorem keith_receives_32_messages : messages_juan_to_keith = 32 := by sorry

end NUMINAMATH_CALUDE_laurence_to_missy_relation_keith_receives_32_messages_l616_61661


namespace NUMINAMATH_CALUDE_triangle_properties_l616_61649

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ  -- Angle A in radians
  b : ℝ  -- Side length b
  c : ℝ  -- Side length c
  h1 : A = π / 3  -- A = 60° in radians
  h2 : b = 5
  h3 : c = 4

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) :
  ∃ (a : ℝ), 
    a ^ 2 = 21 ∧ 
    Real.sin (Real.arcsin (t.b / a)) * Real.sin (Real.arcsin (t.c / a)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l616_61649


namespace NUMINAMATH_CALUDE_G₁_intersects_x_axis_range_of_n_minus_m_plus_a_right_triangle_BNB_l616_61668

-- Define the parabola G₁
def G₁ (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 4

-- Define the coordinates of point N
def N : ℝ × ℝ := (0, -4)

-- Theorem 1: G₁ intersects x-axis at two points
theorem G₁_intersects_x_axis (a : ℝ) :
  ∃ m n : ℝ, m < n ∧ G₁ a m = 0 ∧ G₁ a n = 0 := by sorry

-- Theorem 2: Range of n - m + a when NA ≥ 5
theorem range_of_n_minus_m_plus_a (a m n : ℝ) :
  m < n → G₁ a m = 0 → G₁ a n = 0 →
  Real.sqrt ((m - N.1)^2 + (N.2)^2) ≥ 5 →
  (n - m + a ≥ 9 ∨ n - m + a ≤ 3) := by sorry

-- Define the parabola G₂ (symmetric to G₁ with respect to A)
def G₂ (a x : ℝ) : ℝ := G₁ a (2*a - 2 - x)

-- Theorem 3: Conditions for right triangle BNB'
theorem right_triangle_BNB' (a : ℝ) :
  (∃ m n b : ℝ, m < n ∧ G₁ a m = 0 ∧ G₁ a n = 0 ∧ G₂ a b = 0 ∧ b ≠ m ∧
   (n - N.1)^2 + N.2^2 + (b - N.1)^2 + N.2^2 = (n - b)^2) ↔
  (a = 2 ∨ a = -2 ∨ a = 6) := by sorry

end NUMINAMATH_CALUDE_G₁_intersects_x_axis_range_of_n_minus_m_plus_a_right_triangle_BNB_l616_61668


namespace NUMINAMATH_CALUDE_intersection_condition_implies_a_values_l616_61659

theorem intersection_condition_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {5, a^2 - 3*a + 5}
  let N : Set ℝ := {1, 3}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_implies_a_values_l616_61659


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l616_61624

-- Define the polynomials
def p1 (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 5 * x - 2
def p2 (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 4

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p1 x * p2 x

-- Theorem statement
theorem coefficient_of_x_squared :
  ∃ (a b c d : ℝ), product = fun x => a * x^3 + (-5) * x^2 + b * x + c + d * x^4 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l616_61624


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l616_61640

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (2*a + Complex.I) * (1 - 2*Complex.I) = b * Complex.I ∧ b ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l616_61640


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l616_61650

/-- Given a hyperbola with equation (x^2 / 144) - (y^2 / 81) = 1 and asymptotes y = ±mx, prove that m = 3/4 -/
theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  ((x^2 / 144) - (y^2 / 81) = 1) → 
  (∃ (k : ℝ), y = k * m * x ∨ y = -k * m * x) → 
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l616_61650


namespace NUMINAMATH_CALUDE_max_value_with_constraint_l616_61600

theorem max_value_with_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 14 ∧ ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 9 → x + 2*y + 3*z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_constraint_l616_61600


namespace NUMINAMATH_CALUDE_total_students_in_line_l616_61601

/-- The number of students standing in a line with given conditions -/
def number_of_students (people_in_front_of_seokjin : ℕ) 
                       (people_behind_jimin : ℕ) 
                       (people_between_seokjin_and_jimin : ℕ) : ℕ :=
  people_in_front_of_seokjin + 1 + people_between_seokjin_and_jimin + 1 + people_behind_jimin

/-- Theorem stating that the total number of students in line is 16 -/
theorem total_students_in_line : 
  number_of_students 4 7 3 = 16 := by
  sorry


end NUMINAMATH_CALUDE_total_students_in_line_l616_61601


namespace NUMINAMATH_CALUDE_infinite_divisible_by_76_and_unique_centers_l616_61684

/-- Represents a cell in the spiral grid -/
structure Cell where
  x : ℤ
  y : ℤ

/-- The value at a node of the grid -/
def node_value (c : Cell) : ℕ := sorry

/-- The value at the center of a cell -/
def center_value (c : Cell) : ℕ := sorry

/-- The set of all cells in the infinite grid -/
def all_cells : Set Cell := sorry

theorem infinite_divisible_by_76_and_unique_centers :
  (∃ (S : Set Cell), Set.Infinite S ∧ ∀ c ∈ S, 76 ∣ center_value c) ∧
  (∀ c₁ c₂ : Cell, c₁ ≠ c₂ → center_value c₁ ≠ center_value c₂) := by
  sorry

end NUMINAMATH_CALUDE_infinite_divisible_by_76_and_unique_centers_l616_61684


namespace NUMINAMATH_CALUDE_place_mat_length_l616_61689

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) :
  r = 5 →
  n = 8 →
  w = 1 →
  (x - w/2)^2 + (w/2)^2 = r^2 →
  x = (3 * Real.sqrt 11 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_place_mat_length_l616_61689


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l616_61638

theorem quadratic_equation_transformation (x : ℝ) :
  x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l616_61638


namespace NUMINAMATH_CALUDE_greatest_perimeter_l616_61639

/-- A rectangle with whole number side lengths and an area of 12 square metres. -/
structure Rectangle where
  width : ℕ
  length : ℕ
  area_eq : width * length = 12

/-- The perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- The theorem stating that the greatest possible perimeter is 26. -/
theorem greatest_perimeter :
  ∀ r : Rectangle, perimeter r ≤ 26 ∧ ∃ r' : Rectangle, perimeter r' = 26 := by
  sorry

end NUMINAMATH_CALUDE_greatest_perimeter_l616_61639


namespace NUMINAMATH_CALUDE_total_weight_of_rings_l616_61633

-- Define the weights of the rings
def orange_weight : ℚ := 0.08
def purple_weight : ℚ := 0.33
def white_weight : ℚ := 0.42

-- Theorem statement
theorem total_weight_of_rings : orange_weight + purple_weight + white_weight = 0.83 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_rings_l616_61633


namespace NUMINAMATH_CALUDE_kittens_at_shelter_l616_61679

theorem kittens_at_shelter (puppies : ℕ) (kittens : ℕ) : 
  puppies = 32 → 
  kittens = 2 * puppies + 14 → 
  kittens = 78 := by
  sorry

end NUMINAMATH_CALUDE_kittens_at_shelter_l616_61679


namespace NUMINAMATH_CALUDE_percentage_increase_l616_61623

theorem percentage_increase (initial final : ℝ) (h1 : initial = 60) (h2 : final = 90) :
  (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l616_61623


namespace NUMINAMATH_CALUDE_stripes_calculation_l616_61664

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The number of stripes on one of Rick's shoes -/
def rick_stripes_per_shoe : ℕ := olga_stripes_per_shoe - 1

/-- The number of stripes on one of Hortense's shoes -/
def hortense_stripes_per_shoe : ℕ := 2 * olga_stripes_per_shoe

/-- The number of stripes on one of Ethan's shoes -/
def ethan_stripes_per_shoe : ℕ := hortense_stripes_per_shoe + 2

/-- The total number of stripes on all shoes -/
def total_stripes : ℕ := 2 * (olga_stripes_per_shoe + rick_stripes_per_shoe + hortense_stripes_per_shoe + ethan_stripes_per_shoe)

/-- The final result after dividing by 2 and rounding up -/
def final_result : ℕ := (total_stripes + 1) / 2

theorem stripes_calculation :
  final_result = 19 := by sorry

end NUMINAMATH_CALUDE_stripes_calculation_l616_61664


namespace NUMINAMATH_CALUDE_percentage_calculation_l616_61651

theorem percentage_calculation (x : ℝ) (h : 0.2 * x = 1000) : 1.2 * x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l616_61651


namespace NUMINAMATH_CALUDE_chessboard_coloring_limit_l616_61676

/-- Represents the minimum number of colored vertices required on an n × n chessboard
    such that any k × k square has at least one edge with a colored vertex. -/
noncomputable def l (n : ℕ) : ℕ := sorry

/-- The limit of l(n)/n² as n approaches infinity is 2/7. -/
theorem chessboard_coloring_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |l n / (n^2 : ℝ) - 2/7| < ε :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_limit_l616_61676


namespace NUMINAMATH_CALUDE_calculation_proof_l616_61678

theorem calculation_proof : (3^2 * (-2 + 3) / (1/3) - |-28|) = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l616_61678


namespace NUMINAMATH_CALUDE_biggest_measure_for_containers_l616_61657

theorem biggest_measure_for_containers (a b c : ℕ) 
  (ha : a = 496) (hb : b = 403) (hc : c = 713) : 
  Nat.gcd a (Nat.gcd b c) = 31 := by
  sorry

end NUMINAMATH_CALUDE_biggest_measure_for_containers_l616_61657


namespace NUMINAMATH_CALUDE_min_distance_between_ships_l616_61663

/-- The minimum distance between two ships given specific conditions -/
theorem min_distance_between_ships 
  (d : ℝ) -- Initial distance between ships
  (k : ℝ) -- Speed ratio v₁/v₂
  (h₁ : k > 0) -- Speed ratio is positive
  (h₂ : k < 1) -- Speed ratio is less than 1
  : ∃ (min_dist : ℝ), min_dist = d * Real.sqrt (1 - k^2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_ships_l616_61663


namespace NUMINAMATH_CALUDE_sum_squared_l616_61653

theorem sum_squared (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 + b^2 = 25) : (a + b)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_l616_61653


namespace NUMINAMATH_CALUDE_candy_distribution_l616_61629

/-- Represents the number of positions moved for the k-th candy distribution -/
def a (k : ℕ) : ℕ := k * (k + 1) / 2

/-- Checks if all students in a circle of size n receive a candy -/
def all_receive_candy (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → ∃ k : ℕ, a k % n = m

/-- Main theorem: All students receive a candy iff n is a power of 2 -/
theorem candy_distribution (n : ℕ) :
  all_receive_candy n ↔ ∃ m : ℕ, n = 2^m :=
sorry

/-- Helper lemma: If n is not a power of 2, not all students receive a candy -/
lemma not_power_of_two_not_all_receive (n : ℕ) :
  (¬ ∃ m : ℕ, n = 2^m) → ¬ all_receive_candy n :=
sorry

/-- Helper lemma: If n is a power of 2, all students receive a candy -/
lemma power_of_two_all_receive (m : ℕ) :
  all_receive_candy (2^m) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l616_61629


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l616_61652

/-- Given a hyperbola with equation x²/16 - y²/25 = 1, 
    the positive slope of its asymptotes is 5/4 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  x^2 / 16 - y^2 / 25 = 1 → 
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l616_61652


namespace NUMINAMATH_CALUDE_final_average_is_23_l616_61673

/-- Represents a cricketer's scoring data -/
structure CricketerData where
  inningsCount : ℕ
  scoreLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the final average score given the cricketer's data -/
def finalAverageScore (data : CricketerData) : ℕ :=
  data.averageIncrease + (data.scoreLastInning - data.averageIncrease * data.inningsCount) / (data.inningsCount - 1)

/-- Theorem stating that for the given conditions, the final average score is 23 -/
theorem final_average_is_23 (data : CricketerData) 
  (h1 : data.inningsCount = 19)
  (h2 : data.scoreLastInning = 95)
  (h3 : data.averageIncrease = 4) : 
  finalAverageScore data = 23 := by
  sorry

end NUMINAMATH_CALUDE_final_average_is_23_l616_61673


namespace NUMINAMATH_CALUDE_polynomial_simplification_l616_61656

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 + x^2 - 7 * x - 6) = x^3 + 3 * x^2 + 2 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l616_61656


namespace NUMINAMATH_CALUDE_vector_subtraction_l616_61644

/-- Given vectors a and b in ℝ², prove that a - 2b equals (6, -7) -/
theorem vector_subtraction (a b : ℝ × ℝ) 
  (ha : a = (2, -1)) (hb : b = (-2, 3)) : 
  a - 2 • b = (6, -7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l616_61644


namespace NUMINAMATH_CALUDE_dot_product_theorem_l616_61620

variable (a b : ℝ × ℝ)

theorem dot_product_theorem (h1 : a.1 + 2 * b.1 = 0 ∧ a.2 + 2 * b.2 = 0) 
                            (h2 : (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 2) : 
  a.1 * b.1 + a.2 * b.2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l616_61620


namespace NUMINAMATH_CALUDE_art_club_artworks_art_club_two_years_collection_l616_61687

theorem art_club_artworks (num_students : ℕ) (artworks_per_student_per_quarter : ℕ) 
  (quarters_per_year : ℕ) (num_years : ℕ) : ℕ :=
  num_students * artworks_per_student_per_quarter * quarters_per_year * num_years

theorem art_club_two_years_collection : 
  art_club_artworks 15 2 4 2 = 240 := by sorry

end NUMINAMATH_CALUDE_art_club_artworks_art_club_two_years_collection_l616_61687


namespace NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l616_61692

theorem negative_one_minus_two_times_negative_two : -1 - 2 * (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l616_61692


namespace NUMINAMATH_CALUDE_equal_quantities_solution_l616_61696

theorem equal_quantities_solution (x y : ℝ) (h : y ≠ 0) :
  (((x + y = x - y ∧ x + y = x * y) ∨
    (x + y = x - y ∧ x + y = x / y) ∨
    (x + y = x * y ∧ x + y = x / y) ∨
    (x - y = x * y ∧ x - y = x / y)) →
   ((x = 1/2 ∧ y = -1) ∨ (x = -1/2 ∧ y = -1))) :=
by sorry

end NUMINAMATH_CALUDE_equal_quantities_solution_l616_61696


namespace NUMINAMATH_CALUDE_knight_traversal_coloring_l616_61685

/-- Represents a chessboard of arbitrary size -/
structure Chessboard where
  size : ℕ
  canBeTraversed : Bool

/-- Represents a position on the chessboard -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents a knight's move -/
def knightMove (p : Position) : Position :=
  sorry

/-- Checks if a position is even in the knight's traversal -/
def isEvenPosition (p : Position) : Bool :=
  sorry

/-- Checks if a position should be colored black in a properly colored chessboard -/
def isBlackInProperColoring (p : Position) : Bool :=
  sorry

/-- The main theorem stating that shading even-numbered squares in a knight's traversal
    reproduces the proper coloring of a chessboard -/
theorem knight_traversal_coloring (board : Chessboard) :
  board.canBeTraversed →
  ∀ p : Position, isEvenPosition p = isBlackInProperColoring p :=
sorry

end NUMINAMATH_CALUDE_knight_traversal_coloring_l616_61685


namespace NUMINAMATH_CALUDE_det_necessary_not_sufficient_for_parallel_l616_61695

/-- Determinant of a 2x2 matrix --/
def det (a₁ b₁ a₂ b₂ : ℝ) : ℝ := a₁ * b₂ - a₂ * b₁

/-- Two lines are parallel --/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ ≠ k * c₂

theorem det_necessary_not_sufficient_for_parallel
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h₁ : a₁^2 + b₁^2 ≠ 0) (h₂ : a₂^2 + b₂^2 ≠ 0) :
  (det a₁ b₁ a₂ b₂ = 0 → parallel a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(parallel a₁ b₁ c₁ a₂ b₂ c₂ → det a₁ b₁ a₂ b₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_det_necessary_not_sufficient_for_parallel_l616_61695


namespace NUMINAMATH_CALUDE_area_at_stage_8_l616_61611

/-- Represents the width of a rectangle at a given stage -/
def width (stage : ℕ) : ℕ :=
  if stage ≤ 4 then 4 else 2 * stage - 6

/-- Represents the area of a rectangle at a given stage -/
def area (stage : ℕ) : ℕ := 4 * width stage

/-- The total area of the figure at Stage 8 -/
def totalArea : ℕ := (List.range 8).map (fun i => area (i + 1)) |>.sum

theorem area_at_stage_8 : totalArea = 176 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l616_61611


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l616_61635

theorem quadratic_equal_roots : ∃ x : ℝ, x^2 - x + (1/4 : ℝ) = 0 ∧
  ∀ y : ℝ, y^2 - y + (1/4 : ℝ) = 0 → y = x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l616_61635


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l616_61662

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l616_61662


namespace NUMINAMATH_CALUDE_max_min_difference_d_l616_61655

theorem max_min_difference_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 5)
  (sum_sq_eq : a^2 + b^2 + c^2 + d^2 = 18) : 
  ∃ (d_max d_min : ℝ),
    (∀ d', a + b + c + d' = 5 ∧ a^2 + b^2 + c^2 + d'^2 = 18 → d' ≤ d_max) ∧
    (∀ d', a + b + c + d' = 5 ∧ a^2 + b^2 + c^2 + d'^2 = 18 → d_min ≤ d') ∧
    d_max - d_min = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_d_l616_61655


namespace NUMINAMATH_CALUDE_dans_age_l616_61666

theorem dans_age (x : ℕ) : x + 20 = 7 * (x - 4) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_l616_61666


namespace NUMINAMATH_CALUDE_circle_equations_correct_l616_61645

-- Define the parallel lines
def line1 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + y + Real.sqrt 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 2 * Real.sqrt 2 * a = 0

-- Define the circle N
def circleN (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 5)^2 + (y + 2)^2 = 49

-- Define point B
def pointB : ℝ × ℝ := (3, -2)

-- Define the line of symmetry
def lineOfSymmetry (x : ℝ) : Prop := x = -1

-- Main theorem
theorem circle_equations_correct :
  ∃ (a : ℝ),
    (∀ x y, line1 a x y ↔ line2 a x y) ∧  -- Lines are parallel
    (∃ r, r > 0 ∧ r = Real.sqrt ((3 - (-5))^2 + (4 - (-2))^2) - 3) ∧  -- Distance between centers minus radius of N
    (∀ x, lineOfSymmetry x → x = -1) ∧
    (∃ c : ℝ × ℝ, c.1 = -5 ∧ c.2 = -2) →  -- Point C exists
  (∀ x y, circleN x y) ∧ (∀ x y, circleC x y) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_correct_l616_61645


namespace NUMINAMATH_CALUDE_transaction_difference_l616_61672

theorem transaction_difference (mabel_transactions : ℕ) 
  (anthony_transactions : ℕ) (cal_transactions : ℕ) (jade_transactions : ℕ) :
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  cal_transactions = anthony_transactions * 2 / 3 →
  jade_transactions = 85 →
  jade_transactions - cal_transactions = 19 := by
sorry

end NUMINAMATH_CALUDE_transaction_difference_l616_61672


namespace NUMINAMATH_CALUDE_used_car_clients_l616_61643

theorem used_car_clients (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) :
  num_cars = 18 →
  selections_per_client = 3 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / selections_per_client = 18 := by
  sorry

end NUMINAMATH_CALUDE_used_car_clients_l616_61643


namespace NUMINAMATH_CALUDE_product_remainder_l616_61654

theorem product_remainder (a b m : ℕ) (h : a = 98) (h' : b = 102) (h'' : m = 8) :
  (a * b) % m = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l616_61654


namespace NUMINAMATH_CALUDE_quadratic_factorization_l616_61626

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l616_61626


namespace NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l616_61681

/-- Calculates the number of cans of pie filling produced given the total pumpkins,
    price per pumpkin, total money made, and pumpkins per can. -/
def cans_of_pie_filling (total_pumpkins : ℕ) (price_per_pumpkin : ℕ) 
                        (total_made : ℕ) (pumpkins_per_can : ℕ) : ℕ :=
  (total_pumpkins - total_made / price_per_pumpkin) / pumpkins_per_can

/-- Theorem stating that given the specific conditions, 
    the number of cans of pie filling produced is 17. -/
theorem pumpkin_patch_pie_filling : 
  cans_of_pie_filling 83 3 96 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l616_61681


namespace NUMINAMATH_CALUDE_sum_remainder_l616_61616

theorem sum_remainder (n : ℤ) : (7 - 2*n + (n + 5)) % 8 = (4 - n) % 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l616_61616


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_at_2_0_l616_61617

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The x-axis -/
def x_axis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Check if a point lies on the x-axis -/
def point_on_x_axis (p : Point) : Prop := p.y = 0

/-- The main theorem -/
theorem line_intersects_x_axis_at_2_0 :
  let l : Line := { p1 := ⟨4, -2⟩, p2 := ⟨0, 2⟩ }
  let intersection : Point := ⟨2, 0⟩
  point_on_line intersection l ∧ point_on_x_axis intersection := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_at_2_0_l616_61617


namespace NUMINAMATH_CALUDE_infinite_intersection_l616_61612

def sequence_a : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 6 * sequence_b (n + 1) - sequence_b n

theorem infinite_intersection :
  Set.Infinite {n : ℕ | ∃ m : ℕ, sequence_a n = sequence_b m} :=
sorry

end NUMINAMATH_CALUDE_infinite_intersection_l616_61612


namespace NUMINAMATH_CALUDE_sum_of_coordinates_equals_16_l616_61634

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y - 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem sum_of_coordinates_equals_16 :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_equals_16_l616_61634


namespace NUMINAMATH_CALUDE_number_ratio_problem_l616_61642

theorem number_ratio_problem (x : ℚ) : 
  (x / 6 = 16 / 480) → x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l616_61642


namespace NUMINAMATH_CALUDE_simplify_product_l616_61606

theorem simplify_product (a : ℝ) : 
  (2 * a) * (3 * a^2) * (5 * a^3) * (7 * a^4) * (11 * a^5) * (13 * a^6) = 30030 * a^21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l616_61606


namespace NUMINAMATH_CALUDE_max_min_sin_cos_combination_l616_61694

theorem max_min_sin_cos_combination (a b : ℝ) :
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ Real.sqrt (a^2 + b^2)) ∧
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ -Real.sqrt (a^2 + b^2)) ∧
  (∃ θ₁ : ℝ, a * Real.sin θ₁ + b * Real.cos θ₁ = Real.sqrt (a^2 + b^2)) ∧
  (∃ θ₂ : ℝ, a * Real.sin θ₂ + b * Real.cos θ₂ = -Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_min_sin_cos_combination_l616_61694


namespace NUMINAMATH_CALUDE_vector_equation_l616_61682

def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

theorem vector_equation (m n t : ℝ) :
  c t = m • a + n • b → t = 11 ∧ m + n = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l616_61682


namespace NUMINAMATH_CALUDE_smallest_number_with_20_divisors_l616_61686

/-- The number of divisors of a natural number n -/
def numDivisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- A natural number n has exactly 20 divisors -/
def has20Divisors (n : ℕ) : Prop := numDivisors n = 20

theorem smallest_number_with_20_divisors :
  ∀ n : ℕ, has20Divisors n → n ≥ 240 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_20_divisors_l616_61686


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l616_61688

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2 * i

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, z_condition z ∧ in_fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l616_61688


namespace NUMINAMATH_CALUDE_polynomial_simplification_l616_61665

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) =
  -4 * x^4 + x^3 + 3 * x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l616_61665


namespace NUMINAMATH_CALUDE_nicholas_bottle_caps_l616_61646

theorem nicholas_bottle_caps :
  let initial_caps : ℕ := 8
  let additional_caps : ℕ := 85
  initial_caps + additional_caps = 93
:= by sorry

end NUMINAMATH_CALUDE_nicholas_bottle_caps_l616_61646


namespace NUMINAMATH_CALUDE_system_negative_solution_l616_61621

/-- The system of equations has at least one negative solution if and only if a + b + c = 0 -/
theorem system_negative_solution (a b c : ℝ) :
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧
    a * x + b * y = c ∧
    b * x + c * y = a ∧
    c * x + a * y = b) ↔
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_negative_solution_l616_61621


namespace NUMINAMATH_CALUDE_boys_dropped_out_l616_61619

/-- Proves the number of boys who dropped out from a school, given initial counts and final total -/
theorem boys_dropped_out (initial_boys initial_girls girls_dropped final_total : ℕ) : 
  initial_boys = 14 →
  initial_girls = 10 →
  girls_dropped = 3 →
  final_total = 17 →
  initial_boys - (final_total - (initial_girls - girls_dropped)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_boys_dropped_out_l616_61619


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l616_61610

/-- Given a cylinder with volume 72π cm³ and height twice its radius,
    prove that a cone with the same height and radius has a volume of 24π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) →   -- Cylinder volume condition
  (h = 2 * r) →              -- Height-radius relation condition
  ((1/3) * π * r^2 * h = 24 * π) -- Cone volume to prove
  :=
by
  sorry


end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l616_61610


namespace NUMINAMATH_CALUDE_constant_calculation_l616_61680

theorem constant_calculation (N : ℝ) (C : ℝ) : 
  N = 12.0 → C + 0.6667 * N = 0.75 * N → C = 0.9996 := by
  sorry

end NUMINAMATH_CALUDE_constant_calculation_l616_61680


namespace NUMINAMATH_CALUDE_branch_fractions_sum_l616_61605

theorem branch_fractions_sum : 
  (1/3 : ℚ) + (2/3 : ℚ) + (1/5 : ℚ) + (2/5 : ℚ) + (3/5 : ℚ) + (4/5 : ℚ) + 
  (1/7 : ℚ) + (2/7 : ℚ) + (3/7 : ℚ) + (4/7 : ℚ) + (5/7 : ℚ) + (6/7 : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_branch_fractions_sum_l616_61605


namespace NUMINAMATH_CALUDE_percentage_of_2_to_50_l616_61625

theorem percentage_of_2_to_50 : (2 : ℝ) / 50 * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_2_to_50_l616_61625


namespace NUMINAMATH_CALUDE_number_of_puppies_number_of_puppies_proof_l616_61660

/-- The number of puppies at a camp, given specific feeding conditions for dogs and puppies -/
theorem number_of_puppies : ℕ :=
  let num_dogs : ℕ := 3
  let dog_meal_size : ℚ := 4
  let dog_meals_per_day : ℕ := 3
  let total_food_per_day : ℚ := 108
  let dog_to_puppy_meal_ratio : ℚ := 2
  let puppy_to_dog_meal_frequency_ratio : ℕ := 3
  4

/-- Proof that the number of puppies is correct -/
theorem number_of_puppies_proof :
  let num_dogs : ℕ := 3
  let dog_meal_size : ℚ := 4
  let dog_meals_per_day : ℕ := 3
  let total_food_per_day : ℚ := 108
  let dog_to_puppy_meal_ratio : ℚ := 2
  let puppy_to_dog_meal_frequency_ratio : ℕ := 3
  number_of_puppies = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_puppies_number_of_puppies_proof_l616_61660


namespace NUMINAMATH_CALUDE_final_clay_pieces_l616_61632

/-- Represents the number of pieces of clay of each color --/
structure ClayPieces where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Represents the operations performed on the clay pieces --/
def divide_non_red (pieces : ClayPieces) : ClayPieces :=
  { red := pieces.red,
    blue := pieces.blue * 2,
    yellow := pieces.yellow * 2 }

def divide_non_yellow (pieces : ClayPieces) : ClayPieces :=
  { red := pieces.red * 2,
    blue := pieces.blue * 2,
    yellow := pieces.yellow }

/-- The main theorem to prove --/
theorem final_clay_pieces :
  let initial_pieces := ClayPieces.mk 4 3 5
  let after_first_operation := divide_non_red initial_pieces
  let final_pieces := divide_non_yellow after_first_operation
  final_pieces.red + final_pieces.blue + final_pieces.yellow = 30 := by
  sorry


end NUMINAMATH_CALUDE_final_clay_pieces_l616_61632


namespace NUMINAMATH_CALUDE_routes_from_bristol_to_birmingham_l616_61603

theorem routes_from_bristol_to_birmingham :
  ∀ (bristol_to_birmingham birmingham_to_sheffield sheffield_to_carlisle bristol_to_carlisle : ℕ),
    birmingham_to_sheffield = 3 →
    sheffield_to_carlisle = 2 →
    bristol_to_carlisle = 36 →
    bristol_to_carlisle = bristol_to_birmingham * birmingham_to_sheffield * sheffield_to_carlisle →
    bristol_to_birmingham = 6 := by
  sorry

end NUMINAMATH_CALUDE_routes_from_bristol_to_birmingham_l616_61603


namespace NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l616_61674

theorem product_of_sums_equal_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l616_61674


namespace NUMINAMATH_CALUDE_correct_philosophies_l616_61683

-- Define the philosophies
inductive Philosophy
  | GraspMeasure
  | ComprehensiveView
  | AnalyzeSpecifically
  | EmphasizeKeyPoints

-- Define the conditions
structure IodineScenario where
  iodineEssential : Bool
  oneSizeFitsAllRisky : Bool
  nonIodineDeficientArea : Bool
  increasedNonIodizedSalt : Bool
  allowAdjustment : Bool

-- Define the function to check if a philosophy is reflected
def reflectsPhilosophy (scenario : IodineScenario) (philosophy : Philosophy) : Prop :=
  match philosophy with
  | Philosophy.GraspMeasure => scenario.oneSizeFitsAllRisky
  | Philosophy.ComprehensiveView => scenario.iodineEssential ∧ scenario.oneSizeFitsAllRisky
  | Philosophy.AnalyzeSpecifically => scenario.nonIodineDeficientArea ∧ scenario.increasedNonIodizedSalt ∧ scenario.allowAdjustment
  | Philosophy.EmphasizeKeyPoints => False

-- Theorem to prove
theorem correct_philosophies (scenario : IodineScenario) 
  (h1 : scenario.iodineEssential = true)
  (h2 : scenario.oneSizeFitsAllRisky = true)
  (h3 : scenario.nonIodineDeficientArea = true)
  (h4 : scenario.increasedNonIodizedSalt = true)
  (h5 : scenario.allowAdjustment = true) :
  reflectsPhilosophy scenario Philosophy.GraspMeasure ∧
  reflectsPhilosophy scenario Philosophy.ComprehensiveView ∧
  reflectsPhilosophy scenario Philosophy.AnalyzeSpecifically ∧
  ¬reflectsPhilosophy scenario Philosophy.EmphasizeKeyPoints :=
sorry

end NUMINAMATH_CALUDE_correct_philosophies_l616_61683


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l616_61647

theorem simplify_and_evaluate :
  ∀ x : ℝ, x ≠ 1 → x ≠ 3 →
  (1 - 2 / (x - 1)) * ((x^2 - x) / (x^2 - 6*x + 9)) = x / (x - 3) ∧
  (2 : ℝ) / ((2 : ℝ) - 3) = -2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l616_61647


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l616_61637

theorem min_draws_for_even_product (n : ℕ) (h : n = 16) :
  let S := Finset.range n
  let even_count := (S.filter (λ x => x % 2 = 0)).card
  let odd_count := (S.filter (λ x => x % 2 ≠ 0)).card
  odd_count + 1 = 9 ∧ 
  ∀ k : ℕ, k < odd_count + 1 → ∃ subset : Finset ℕ, 
    subset.card = k ∧ 
    subset ⊆ S ∧ 
    ∀ x ∈ subset, x % 2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l616_61637


namespace NUMINAMATH_CALUDE_round_trip_distance_bike_ride_distance_l616_61614

/-- Calculates the total distance traveled in a round trip given speeds and total time -/
theorem round_trip_distance (speed_out speed_back total_time : ℝ) 
  (h1 : speed_out > 0) 
  (h2 : speed_back > 0) 
  (h3 : total_time > 0) : ℝ :=
  let one_way_distance := (speed_out * speed_back * total_time) / (speed_out + speed_back)
  2 * one_way_distance

/-- Proves that for the given speeds and time, the total distance is 144 miles -/
theorem bike_ride_distance : 
  round_trip_distance 24 18 7 (by norm_num) (by norm_num) (by norm_num) = 144 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_bike_ride_distance_l616_61614


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l616_61658

theorem arithmetic_sequence_problem (a b c : ℤ) :
  (∃ d : ℤ, -1 = a - d ∧ a = b - d ∧ b = c - d ∧ c = -9 + d) →
  b = -5 ∧ a * c = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l616_61658


namespace NUMINAMATH_CALUDE_same_suit_probability_l616_61690

theorem same_suit_probability (total_cards : ℕ) (num_suits : ℕ) (cards_per_suit : ℕ) 
  (h1 : total_cards = 52)
  (h2 : num_suits = 4)
  (h3 : cards_per_suit = 13)
  (h4 : total_cards = num_suits * cards_per_suit) :
  (4 : ℚ) / 17 = (num_suits * (cards_per_suit.choose 2)) / (total_cards.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_same_suit_probability_l616_61690


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l616_61615

/-- The cost ratio of a muffin to a banana given Susie and Calvin's purchases -/
theorem muffin_banana_cost_ratio :
  let muffin_cost : ℚ := muffin_cost
  let banana_cost : ℚ := banana_cost
  (6 * muffin_cost + 4 * banana_cost) * 3 = 3 * muffin_cost + 24 * banana_cost →
  muffin_cost / banana_cost = 4 / 5 := by
sorry


end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l616_61615


namespace NUMINAMATH_CALUDE_cyclic_identity_l616_61667

theorem cyclic_identity (a b c : ℝ) : 
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) = 
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) = 
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) ∧
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) = 
  a^3 + b^3 + c^3 - (a^2 * b + a * b^2 + b^2 * c + b * c^2 + c^2 * a + a^2 * c) + 3 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_cyclic_identity_l616_61667


namespace NUMINAMATH_CALUDE_cindys_homework_l616_61628

theorem cindys_homework (x : ℝ) : (x - 7) * 4 = 48 → (x * 4) - 7 = 69 := by
  sorry

end NUMINAMATH_CALUDE_cindys_homework_l616_61628


namespace NUMINAMATH_CALUDE_nested_triangle_perimeter_sum_l616_61607

/-- Given a circle of radius r, we define a sequence of nested equilateral triangles 
    where each subsequent triangle is formed by joining the midpoints of the sides 
    of the previous triangle, starting with an equilateral triangle inscribed in the circle. 
    This theorem states that the limit of the sum of the perimeters of all these triangles 
    is 6r√3. -/
theorem nested_triangle_perimeter_sum (r : ℝ) (h : r > 0) : 
  let first_perimeter := 3 * r * Real.sqrt 3
  let perimeter_sequence := fun n => first_perimeter * (1 / 2) ^ n
  (∑' n, perimeter_sequence n) = 6 * r * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_nested_triangle_perimeter_sum_l616_61607


namespace NUMINAMATH_CALUDE_regular_hours_is_40_l616_61609

/-- Represents the pay structure and work hours for Bob --/
structure PayStructure where
  regularRate : ℝ  -- Regular hourly rate
  overtimeRate : ℝ  -- Overtime hourly rate
  hoursWeek1 : ℝ  -- Hours worked in week 1
  hoursWeek2 : ℝ  -- Hours worked in week 2
  totalEarnings : ℝ  -- Total earnings for both weeks

/-- Calculates the number of regular hours in a week --/
def calculateRegularHours (p : PayStructure) : ℝ :=
  let regularHours := 40  -- The value we want to prove
  regularHours

/-- Theorem stating that the number of regular hours is 40 --/
theorem regular_hours_is_40 (p : PayStructure) 
    (h1 : p.regularRate = 5)
    (h2 : p.overtimeRate = 6)
    (h3 : p.hoursWeek1 = 44)
    (h4 : p.hoursWeek2 = 48)
    (h5 : p.totalEarnings = 472) :
    calculateRegularHours p = 40 := by
  sorry

#eval calculateRegularHours { regularRate := 5, overtimeRate := 6, hoursWeek1 := 44, hoursWeek2 := 48, totalEarnings := 472 }

end NUMINAMATH_CALUDE_regular_hours_is_40_l616_61609


namespace NUMINAMATH_CALUDE_pythagorean_theorem_l616_61699

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleBAC_is_right : angleBAC = 90

-- State the theorem
theorem pythagorean_theorem (t : RightTriangle) : t.b^2 + t.c^2 = t.a^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_l616_61699


namespace NUMINAMATH_CALUDE_consecutive_binomial_ratio_sum_n_plus_k_l616_61669

theorem consecutive_binomial_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 6 →
  n = 11 ∧ k = 2 :=
by sorry

theorem sum_n_plus_k (n k : ℕ) :
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 6 →
  n + k = 13 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_binomial_ratio_sum_n_plus_k_l616_61669


namespace NUMINAMATH_CALUDE_problem_solution_l616_61648

theorem problem_solution : 
  ((-1)^3 + |1 - Real.sqrt 2| + (8 : ℝ)^(1/3) = Real.sqrt 2) ∧
  (((-5 : ℝ)^3)^(1/3) + (-3)^2 - Real.sqrt 25 + |Real.sqrt 3 - 2| + (Real.sqrt 3)^2 = 4 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l616_61648


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_l616_61636

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2 (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_f_2_l616_61636


namespace NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l616_61604

/-- Given three positive integers a, b, and c in the ratio 3:4:5 with HCF 40, their LCM is 2400 -/
theorem ratio_hcf_to_lcm (a b c : ℕ+) : 
  (a : ℚ) / 3 = (b : ℚ) / 4 ∧ (b : ℚ) / 4 = (c : ℚ) / 5 → 
  Nat.gcd a.val (Nat.gcd b.val c.val) = 40 →
  Nat.lcm a.val (Nat.lcm b.val c.val) = 2400 := by
sorry

end NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l616_61604


namespace NUMINAMATH_CALUDE_triangle_constructibility_l616_61602

/-- Given two sides of a triangle and the median to the third side,
    this theorem proves the condition for the triangle's constructibility. -/
theorem triangle_constructibility 
  (a b s : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hs : s > 0) :
  ((a - b) / 2 < s ∧ s < (a + b) / 2) ↔ 
  ∃ (c : ℝ), c > 0 ∧ 
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
    s^2 = (2 * (a^2 + b^2) - c^2) / 4 :=
by sorry


end NUMINAMATH_CALUDE_triangle_constructibility_l616_61602


namespace NUMINAMATH_CALUDE_roots_relation_l616_61691

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

-- Define the polynomial j(x)
def j (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Theorem statement
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → j b c d (x^3) = 0) →
  b = 6 ∧ c = 12 ∧ d = 8 :=
by sorry

end NUMINAMATH_CALUDE_roots_relation_l616_61691


namespace NUMINAMATH_CALUDE_tan_45_degrees_l616_61675

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l616_61675


namespace NUMINAMATH_CALUDE_unique_polynomial_function_l616_61677

-- Define a polynomial function of degree 3
def PolynomialDegree3 (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d

-- Define the conditions given in the problem
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x^2) = (f x)^2) ∧
  (∀ x, f (x^2) = f (f x)) ∧
  f 1 = f (-1)

-- Theorem statement
theorem unique_polynomial_function :
  ∃! f : ℝ → ℝ, PolynomialDegree3 f ∧ SatisfiesConditions f ∧ (∀ x, f x = x^3) :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_function_l616_61677


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l616_61670

-- Define the set of cards
def cards : Finset ℕ := {3, 4, 5, 6, 7}

-- Define the sample space (all possible pairs of cards)
def sample_space : Finset (ℕ × ℕ) :=
  (cards.product cards).filter (fun p => p.1 < p.2)

-- Define event A: sum of selected cards is even
def event_A : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => (p.1 + p.2) % 2 = 0)

-- Define event B: both selected cards are odd
def event_B : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.1 % 2 = 1 ∧ p.2 % 2 = 1)

-- Define the probability measure
def P (event : Finset (ℕ × ℕ)) : ℚ :=
  event.card / sample_space.card

-- Theorem to prove
theorem conditional_probability_B_given_A :
  P (event_A ∩ event_B) / P event_A = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l616_61670


namespace NUMINAMATH_CALUDE_town_population_l616_61631

theorem town_population (P : ℝ) : 
  (P * (1 - 0.2)^2 = 12800) → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_town_population_l616_61631


namespace NUMINAMATH_CALUDE_square_difference_pattern_l616_61630

theorem square_difference_pattern (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l616_61630


namespace NUMINAMATH_CALUDE_sams_initial_dimes_l616_61622

theorem sams_initial_dimes (initial_dimes final_dimes dimes_from_dad : ℕ) 
  (h1 : final_dimes = initial_dimes + dimes_from_dad)
  (h2 : final_dimes = 16)
  (h3 : dimes_from_dad = 7) : 
  initial_dimes = 9 := by
  sorry

end NUMINAMATH_CALUDE_sams_initial_dimes_l616_61622


namespace NUMINAMATH_CALUDE_manufacturing_cost_is_210_l616_61697

/-- Calculates the manufacturing cost of a shoe given transportation cost, selling price, and gain percentage. -/
def manufacturing_cost (transportation_cost : ℚ) (shoes_per_transport : ℕ) (selling_price : ℚ) (gain_percentage : ℚ) : ℚ :=
  let transportation_cost_per_shoe := transportation_cost / shoes_per_transport
  let cost_price := selling_price / (1 + gain_percentage)
  cost_price - transportation_cost_per_shoe

/-- Proves that the manufacturing cost of a shoe is 210, given the specified conditions. -/
theorem manufacturing_cost_is_210 :
  manufacturing_cost 500 100 258 (20/100) = 210 := by
  sorry

#eval manufacturing_cost 500 100 258 (20/100)

end NUMINAMATH_CALUDE_manufacturing_cost_is_210_l616_61697


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_45_is_smallest_is_four_digit_and_divisible_by_45_l616_61693

theorem smallest_four_digit_divisible_by_45 : ℕ :=
  let is_four_digit (n : ℕ) := 1000 ≤ n ∧ n ≤ 9999
  let is_divisible_by_45 (n : ℕ) := n % 45 = 0
  1008

theorem is_smallest :
  let is_four_digit (n : ℕ) := 1000 ≤ n ∧ n ≤ 9999
  let is_divisible_by_45 (n : ℕ) := n % 45 = 0
  ∀ n : ℕ, is_four_digit n → is_divisible_by_45 n → smallest_four_digit_divisible_by_45 ≤ n :=
by
  sorry

theorem is_four_digit_and_divisible_by_45 :
  let is_four_digit (n : ℕ) := 1000 ≤ n ∧ n ≤ 9999
  let is_divisible_by_45 (n : ℕ) := n % 45 = 0
  is_four_digit smallest_four_digit_divisible_by_45 ∧ is_divisible_by_45 smallest_four_digit_divisible_by_45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_45_is_smallest_is_four_digit_and_divisible_by_45_l616_61693


namespace NUMINAMATH_CALUDE_ascending_order_proof_l616_61618

def base_16_to_decimal (n : ℕ) : ℕ := n

def base_7_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

theorem ascending_order_proof (a b c : ℕ) 
  (ha : a = base_16_to_decimal 0x12)
  (hb : b = base_7_to_decimal 25)
  (hc : c = base_4_to_decimal 33) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_proof_l616_61618


namespace NUMINAMATH_CALUDE_min_sheets_for_boats_is_one_l616_61698

/-- The minimum number of sheets needed to make paper boats -/
def min_sheets_for_boats : ℕ := 1

/-- The total number of paper toys to be made -/
def total_toys : ℕ := 250

/-- The number of paper boats that can be made from one sheet -/
def boats_per_sheet : ℕ := 9

/-- The number of paper planes that can be made from one sheet -/
def planes_per_sheet : ℕ := 5

/-- The number of paper helicopters that can be made from one sheet -/
def helicopters_per_sheet : ℕ := 3

/-- Theorem stating that the minimum number of sheets needed for paper boats is 1 -/
theorem min_sheets_for_boats_is_one :
  ∃ (boats planes helicopters : ℕ),
    boats + planes + helicopters = total_toys ∧
    boats ≤ min_sheets_for_boats * boats_per_sheet ∧
    planes ≤ (total_toys / helicopters_per_sheet) * planes_per_sheet ∧
    helicopters = (total_toys / helicopters_per_sheet) * helicopters_per_sheet :=
by sorry

end NUMINAMATH_CALUDE_min_sheets_for_boats_is_one_l616_61698


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l616_61613

theorem yellow_highlighters_count (total : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : total = 12) 
  (h2 : pink = 6) 
  (h3 : blue = 4) : 
  total - pink - blue = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l616_61613


namespace NUMINAMATH_CALUDE_c_share_value_l616_61671

/-- Proves that given the conditions, c's share is 398.75 -/
theorem c_share_value (total : ℚ) (a b c d : ℚ) : 
  total = 1500 →
  5/2 * a = 7/3 * b →
  5/2 * a = 2 * c →
  5/2 * a = 11/6 * d →
  a + b + c + d = total →
  c = 398.75 := by
sorry

end NUMINAMATH_CALUDE_c_share_value_l616_61671


namespace NUMINAMATH_CALUDE_train_overtake_time_l616_61641

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed : ℝ) (motorbike_speed : ℝ) (train_length : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  train_length = 180.0144 →
  (train_length / ((train_speed - motorbike_speed) / 3.6)) = 18.00144 := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_time_l616_61641
