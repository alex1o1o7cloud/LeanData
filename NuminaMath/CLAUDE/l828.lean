import Mathlib

namespace NUMINAMATH_CALUDE_pop_survey_result_l828_82800

theorem pop_survey_result (total_surveyed : ℕ) (pop_angle : ℕ) (people_chose_pop : ℕ) : 
  total_surveyed = 540 → pop_angle = 270 → people_chose_pop = total_surveyed * pop_angle / 360 →
  people_chose_pop = 405 := by
sorry

end NUMINAMATH_CALUDE_pop_survey_result_l828_82800


namespace NUMINAMATH_CALUDE_combined_boys_avg_is_correct_l828_82890

/-- Represents a high school with exam scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools -/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_girls_avg : ℝ

/-- Calculates the combined average score for boys given two schools' data -/
def combined_boys_avg (schools : CombinedSchools) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that the combined boys' average is approximately 48.57 -/
theorem combined_boys_avg_is_correct (schools : CombinedSchools) 
  (h1 : schools.school1 = ⟨68, 72, 70⟩)
  (h2 : schools.school2 = ⟨74, 88, 82⟩)
  (h3 : schools.combined_girls_avg = 83) :
  abs (combined_boys_avg schools - 48.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_combined_boys_avg_is_correct_l828_82890


namespace NUMINAMATH_CALUDE_no_intersection_eq_two_five_l828_82863

theorem no_intersection_eq_two_five : ¬∃ a : ℝ, 
  ({2, 4, a^3 - 2*a^2 - a + 7} : Set ℝ) ∩ 
  ({1, 5*a - 5, -1/2*a^2 + 3/2*a + 4, a^3 + a^2 + 3*a + 7} : Set ℝ) = 
  ({2, 5} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_no_intersection_eq_two_five_l828_82863


namespace NUMINAMATH_CALUDE_sum_of_even_integers_202_to_300_l828_82874

def sum_of_first_n_even_integers (n : ℕ) : ℕ := n * (n + 1)

def count_even_numbers_in_range (first last : ℕ) : ℕ :=
  (last - first) / 2 + 1

def sum_of_arithmetic_sequence (n first last : ℕ) : ℕ :=
  n / 2 * (first + last)

theorem sum_of_even_integers_202_to_300 
  (h : sum_of_first_n_even_integers 50 = 2550) :
  sum_of_arithmetic_sequence 
    (count_even_numbers_in_range 202 300) 
    202 
    300 = 12550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_202_to_300_l828_82874


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l828_82870

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 6*x ∧ 6*s^2 = 2*x) → x = 1/972 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l828_82870


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l828_82882

theorem integral_x_squared_plus_sin_x : ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l828_82882


namespace NUMINAMATH_CALUDE_teacher_student_relationship_l828_82847

/-- In a school system, prove the relationship between teachers and students -/
theorem teacher_student_relationship (m n k l : ℕ) 
  (h1 : m > 0) -- Ensure there's at least one teacher
  (h2 : n > 0) -- Ensure there's at least one student
  (h3 : k > 0) -- Each teacher has at least one student
  (h4 : l > 0) -- Each student has at least one teacher
  (h5 : ∀ t, t ≤ m → (∃ s, s = k)) -- Each teacher has exactly k students
  (h6 : ∀ s, s ≤ n → (∃ t, t = l)) -- Each student has exactly l teachers
  : m * k = n * l := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_relationship_l828_82847


namespace NUMINAMATH_CALUDE_ln_squared_plus_ln_inequality_l828_82881

theorem ln_squared_plus_ln_inequality (x : ℝ) :
  x > 0 → (Real.log x ^ 2 + Real.log x < 0 ↔ Real.exp (-1) < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_ln_squared_plus_ln_inequality_l828_82881


namespace NUMINAMATH_CALUDE_initial_shoe_pairs_l828_82842

/-- 
Given that a person loses 9 individual shoes and is left with a maximum of 20 matching pairs,
prove that the initial number of pairs of shoes was 25.
-/
theorem initial_shoe_pairs (lost_shoes : ℕ) (max_pairs_left : ℕ) : 
  lost_shoes = 9 →
  max_pairs_left = 20 →
  ∃ (initial_pairs : ℕ), initial_pairs = 25 ∧ 
    initial_pairs * 2 = max_pairs_left * 2 + lost_shoes :=
by sorry


end NUMINAMATH_CALUDE_initial_shoe_pairs_l828_82842


namespace NUMINAMATH_CALUDE_areas_sum_equal_largest_l828_82814

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  -- The sides of the triangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  -- The areas of the non-triangular regions
  area_D : ℝ
  area_E : ℝ
  area_F : ℝ
  -- Conditions
  isosceles : side1 = side2
  sides : side1 = 12 ∧ side2 = 12 ∧ side3 = 20
  largest_F : area_F ≥ area_D ∧ area_F ≥ area_E

/-- Theorem stating that D + E = F for the given inscribed triangle -/
theorem areas_sum_equal_largest (t : InscribedTriangle) : t.area_D + t.area_E = t.area_F := by
  sorry

end NUMINAMATH_CALUDE_areas_sum_equal_largest_l828_82814


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l828_82807

/-- Given a purchase with a total cost, tax rate, and cost of tax-free items,
    calculate the percentage of the total cost that went on sales tax. -/
theorem sales_tax_percentage
  (total_cost : ℝ)
  (tax_rate : ℝ)
  (tax_free_cost : ℝ)
  (h1 : total_cost = 20)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 14.7) :
  (tax_rate * (total_cost - tax_free_cost)) / total_cost * 100 = 1.59 := by
sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l828_82807


namespace NUMINAMATH_CALUDE_exist_n_points_with_integer_distances_l828_82852

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Main theorem statement -/
theorem exist_n_points_with_integer_distances (n : ℕ) (h : n ≥ 2) :
  ∃ (points : Fin n → Point),
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬areCollinear (points i) (points j) (points k)) ∧
    (∀ (i j : Fin n), i ≠ j → ∃ (d : ℤ), squaredDistance (points i) (points j) = d^2) :=
by sorry

end NUMINAMATH_CALUDE_exist_n_points_with_integer_distances_l828_82852


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l828_82860

theorem complex_modulus_problem (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : 
  Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l828_82860


namespace NUMINAMATH_CALUDE_triangle_roots_range_l828_82826

theorem triangle_roots_range (m : ℝ) : 
  (∃ x y z : ℝ, (x - 1) * (x^2 - 2*x + m) = 0 ∧ 
                (y - 1) * (y^2 - 2*y + m) = 0 ∧ 
                (z - 1) * (z^2 - 2*z + m) = 0 ∧
                x + y > z ∧ y + z > x ∧ z + x > y) ↔ 
  (3/4 < m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_roots_range_l828_82826


namespace NUMINAMATH_CALUDE_smallest_fraction_l828_82844

theorem smallest_fraction (x : ℝ) (h : x = 5) : 
  min (min (min (min (8/x) (8/(x+1))) (8/(x-1))) (x/8)) ((x+1)/8) = x/8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l828_82844


namespace NUMINAMATH_CALUDE_shoes_to_belts_ratio_l828_82895

def number_of_hats : ℕ := 5
def number_of_shoes : ℕ := 14
def belt_hat_difference : ℕ := 2

def number_of_belts : ℕ := number_of_hats + belt_hat_difference

theorem shoes_to_belts_ratio :
  (number_of_shoes : ℚ) / (number_of_belts : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_shoes_to_belts_ratio_l828_82895


namespace NUMINAMATH_CALUDE_circle_ellipse_ratio_l828_82899

/-- A circle with equation x^2 + (y+1)^2 = n -/
structure Circle where
  n : ℝ

/-- An ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse where
  m : ℝ

/-- The theorem stating that for a circle C and an ellipse M satisfying certain conditions,
    the ratio of n/m equals 8 -/
theorem circle_ellipse_ratio (C : Circle) (M : Ellipse) 
  (h1 : C.n > 0) 
  (h2 : M.m > 0) 
  (h3 : ∃ (x y : ℝ), x^2 + (y+1)^2 = C.n ∧ x^2 + M.m * y^2 = 1) 
  (h4 : ∃ (x y : ℝ), x^2 + y^2 = C.n ∧ x^2 + M.m * y^2 = 1) : 
  C.n / M.m = 8 := by
sorry

end NUMINAMATH_CALUDE_circle_ellipse_ratio_l828_82899


namespace NUMINAMATH_CALUDE_remainder_problem_l828_82853

theorem remainder_problem (n : ℤ) (h : n ≡ 16 [ZMOD 30]) : 2 * n ≡ 2 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l828_82853


namespace NUMINAMATH_CALUDE_fifth_score_proof_l828_82854

theorem fifth_score_proof (s1 s2 s3 s4 s5 : ℕ) : 
  s1 = 90 → s2 = 93 → s3 = 85 → s4 = 97 → 
  (s1 + s2 + s3 + s4 + s5) / 5 = 92 → 
  s5 = 95 := by
sorry

end NUMINAMATH_CALUDE_fifth_score_proof_l828_82854


namespace NUMINAMATH_CALUDE_new_train_distance_l828_82837

theorem new_train_distance (old_distance : ℝ) (percentage_increase : ℝ) : 
  old_distance = 300 → percentage_increase = 30 → 
  old_distance * (1 + percentage_increase / 100) = 390 := by
  sorry

end NUMINAMATH_CALUDE_new_train_distance_l828_82837


namespace NUMINAMATH_CALUDE_expand_product_l828_82897

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l828_82897


namespace NUMINAMATH_CALUDE_inequality_range_l828_82817

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l828_82817


namespace NUMINAMATH_CALUDE_four_lighthouses_cover_plane_l828_82898

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 90-degree angle in the plane -/
inductive Quadrant
  | NE
  | SE
  | SW
  | NW

/-- Represents a lighthouse with its position and illumination direction -/
structure Lighthouse where
  position : Point
  direction : Quadrant

/-- Checks if a point is illuminated by a lighthouse -/
def isIlluminated (p : Point) (l : Lighthouse) : Prop :=
  sorry

/-- The main theorem: four lighthouses can illuminate the entire plane -/
theorem four_lighthouses_cover_plane (a b c d : Point) :
  ∃ (la lb lc ld : Lighthouse),
    la.position = a ∧ lb.position = b ∧ lc.position = c ∧ ld.position = d ∧
    ∀ p : Point, isIlluminated p la ∨ isIlluminated p lb ∨ isIlluminated p lc ∨ isIlluminated p ld :=
  sorry


end NUMINAMATH_CALUDE_four_lighthouses_cover_plane_l828_82898


namespace NUMINAMATH_CALUDE_candy_division_l828_82887

theorem candy_division (total_candy : ℕ) (non_chocolate_candy : ℕ) 
  (chocolate_heart_bags : ℕ) (chocolate_kiss_bags : ℕ) :
  total_candy = 63 →
  non_chocolate_candy = 28 →
  chocolate_heart_bags = 2 →
  chocolate_kiss_bags = 3 →
  ∃ (pieces_per_bag : ℕ),
    pieces_per_bag > 0 ∧
    (total_candy - non_chocolate_candy) = 
      (chocolate_heart_bags + chocolate_kiss_bags) * pieces_per_bag ∧
    non_chocolate_candy % pieces_per_bag = 0 ∧
    chocolate_heart_bags + chocolate_kiss_bags + (non_chocolate_candy / pieces_per_bag) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_division_l828_82887


namespace NUMINAMATH_CALUDE_square_2023_position_l828_82822

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DABC
  | SquarePosition.DABC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the function to get the nth square position
def nthSquarePosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.DABC
  | 2 => SquarePosition.CBAD
  | 3 => SquarePosition.DCBA
  | _ => SquarePosition.ABCD -- This case is not actually possible

theorem square_2023_position : nthSquarePosition 2023 = SquarePosition.DABC := by
  sorry

end NUMINAMATH_CALUDE_square_2023_position_l828_82822


namespace NUMINAMATH_CALUDE_alice_bushes_l828_82886

/-- The number of bushes needed to cover three sides of a yard -/
def bushes_needed (side_length : ℕ) (sides : ℕ) (bush_coverage : ℕ) : ℕ :=
  (side_length * sides) / bush_coverage

/-- Theorem: Alice needs 12 bushes for her yard -/
theorem alice_bushes :
  bushes_needed 16 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_l828_82886


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l828_82824

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 8 = 0) →
  (b^3 - 15*b^2 + 25*b - 8 = 0) →
  (c^3 - 15*c^2 + 25*c - 8 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/9) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l828_82824


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l828_82848

/-- The probability of drawing 4 white balls from a box containing 7 white and 8 black balls -/
theorem probability_four_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 15 →
  white_balls = 7 →
  black_balls = 8 →
  drawn_balls = 4 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 39 :=
by sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l828_82848


namespace NUMINAMATH_CALUDE_perfect_match_production_l828_82831

theorem perfect_match_production (total_workers : ℕ) 
  (tables_per_worker : ℕ) (chairs_per_worker : ℕ) 
  (table_workers : ℕ) (chair_workers : ℕ) : 
  total_workers = 36 → 
  tables_per_worker = 20 → 
  chairs_per_worker = 50 → 
  table_workers = 20 → 
  chair_workers = 16 → 
  table_workers + chair_workers = total_workers → 
  2 * (table_workers * tables_per_worker) = chair_workers * chairs_per_worker :=
by
  sorry

#check perfect_match_production

end NUMINAMATH_CALUDE_perfect_match_production_l828_82831


namespace NUMINAMATH_CALUDE_total_cost_calculation_l828_82889

def beef_amount : ℕ := 1000
def beef_price : ℕ := 8
def chicken_amount : ℕ := 2 * beef_amount
def chicken_price : ℕ := 3

theorem total_cost_calculation :
  beef_amount * beef_price + chicken_amount * chicken_price = 14000 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l828_82889


namespace NUMINAMATH_CALUDE_tissue_length_l828_82867

/-- The total length of overlapped tissue pieces. -/
def totalLength (n : ℕ) (pieceLength : ℝ) (overlap : ℝ) : ℝ :=
  pieceLength + (n - 1 : ℝ) * (pieceLength - overlap)

/-- Theorem stating the total length of 30 pieces of tissue, each 25 cm long,
    overlapped by 6 cm, is 576 cm. -/
theorem tissue_length :
  totalLength 30 25 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_tissue_length_l828_82867


namespace NUMINAMATH_CALUDE_min_value_theorem_l828_82835

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 + x*y = 315) :
  x^2 + y^2 - x*y ≥ 105 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l828_82835


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l828_82846

theorem large_monkey_doll_cost (total_spent : ℚ) (price_difference : ℚ) (extra_dolls : ℕ) 
  (h1 : total_spent = 320)
  (h2 : price_difference = 4)
  (h3 : extra_dolls = 40)
  (h4 : total_spent / (large_cost - price_difference) = total_spent / large_cost + extra_dolls) :
  large_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l828_82846


namespace NUMINAMATH_CALUDE_solve_cake_baking_l828_82833

def cake_baking_problem (jane_rate roy_rate : ℚ) (jane_remaining_time : ℚ) (jane_remaining_work : ℚ) : Prop :=
  let combined_rate := jane_rate + roy_rate
  let total_work := 1
  ∃ t : ℚ, 
    t > 0 ∧
    combined_rate * t + jane_remaining_work = total_work ∧
    jane_rate * jane_remaining_time = jane_remaining_work ∧
    t = 2

theorem solve_cake_baking :
  cake_baking_problem (1/4) (1/5) (2/5) (1/10) :=
sorry

end NUMINAMATH_CALUDE_solve_cake_baking_l828_82833


namespace NUMINAMATH_CALUDE_quadratic_equation_factor_l828_82856

theorem quadratic_equation_factor (a : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_factor_l828_82856


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l828_82893

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 2) ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l828_82893


namespace NUMINAMATH_CALUDE_parabola_vertex_l828_82804

/-- The equation of a parabola in the xy-plane. -/
def ParabolaEquation (x y : ℝ) : Prop :=
  y^2 - 4*y + x + 7 = 0

/-- The vertex of a parabola. -/
def Vertex : ℝ × ℝ := (-3, 2)

/-- Theorem stating that the vertex of the parabola defined by y^2 - 4y + x + 7 = 0 is (-3, 2). -/
theorem parabola_vertex :
  ∀ x y : ℝ, ParabolaEquation x y → (x, y) = Vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l828_82804


namespace NUMINAMATH_CALUDE_complex_square_equality_l828_82865

theorem complex_square_equality (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 3 + 4 * Complex.I →
  a + b * Complex.I = 2 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l828_82865


namespace NUMINAMATH_CALUDE_total_is_300_l828_82820

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
def penny_throwing (rachelle gretchen rocky : ℕ) : Prop :=
  rachelle = 180 ∧ 
  gretchen = rachelle / 2 ∧ 
  rocky = gretchen / 3

/-- The total number of pennies thrown by all three -/
def total_pennies (rachelle gretchen rocky : ℕ) : ℕ :=
  rachelle + gretchen + rocky

/-- Theorem stating that the total number of pennies thrown is 300 -/
theorem total_is_300 (rachelle gretchen rocky : ℕ) : 
  penny_throwing rachelle gretchen rocky → total_pennies rachelle gretchen rocky = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_total_is_300_l828_82820


namespace NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l828_82880

structure Student where
  name : String
  variance : ℝ

def more_stable (a b : Student) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : Student) 
  (h_mean : a.variance ≠ b.variance) :
  more_stable a b ∨ more_stable b a := by
  sorry

-- Define the specific students from the problem
def student_A : Student := ⟨"A", 1.4⟩
def student_B : Student := ⟨"B", 2.5⟩

-- Theorem for the specific case in the problem
theorem A_more_stable_than_B : 
  more_stable student_A student_B := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l828_82880


namespace NUMINAMATH_CALUDE_uniform_price_l828_82873

def full_year_salary : ℕ := 500
def months_worked : ℕ := 9
def payment_received : ℕ := 300

theorem uniform_price : 
  ∃ (uniform_price : ℕ), 
    (uniform_price + payment_received = (months_worked * full_year_salary) / 12) ∧
    uniform_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_l828_82873


namespace NUMINAMATH_CALUDE_roundness_of_1764000_l828_82869

/-- The roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- Theorem: The roundness of 1,764,000 is 11. -/
theorem roundness_of_1764000 : roundness 1764000 = 11 := by sorry

end NUMINAMATH_CALUDE_roundness_of_1764000_l828_82869


namespace NUMINAMATH_CALUDE_floor_equality_l828_82816

theorem floor_equality (m : ℝ) (h : m ≥ 3) :
  ⌊m * (m + 1) / (2 * (2 * m - 1))⌋ = ⌊(m + 1) / 4⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_l828_82816


namespace NUMINAMATH_CALUDE_project_completion_time_l828_82808

/-- The number of days it takes for two workers to complete a job -/
structure WorkerPair :=
  (worker1 : ℕ)
  (worker2 : ℕ)
  (days : ℕ)

/-- The rate at which a worker completes the job per day -/
def workerRate (days : ℕ) : ℚ :=
  1 / days

theorem project_completion_time 
  (ab : WorkerPair) 
  (bc : WorkerPair) 
  (c_alone : ℕ) 
  (a_days : ℕ) 
  (b_days : ℕ) :
  ab.days = 10 →
  bc.days = 18 →
  c_alone = 45 →
  a_days = 5 →
  b_days = 10 →
  ∃ (c_days : ℕ), c_days = 15 ∧ 
    (workerRate ab.days * a_days + 
     workerRate ab.days * b_days + 
     workerRate c_alone * c_days = 1) :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l828_82808


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l828_82806

theorem basic_astrophysics_degrees (total_degrees : ℝ) 
  (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants : ℝ) :
  total_degrees = 360 →
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  gm_microorganisms = 29 →
  industrial_lubricants = 8 →
  let other_sectors := microphotonics + home_electronics + food_additives + gm_microorganisms + industrial_lubricants
  let basic_astrophysics_percent := 100 - other_sectors
  let basic_astrophysics_degrees := (basic_astrophysics_percent / 100) * total_degrees
  basic_astrophysics_degrees = 54 :=
by sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l828_82806


namespace NUMINAMATH_CALUDE_time_to_find_artifacts_is_120_months_l828_82868

/-- The time taken to find two artifacts given research and expedition times for the first artifact,
    and a multiplier for the second artifact's time. -/
def time_to_find_artifacts (research_time_1 : ℕ) (expedition_time_1 : ℕ) (multiplier : ℕ) : ℕ :=
  let first_artifact_time := research_time_1 + expedition_time_1
  let second_artifact_time := multiplier * first_artifact_time
  first_artifact_time + second_artifact_time

/-- Theorem stating that the time to find both artifacts is 120 months. -/
theorem time_to_find_artifacts_is_120_months :
  time_to_find_artifacts 6 24 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_time_to_find_artifacts_is_120_months_l828_82868


namespace NUMINAMATH_CALUDE_work_completion_time_l828_82879

/-- Represents the time it takes for a worker to complete a task alone -/
structure WorkTime where
  days : ℝ
  work_rate : ℝ
  inv_days_eq_work_rate : work_rate = 1 / days

/-- Represents a work scenario with two workers -/
structure WorkScenario where
  x : WorkTime
  y : WorkTime
  total_days : ℝ
  x_solo_days : ℝ
  both_days : ℝ
  total_days_eq_sum : total_days = x_solo_days + both_days

/-- The theorem to be proved -/
theorem work_completion_time (w : WorkScenario) (h1 : w.y.days = 12) 
  (h2 : w.x_solo_days = 4) (h3 : w.total_days = 10) : w.x.days = 20 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l828_82879


namespace NUMINAMATH_CALUDE_beach_trip_time_difference_l828_82830

theorem beach_trip_time_difference (bus_time car_round_trip : ℕ) : 
  bus_time = 40 → car_round_trip = 70 → bus_time - car_round_trip / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_time_difference_l828_82830


namespace NUMINAMATH_CALUDE_gcf_of_72_and_90_l828_82859

theorem gcf_of_72_and_90 : Nat.gcd 72 90 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_72_and_90_l828_82859


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l828_82801

theorem quadratic_coefficient_sum (a k n : ℤ) : 
  (∀ x : ℤ, (3*x + 2)*(2*x - 7) = a*x^2 + k*x + n) → 
  a - n + k = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l828_82801


namespace NUMINAMATH_CALUDE_final_state_correct_l828_82872

/-- Represents the state of variables A, B, and C -/
structure State :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)

/-- Executes the assignment statements and returns the final state -/
def executeAssignments : State := by
  let s1 : State := { A := 0, B := 0, C := 2 }  -- C ← 2
  let s2 : State := { A := s1.A, B := 1, C := s1.C }  -- B ← 1
  let s3 : State := { A := 2, B := s2.B, C := s2.C }  -- A ← 2
  exact s3

/-- Theorem stating that the final values of A, B, and C are 2, 1, and 2 respectively -/
theorem final_state_correct : 
  let final := executeAssignments
  final.A = 2 ∧ final.B = 1 ∧ final.C = 2 := by
  sorry

end NUMINAMATH_CALUDE_final_state_correct_l828_82872


namespace NUMINAMATH_CALUDE_pauls_and_sarahs_ages_l828_82803

theorem pauls_and_sarahs_ages (p s : ℕ) : 
  p = s + 8 →                   -- Paul is eight years older than Sarah
  p + 6 = 3 * (s - 2) →         -- In six years, Paul will be three times as old as Sarah was two years ago
  p + s = 28                    -- The sum of their current ages is 28
  := by sorry

end NUMINAMATH_CALUDE_pauls_and_sarahs_ages_l828_82803


namespace NUMINAMATH_CALUDE_cubic_minus_linear_l828_82884

theorem cubic_minus_linear (n : ℕ) : ∃ n : ℕ, n^3 - n = 5814 :=
by
  -- We need to prove that there exists a natural number n such that n^3 - n = 5814
  -- given that n^3 - n is even and is the product of three consecutive natural numbers
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_l828_82884


namespace NUMINAMATH_CALUDE_three_roots_symmetric_about_two_l828_82896

/-- A function f: ℝ → ℝ that satisfies f(2+x) = f(2-x) for all x ∈ ℝ -/
def symmetric_about_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

/-- The set of roots of f -/
def roots (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = 0}

theorem three_roots_symmetric_about_two (f : ℝ → ℝ) :
  symmetric_about_two f →
  (∃ a b : ℝ, roots f = {0, a, b} ∧ a ≠ b ∧ a ≠ 0 ∧ b ≠ 0) →
  roots f = {0, 2, 4} :=
sorry

end NUMINAMATH_CALUDE_three_roots_symmetric_about_two_l828_82896


namespace NUMINAMATH_CALUDE_cylinder_packing_l828_82828

theorem cylinder_packing (n : ℕ) (d : ℝ) (h : d > 0) :
  let rectangular_width := 8 * d
  let hexagonal_width := n * d * (Real.sqrt 3 / 2) + d
  40 < n → n < 42 →
  hexagonal_width < rectangular_width ∧
  hexagonal_width > rectangular_width - d :=
by sorry

end NUMINAMATH_CALUDE_cylinder_packing_l828_82828


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l828_82802

theorem triangle_side_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / b^2 ≥ (1 : ℝ) / 2 ∧ ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧ (a'^2 + c'^2) / b'^2 = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l828_82802


namespace NUMINAMATH_CALUDE_rodney_commission_l828_82810

/-- Rodney's commission for selling home security systems --/
def commission_per_sale : ℕ := 25

/-- Number of streets in the neighborhood --/
def num_streets : ℕ := 4

/-- Number of houses on each street --/
def houses_per_street : ℕ := 8

/-- Sales on the second street --/
def sales_second_street : ℕ := 4

/-- Sales on the first street (half of second street) --/
def sales_first_street : ℕ := sales_second_street / 2

/-- Sales on the third street (no sales) --/
def sales_third_street : ℕ := 0

/-- Sales on the fourth street --/
def sales_fourth_street : ℕ := 1

/-- Total sales across all streets --/
def total_sales : ℕ := sales_first_street + sales_second_street + sales_third_street + sales_fourth_street

/-- Rodney's total commission --/
def total_commission : ℕ := total_sales * commission_per_sale

theorem rodney_commission : total_commission = 175 := by
  sorry

end NUMINAMATH_CALUDE_rodney_commission_l828_82810


namespace NUMINAMATH_CALUDE_distance_is_sqrt_206_l828_82838

def point : ℝ × ℝ × ℝ := (2, 3, 1)

def line_point : ℝ × ℝ × ℝ := (8, 10, 12)

def line_direction : ℝ × ℝ × ℝ := (2, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_206 : 
  distance_to_line point line_point line_direction = Real.sqrt 206 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_206_l828_82838


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l828_82841

theorem complex_fraction_equality : (1 - Complex.I * Real.sqrt 3) / ((Real.sqrt 3 + Complex.I) ^ 2) = -1/4 - (Complex.I * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l828_82841


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_24_consecutive_integers_l828_82885

theorem smallest_perfect_square_sum_of_24_consecutive_integers :
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (∃ (k : ℕ), k * k = 12 * (2 * n + 23)) ∧
    (∀ (m : ℕ), m > 0 → m < n → 
      ¬∃ (j : ℕ), j * j = 12 * (2 * m + 23)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_24_consecutive_integers_l828_82885


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l828_82825

theorem quadratic_equation_conversion :
  ∀ x : ℝ, (x - 3)^2 = 4 ↔ x^2 - 6*x + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l828_82825


namespace NUMINAMATH_CALUDE_simplify_expression_l828_82812

theorem simplify_expression (x : ℝ) : (4*x)^4 + (5*x)*(x^3) = 261*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l828_82812


namespace NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l828_82876

/-- The number of axes of symmetry for a square -/
def square_symmetry_axes : ℕ := 4

/-- The number of axes of symmetry for an equilateral triangle -/
def equilateral_triangle_symmetry_axes : ℕ := 3

/-- The number of axes of symmetry for an isosceles triangle -/
def isosceles_triangle_symmetry_axes : ℕ := 1

/-- The number of axes of symmetry for an isosceles trapezoid -/
def isosceles_trapezoid_symmetry_axes : ℕ := 1

/-- The shape with the most axes of symmetry -/
def shape_with_most_symmetry_axes : ℕ := square_symmetry_axes

theorem square_has_most_symmetry_axes :
  shape_with_most_symmetry_axes = square_symmetry_axes ∧
  shape_with_most_symmetry_axes > equilateral_triangle_symmetry_axes ∧
  shape_with_most_symmetry_axes > isosceles_triangle_symmetry_axes ∧
  shape_with_most_symmetry_axes > isosceles_trapezoid_symmetry_axes :=
by sorry

end NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l828_82876


namespace NUMINAMATH_CALUDE_max_y_value_l828_82891

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l828_82891


namespace NUMINAMATH_CALUDE_arithmetic_problem_l828_82819

theorem arithmetic_problem : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l828_82819


namespace NUMINAMATH_CALUDE_vector_parallel_problem_l828_82836

/-- Two 2D vectors are parallel if their components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_parallel_problem (m : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-3, 0)
  parallel (2 • a + b) (a - m • b) →
  m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_problem_l828_82836


namespace NUMINAMATH_CALUDE_first_channel_ends_earlier_l828_82878

/-- Represents the runtime of a film on a TV channel with commercials -/
structure ChannelRuntime where
  segment_length : ℕ
  commercial_length : ℕ
  num_segments : ℕ

/-- Calculates the total runtime for a channel -/
def total_runtime (c : ChannelRuntime) : ℕ :=
  c.segment_length * c.num_segments + c.commercial_length * (c.num_segments - 1)

/-- The theorem to be proved -/
theorem first_channel_ends_earlier (film_length : ℕ) :
  ∃ (n : ℕ), 
    let channel1 := ChannelRuntime.mk 20 2 n
    let channel2 := ChannelRuntime.mk 10 1 (2 * n)
    film_length = 20 * n ∧ 
    film_length = 10 * (2 * n) ∧
    total_runtime channel1 < total_runtime channel2 := by
  sorry

end NUMINAMATH_CALUDE_first_channel_ends_earlier_l828_82878


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l828_82855

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 18 * x + 9 = (r * x + s)^2) → a = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l828_82855


namespace NUMINAMATH_CALUDE_construction_team_distance_l828_82888

/-- Calculates the total distance built by a construction team -/
def total_distance_built (days : ℕ) (rate : ℕ) : ℕ :=
  days * rate

/-- Proves that a construction team working for 5 days at 120 meters per day builds 600 meters -/
theorem construction_team_distance : total_distance_built 5 120 = 600 := by
  sorry

end NUMINAMATH_CALUDE_construction_team_distance_l828_82888


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l828_82850

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (3, 0) and tangent to the y-axis at (0, 2) -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ
  tangent_x : center.1 - semi_major_axis = 3
  tangent_y : center.2 - semi_minor_axis = 2

/-- The distance between the foci of the ellipse is 2√5 -/
theorem ellipse_foci_distance (e : Ellipse) : 
  Real.sqrt (4 * (e.semi_major_axis^2 - e.semi_minor_axis^2)) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l828_82850


namespace NUMINAMATH_CALUDE_reverse_product_inequality_l828_82862

/-- Reverses the digits and decimal point of a positive real number with finitely many decimal places -/
noncomputable def reverse (x : ℝ) : ℝ := sorry

/-- Predicate to check if a real number has finitely many decimal places -/
def has_finite_decimals (x : ℝ) : Prop := sorry

/-- The main theorem to be proved -/
theorem reverse_product_inequality {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (hfx : has_finite_decimals x) (hfy : has_finite_decimals y) : 
  reverse (x * y) ≤ 10 * reverse x * reverse y := by sorry

end NUMINAMATH_CALUDE_reverse_product_inequality_l828_82862


namespace NUMINAMATH_CALUDE_bakers_cake_inventory_l828_82851

/-- Baker's cake inventory problem -/
theorem bakers_cake_inventory (cakes_made cakes_bought cakes_sold : ℕ) :
  cakes_made = 8 →
  cakes_bought = 139 →
  cakes_sold = 145 →
  cakes_sold - cakes_bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cake_inventory_l828_82851


namespace NUMINAMATH_CALUDE_circle_equation_l828_82805

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let point : ℝ × ℝ := (-1, 3)
  let radius : ℝ := Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2)
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l828_82805


namespace NUMINAMATH_CALUDE_robot_race_track_length_l828_82843

/-- Represents the race between three robots A, B, and C --/
structure RobotRace where
  track_length : ℝ
  va : ℝ
  vb : ℝ
  vc : ℝ

/-- The conditions of the race --/
def race_conditions (race : RobotRace) : Prop :=
  race.track_length > 0 ∧
  race.va > 0 ∧ race.vb > 0 ∧ race.vc > 0 ∧
  race.track_length / race.va = (race.track_length - 1) / race.vb ∧
  race.track_length / race.va = (race.track_length - 2) / race.vc ∧
  race.track_length / race.vb = (race.track_length - 1.01) / race.vc

theorem robot_race_track_length (race : RobotRace) :
  race_conditions race → race.track_length = 101 := by
  sorry

#check robot_race_track_length

end NUMINAMATH_CALUDE_robot_race_track_length_l828_82843


namespace NUMINAMATH_CALUDE_inquisitive_tourist_ratio_l828_82823

/-- Represents a tour group with a number of people -/
structure TourGroup where
  people : ℕ

/-- Represents a day of tours -/
structure TourDay where
  groups : List TourGroup
  usualQuestionsPerTourist : ℕ
  totalQuestionsAnswered : ℕ
  inquisitiveTouristGroup : ℕ  -- Index of the group with the inquisitive tourist

def calculateRatio (day : TourDay) : ℚ :=
  let regularQuestions := day.groups.enum.foldl
    (fun acc (i, group) =>
      if i = day.inquisitiveTouristGroup
      then acc + (group.people - 1) * day.usualQuestionsPerTourist
      else acc + group.people * day.usualQuestionsPerTourist)
    0
  let inquisitiveQuestions := day.totalQuestionsAnswered - regularQuestions
  inquisitiveQuestions / day.usualQuestionsPerTourist

theorem inquisitive_tourist_ratio (day : TourDay)
  (h1 : day.groups = [⟨6⟩, ⟨11⟩, ⟨8⟩, ⟨7⟩])
  (h2 : day.usualQuestionsPerTourist = 2)
  (h3 : day.totalQuestionsAnswered = 68)
  (h4 : day.inquisitiveTouristGroup = 2)  -- 0-based index for the third group
  : calculateRatio day = 3 := by
  sorry

#eval calculateRatio {
  groups := [⟨6⟩, ⟨11⟩, ⟨8⟩, ⟨7⟩],
  usualQuestionsPerTourist := 2,
  totalQuestionsAnswered := 68,
  inquisitiveTouristGroup := 2
}

end NUMINAMATH_CALUDE_inquisitive_tourist_ratio_l828_82823


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_seventeen_l828_82845

def sum_of_last_two_digits (n : ℕ) : ℕ :=
  (n % 100) / 10 + n % 10

theorem sum_of_digits_of_seven_to_seventeen (n : ℕ) (h : n = (3 + 4)^17) :
  sum_of_last_two_digits n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_seventeen_l828_82845


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l828_82821

theorem sqrt_two_irrational :
  ∃ (x : ℝ), Irrational x ∧ (x = Real.sqrt 2) ∧
  (∀ y : ℝ, (y = 1/3 ∨ y = 3.1415 ∨ y = -5) → ¬Irrational y) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l828_82821


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l828_82857

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 4328) : 
  n + 7 = 544 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l828_82857


namespace NUMINAMATH_CALUDE_workshop_efficiency_l828_82864

theorem workshop_efficiency (x : ℝ) (h : x > 0) : 
  (3000 / x) - (3000 / (2.5 * x)) = (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_workshop_efficiency_l828_82864


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l828_82811

-- Define a quadratic polynomial P(x) = ax^2 + bx + c
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic polynomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- State the theorem
theorem quadratic_discriminant (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃! x, P a b c x = x - 2) →
  (∃! x, P a b c x = 1 - x/2) →
  discriminant a b c = -1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l828_82811


namespace NUMINAMATH_CALUDE_described_relationship_is_correlation_l828_82813

/-- Represents a variable in a statistical relationship -/
structure Variable where
  name : String
  is_independent : Bool

/-- Represents a relationship between two variables -/
structure Relationship where
  x : Variable
  y : Variable
  is_uncertain : Bool
  y_has_randomness : Bool

/-- Defines what a correlation is -/
def is_correlation (r : Relationship) : Prop :=
  r.x.is_independent ∧ 
  ¬r.y.is_independent ∧ 
  r.is_uncertain ∧ 
  r.y_has_randomness

/-- Theorem stating that the described relationship is a correlation -/
theorem described_relationship_is_correlation (x y : Variable) (r : Relationship) 
  (h1 : x.is_independent)
  (h2 : ¬y.is_independent)
  (h3 : r.x = x)
  (h4 : r.y = y)
  (h5 : r.is_uncertain)
  (h6 : r.y_has_randomness) :
  is_correlation r := by
  sorry


end NUMINAMATH_CALUDE_described_relationship_is_correlation_l828_82813


namespace NUMINAMATH_CALUDE_problem_solution_l828_82875

theorem problem_solution (x y : ℝ) 
  (eq1 : x + Real.sin y = 2010)
  (eq2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005)
  (h : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l828_82875


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_two_l828_82827

theorem divisibility_by_power_of_two (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_two_l828_82827


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l828_82894

-- Define the "graph number" type
def GraphNumber := ℝ × ℝ × ℝ

-- Define a function to get the graph number of a quadratic function
def getGraphNumber (a b c : ℝ) : GraphNumber :=
  (a, b, c)

-- Define a predicate for when a quadratic function intersects x-axis at one point
def intersectsXAxisOnce (a b c : ℝ) : Prop :=
  b ^ 2 - 4 * a * c = 0

theorem quadratic_function_properties :
  -- Part 1
  getGraphNumber (1/3) (-1) (-1) = (1/3, -1, -1) ∧
  -- Part 2
  ∀ m : ℝ, intersectsXAxisOnce m (m+1) (m+1) → (m = -1 ∨ m = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l828_82894


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l828_82861

/-- A geometric sequence with a_2 = 2 and a_4 = 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 2 = 2 ∧ a 4 = 4

/-- In a geometric sequence with a_2 = 2 and a_4 = 4, a_6 = 8 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h : geometric_sequence a) : a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l828_82861


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l828_82834

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x + 5| = 3 * x - 2 :=
by
  use 7/2
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l828_82834


namespace NUMINAMATH_CALUDE_range_of_a_l828_82866

def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0

def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - 4*a^2 ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, a ≥ 0 →
    (∀ x, ¬(p x) → ¬(q x a)) ∧
    (∃ x, ¬(p x) ∧ q x a)) →
  {a : ℝ | a ≥ 5/2} = {a : ℝ | ∀ x, ¬(p x) → ¬(q x a)} := by sorry

end NUMINAMATH_CALUDE_range_of_a_l828_82866


namespace NUMINAMATH_CALUDE_product_of_roots_eq_one_l828_82809

theorem product_of_roots_eq_one : 
  ∃ (r₁ r₂ : ℝ), r₁ * r₂ = 1 ∧ r₁^(2*Real.log r₁) = ℯ ∧ r₂^(2*Real.log r₂) = ℯ ∧
  ∀ (x : ℝ), x^(2*Real.log x) = ℯ → x = r₁ ∨ x = r₂ :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_eq_one_l828_82809


namespace NUMINAMATH_CALUDE_correct_batteries_in_toys_l828_82832

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys : ℕ := 15

/-- The number of batteries Tom used in his flashlights -/
def batteries_in_flashlights : ℕ := 2

/-- Theorem stating that the number of batteries in toys is correct -/
theorem correct_batteries_in_toys :
  batteries_in_toys = batteries_in_flashlights + 13 :=
by sorry

end NUMINAMATH_CALUDE_correct_batteries_in_toys_l828_82832


namespace NUMINAMATH_CALUDE_min_value_theorem_l828_82839

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x + 1 / x^6 ≥ 10 ∧ (9 * x + 1 / x^6 = 10 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l828_82839


namespace NUMINAMATH_CALUDE_equation_solution_l828_82858

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop :=
  (2*m - 6) * x^(|m| - 2) = m^2

-- State the theorem
theorem equation_solution :
  ∃ (m : ℝ), ∀ (x : ℝ), equation m x ↔ x = -3/4 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l828_82858


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l828_82815

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)
  ∀ x : ℝ, f x ≤ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 := by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l828_82815


namespace NUMINAMATH_CALUDE_parabola_line_intersection_bounds_l828_82871

/-- Parabola P with equation y = 2x^2 -/
def P : ℝ → ℝ := λ x => 2 * x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x => m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

/-- Theorem stating the existence of r and s, and their sum -/
theorem parabola_line_intersection_bounds :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_bounds_l828_82871


namespace NUMINAMATH_CALUDE_g_neg_two_equals_fifteen_l828_82892

theorem g_neg_two_equals_fifteen :
  let g : ℝ → ℝ := λ x ↦ x^2 - 4*x + 3
  g (-2) = 15 := by sorry

end NUMINAMATH_CALUDE_g_neg_two_equals_fifteen_l828_82892


namespace NUMINAMATH_CALUDE_pencils_leftover_l828_82829

theorem pencils_leftover : Int.mod 33333332 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_leftover_l828_82829


namespace NUMINAMATH_CALUDE_expression_value_l828_82849

theorem expression_value :
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 1
  (x^2 * y * z - x * y * z^2) = 6 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l828_82849


namespace NUMINAMATH_CALUDE_inequalities_hold_l828_82883

theorem inequalities_hold (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a^2 * b < b^2 * c) ∧ (a^2 * c < b^2 * c) ∧ (a^2 * b < a^2 * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l828_82883


namespace NUMINAMATH_CALUDE_divisibility_by_eight_and_nine_l828_82840

theorem divisibility_by_eight_and_nine (x y : Nat) : 
  x < 10 ∧ y < 10 →
  (1234 * 10 * x + 1234 * y) % 8 = 0 ∧ 
  (1234 * 10 * x + 1234 * y) % 9 = 0 ↔ 
  (x = 8 ∧ y = 0) ∨ (x = 0 ∧ y = 8) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_and_nine_l828_82840


namespace NUMINAMATH_CALUDE_debbie_tape_usage_l828_82877

/-- The amount of tape needed to pack boxes of different sizes --/
def total_tape_used (large_boxes medium_boxes small_boxes : ℕ) : ℕ :=
  let large_tape := 5 * large_boxes  -- 4 feet for sealing + 1 foot for label
  let medium_tape := 3 * medium_boxes  -- 2 feet for sealing + 1 foot for label
  let small_tape := 2 * small_boxes  -- 1 foot for sealing + 1 foot for label
  large_tape + medium_tape + small_tape

/-- Theorem stating that Debbie used 44 feet of tape --/
theorem debbie_tape_usage : total_tape_used 2 8 5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_debbie_tape_usage_l828_82877


namespace NUMINAMATH_CALUDE_value_preserving_interval_iff_m_in_M_l828_82818

/-- A function f has a value-preserving interval [a,b] if it is monotonic on [a,b]
    and its range on [a,b] is [a,b] -/
def has_value_preserving_interval (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
    (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

/-- The set of m values for which f(x) = x^2 - (1/2)x + m has a value-preserving interval -/
def M : Set ℝ :=
  {m | m ∈ Set.Icc (5/16) (9/16) ∪ Set.Icc (-11/16) (-7/16)}

/-- The main theorem stating the equivalence between the existence of a value-preserving interval
    and m being in the set M -/
theorem value_preserving_interval_iff_m_in_M :
  ∀ m : ℝ, has_value_preserving_interval (fun x => x^2 - (1/2)*x + m) ↔ m ∈ M :=
sorry

end NUMINAMATH_CALUDE_value_preserving_interval_iff_m_in_M_l828_82818
