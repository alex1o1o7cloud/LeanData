import Mathlib

namespace solve_equation_l1330_133009

theorem solve_equation (x : ℚ) :
  (2 / (x + 2) + 4 / (x + 2) + (2 * x) / (x + 2) = 5) → x = -4/3 := by
  sorry

end solve_equation_l1330_133009


namespace modified_sequence_last_term_l1330_133031

def sequence_rule (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | i + 1 => 
    let prev := sequence_rule n i
    if prev < 10 then
      2 * prev
    else
      (prev % 10) + 5

def modified_sequence (n : ℕ) (m : ℕ) : ℕ → ℕ
  | i => if i = 99 then sequence_rule n i + m else sequence_rule n i

theorem modified_sequence_last_term (n : ℕ) :
  ∃ m : ℕ, m < 10 ∧ modified_sequence 2012 m 2011 = 5 → m = 8 := by
  sorry

end modified_sequence_last_term_l1330_133031


namespace vector_operation_l1330_133046

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

theorem vector_operation :
  (2 : ℝ) • a - b = (5, 7) := by sorry

end vector_operation_l1330_133046


namespace divisibility_pairs_l1330_133039

theorem divisibility_pairs : 
  {p : ℕ × ℕ | (p.1 + 1) % p.2 = 0 ∧ (p.2^2 - p.2 + 1) % p.1 = 0} = 
  {(1, 1), (1, 2), (3, 2)} := by
sorry

end divisibility_pairs_l1330_133039


namespace arithmetic_sequence_150th_term_l1330_133089

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

/-- The 150th term of the specific arithmetic sequence -/
def term_150 : ℝ :=
  arithmetic_sequence 3 5 150

theorem arithmetic_sequence_150th_term :
  term_150 = 748 := by sorry

end arithmetic_sequence_150th_term_l1330_133089


namespace largest_k_for_inequality_l1330_133099

theorem largest_k_for_inequality : 
  (∃ (k : ℝ), ∀ (x : ℝ), (1 + Real.sin x) / (2 + Real.cos x) ≥ k) ∧ 
  (∀ (k : ℝ), k > 4/3 → ¬(∃ (x : ℝ), (1 + Real.sin x) / (2 + Real.cos x) ≥ k)) :=
sorry

end largest_k_for_inequality_l1330_133099


namespace f_decreasing_interval_l1330_133011

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem f_decreasing_interval :
  ∀ x y : ℝ, x < y → y ≤ 1 → f y ≤ f x := by
sorry

end f_decreasing_interval_l1330_133011


namespace three_cubes_exposed_faces_sixty_cubes_exposed_faces_l1330_133038

/-- The number of exposed faces for n cubes in a row on a table -/
def exposed_faces (n : ℕ) : ℕ := 3 * n + 2

/-- Theorem stating that for 3 cubes, there are 11 exposed faces -/
theorem three_cubes_exposed_faces : exposed_faces 3 = 11 := by sorry

/-- Theorem to prove the number of exposed faces for 60 cubes -/
theorem sixty_cubes_exposed_faces : exposed_faces 60 = 182 := by sorry

end three_cubes_exposed_faces_sixty_cubes_exposed_faces_l1330_133038


namespace parallel_vectors_y_value_l1330_133052

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (2, y)
  parallel a b → y = 1 := by
  sorry

end parallel_vectors_y_value_l1330_133052


namespace average_existence_l1330_133048

theorem average_existence : ∃ N : ℝ, 12 < N ∧ N < 18 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end average_existence_l1330_133048


namespace hyperbolas_same_asymptotes_l1330_133070

/-- Given two hyperbolas with the same asymptotes, prove that M = 576/25 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, y^2/16 - x^2/25 = 1 ↔ x^2/36 - y^2/M = 1) → 
  (∀ x y : ℝ, y = (4/5)*x ↔ y = (Real.sqrt M / 6)*x) → 
  M = 576/25 := by
  sorry

end hyperbolas_same_asymptotes_l1330_133070


namespace fifth_term_of_geometric_sequence_l1330_133047

/-- Given a geometric sequence with 7 terms, where the first term is 8 and the last term is 5832,
    prove that the fifth term is 648. -/
theorem fifth_term_of_geometric_sequence (a : Fin 7 → ℝ) :
  (∀ i j, a (i + 1) / a i = a (j + 1) / a j) →  -- geometric sequence condition
  a 0 = 8 →                                     -- first term is 8
  a 6 = 5832 →                                  -- last term is 5832
  a 4 = 648 := by                               -- fifth term (index 4) is 648
sorry


end fifth_term_of_geometric_sequence_l1330_133047


namespace partial_fraction_decomposition_l1330_133036

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 4*x + 8) / ((x - 1)*(x - 4)*(x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
      P = 1/3 ∧ Q = -4/3 ∧ R = 2 :=
by sorry

end partial_fraction_decomposition_l1330_133036


namespace polynomial_determination_l1330_133013

theorem polynomial_determination (p : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →  -- p is quadratic
  p 3 = 0 →                                        -- p(3) = 0
  p (-1) = 0 →                                     -- p(-1) = 0
  p 2 = 10 →                                       -- p(2) = 10
  ∀ x, p x = -10/3 * x^2 + 20/3 * x + 10 :=        -- conclusion
by sorry

end polynomial_determination_l1330_133013


namespace feb_1_is_sunday_l1330_133041

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

-- Define a function to get the day n days before a given day
def daysBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => daysBefore (prevDay d) n

-- Theorem statement
theorem feb_1_is_sunday (h : DayOfWeek.Saturday = daysBefore DayOfWeek.Saturday 13) :
  DayOfWeek.Sunday = daysBefore DayOfWeek.Saturday 13 :=
by sorry

end feb_1_is_sunday_l1330_133041


namespace isosceles_triangle_leg_length_l1330_133053

/-- An isosceles triangle with given perimeter and base -/
structure IsoscelesTriangle where
  perimeter : ℝ
  base : ℝ
  legs_equal : ℝ
  perimeter_eq : perimeter = 2 * legs_equal + base

/-- Theorem: In an isosceles triangle with perimeter 26 cm and base 11 cm, each leg is 7.5 cm -/
theorem isosceles_triangle_leg_length 
  (triangle : IsoscelesTriangle) 
  (h_perimeter : triangle.perimeter = 26) 
  (h_base : triangle.base = 11) : 
  triangle.legs_equal = 7.5 := by
sorry


end isosceles_triangle_leg_length_l1330_133053


namespace intersection_M_N_l1330_133087

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x / (x - 1) ≤ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end intersection_M_N_l1330_133087


namespace shopping_tax_theorem_l1330_133024

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingPercent : ℝ) (foodPercent : ℝ) (otherPercent : ℝ)
                       (clothingTaxRate : ℝ) (foodTaxRate : ℝ) (otherTaxRate : ℝ) : ℝ :=
  clothingPercent * clothingTaxRate + foodPercent * foodTaxRate + otherPercent * otherTaxRate

theorem shopping_tax_theorem :
  totalTaxPercentage 0.4 0.3 0.3 0.04 0 0.08 = 0.04 := by
  sorry

end shopping_tax_theorem_l1330_133024


namespace train_cars_count_l1330_133088

/-- Represents a train with a consistent speed --/
structure Train where
  cars_per_12_seconds : ℕ
  total_passing_time : ℕ

/-- Calculates the total number of cars in the train --/
def total_cars (t : Train) : ℕ :=
  (t.cars_per_12_seconds * t.total_passing_time) / 12

/-- Theorem stating that a train with 8 cars passing in 12 seconds 
    and taking 210 seconds to pass has 140 cars --/
theorem train_cars_count :
  ∀ (t : Train), t.cars_per_12_seconds = 8 ∧ t.total_passing_time = 210 → 
  total_cars t = 140 := by
  sorry

end train_cars_count_l1330_133088


namespace mean_of_combined_sets_l1330_133092

theorem mean_of_combined_sets (set1_count : Nat) (set1_mean : ℚ) (set2_count : Nat) (set2_mean : ℚ) 
  (h1 : set1_count = 7)
  (h2 : set1_mean = 15)
  (h3 : set2_count = 8)
  (h4 : set2_mean = 18) :
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  total_sum / total_count = 249 / 15 := by
  sorry

end mean_of_combined_sets_l1330_133092


namespace odd_power_decomposition_l1330_133016

theorem odd_power_decomposition (m : ℤ) : 
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end odd_power_decomposition_l1330_133016


namespace circle_radius_proof_l1330_133055

theorem circle_radius_proof (chord_length : ℝ) (center_to_intersection : ℝ) (ratio_left : ℝ) (ratio_right : ℝ) :
  chord_length = 18 →
  center_to_intersection = 7 →
  ratio_left = 2 * ratio_right →
  ratio_left + ratio_right = chord_length →
  ∃ (radius : ℝ), radius = 11 ∧ 
    (radius - center_to_intersection) * (radius + center_to_intersection) = ratio_left * ratio_right :=
by sorry

end circle_radius_proof_l1330_133055


namespace percentage_in_70_79_is_one_third_l1330_133093

/-- Represents the frequency distribution of test scores -/
def score_distribution : List (String × ℕ) :=
  [("90% - 100%", 3),
   ("80% - 89%", 5),
   ("70% - 79%", 8),
   ("60% - 69%", 4),
   ("50% - 59%", 1),
   ("Below 50%", 3)]

/-- Total number of students in the class -/
def total_students : ℕ := (score_distribution.map (λ x => x.2)).sum

/-- Number of students who scored in the 70%-79% range -/
def students_in_70_79 : ℕ := 
  (score_distribution.filter (λ x => x.1 = "70% - 79%")).map (λ x => x.2) |>.sum

/-- Theorem stating that the percentage of students who scored in the 70%-79% range is 1/3 of the class -/
theorem percentage_in_70_79_is_one_third :
  (students_in_70_79 : ℚ) / (total_students : ℚ) = 1 / 3 := by
  sorry

end percentage_in_70_79_is_one_third_l1330_133093


namespace curve_is_two_intersecting_lines_l1330_133006

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  2 * x^2 - y^2 - 4 * x - 4 * y - 2 = 0

/-- The first line equation derived from the curve equation -/
def line1 (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x - Real.sqrt 2 - 2

/-- The second line equation derived from the curve equation -/
def line2 (x y : ℝ) : Prop :=
  y = -Real.sqrt 2 * x + Real.sqrt 2 - 2

/-- Theorem stating that the curve equation represents two intersecting lines -/
theorem curve_is_two_intersecting_lines :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (∀ x y, curve_equation x y ↔ (line1 x y ∨ line2 x y)) ∧ 
    (line1 x₁ y₁ ∧ line2 x₁ y₁) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
  sorry

end curve_is_two_intersecting_lines_l1330_133006


namespace two_x_less_than_one_necessary_not_sufficient_l1330_133030

theorem two_x_less_than_one_necessary_not_sufficient :
  (∀ x : ℝ, -1 < x ∧ x < 0 → 2*x < 1) ∧
  (∃ x : ℝ, 2*x < 1 ∧ ¬(-1 < x ∧ x < 0)) :=
by sorry

end two_x_less_than_one_necessary_not_sufficient_l1330_133030


namespace smallest_number_with_digit_sum_2017_properties_l1330_133014

/-- The smallest natural number with digit sum 2017 -/
def smallest_number_with_digit_sum_2017 : ℕ :=
  1 * 10^224 + (10^224 - 1)

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  sorry

theorem smallest_number_with_digit_sum_2017_properties :
  digit_sum smallest_number_with_digit_sum_2017 = 2017 ∧
  num_digits smallest_number_with_digit_sum_2017 = 225 ∧
  smallest_number_with_digit_sum_2017 / 10^224 = 1 ∧
  ∀ m : ℕ, m < smallest_number_with_digit_sum_2017 → digit_sum m ≠ 2017 :=
by sorry

end smallest_number_with_digit_sum_2017_properties_l1330_133014


namespace digging_time_for_second_hole_l1330_133076

/-- Proves that given the conditions of the digging problem, the time required to dig the second hole is 6 hours -/
theorem digging_time_for_second_hole 
  (workers_first : ℕ) 
  (hours_first : ℕ) 
  (depth_first : ℕ) 
  (extra_workers : ℕ) 
  (depth_second : ℕ) 
  (h : workers_first = 45)
  (i : hours_first = 8)
  (j : depth_first = 30)
  (k : extra_workers = 65)
  (l : depth_second = 55) :
  (workers_first + extra_workers) * (660 / (workers_first + extra_workers) : ℚ) * depth_second = 
  workers_first * hours_first * depth_second := by
sorry

#eval (45 + 65) * (660 / (45 + 65) : ℚ)

end digging_time_for_second_hole_l1330_133076


namespace f_two_zeros_implies_k_nonneg_l1330_133079

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x / (x - 2) + k * x^2 else Real.log x

theorem f_two_zeros_implies_k_nonneg (k : ℝ) :
  (∃ x y, x ≠ y ∧ f k x = 0 ∧ f k y = 0 ∧ ∀ z, f k z = 0 → z = x ∨ z = y) →
  k ≥ 0 := by
  sorry

end f_two_zeros_implies_k_nonneg_l1330_133079


namespace abc_is_zero_l1330_133083

theorem abc_is_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) :
  a * b * c = 0 := by
sorry

end abc_is_zero_l1330_133083


namespace division_remainder_and_divisibility_l1330_133027

theorem division_remainder_and_divisibility : 
  let n : ℕ := 1234567
  let d : ℕ := 256
  let r : ℕ := n % d
  (r = 2) ∧ (r % 7 ≠ 0) := by sorry

end division_remainder_and_divisibility_l1330_133027


namespace existence_of_special_polynomial_l1330_133086

theorem existence_of_special_polynomial :
  ∃ (P : Polynomial ℝ), 
    (∃ (i : ℕ), (P.coeff i < 0)) ∧ 
    (∀ (n : ℕ), n > 1 → ∀ (j : ℕ), ((P^n).coeff j > 0)) := by
  sorry

end existence_of_special_polynomial_l1330_133086


namespace x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1330_133010

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) :=
by sorry

end x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1330_133010


namespace no_real_solutions_log_equation_l1330_133067

theorem no_real_solutions_log_equation :
  ¬∃ (x : ℝ), (x + 3 > 0 ∧ x - 1 > 0 ∧ x^2 - 2*x - 3 > 0) ∧
  (Real.log (x + 3) + Real.log (x - 1) = Real.log (x^2 - 2*x - 3)) := by
  sorry

end no_real_solutions_log_equation_l1330_133067


namespace cube_has_twelve_edges_l1330_133080

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  -- We don't need to specify any fields for this simple definition

/-- The number of edges in a cube. -/
def num_edges (c : Cube) : ℕ := 12

/-- Theorem: A cube has 12 edges. -/
theorem cube_has_twelve_edges (c : Cube) : num_edges c = 12 := by
  sorry

end cube_has_twelve_edges_l1330_133080


namespace arithmetic_geometric_sequence_ratio_l1330_133040

/-- Given an arithmetic sequence {a_n} where a_n ≠ 0 for all n,
    if a_1, a_3, and a_4 form a geometric sequence,
    then the common ratio of this geometric sequence is either 1 or 1/2. -/
theorem arithmetic_geometric_sequence_ratio
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arith : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)  -- Arithmetic sequence condition
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)  -- Non-zero condition
  (h_geom : ∃ q : ℝ, a 3 = a 1 * q ∧ a 4 = a 3 * q)  -- Geometric sequence condition
  : ∃ q : ℝ, (q = 1 ∨ q = 1/2) ∧ a 3 = a 1 * q ∧ a 4 = a 3 * q :=
sorry

end arithmetic_geometric_sequence_ratio_l1330_133040


namespace watercolor_painting_distribution_l1330_133000

theorem watercolor_painting_distribution (total_paintings : ℕ) (paintings_per_room : ℕ) (num_rooms : ℕ) : 
  total_paintings = 32 → paintings_per_room = 8 → num_rooms * paintings_per_room = total_paintings → num_rooms = 4 := by
  sorry

end watercolor_painting_distribution_l1330_133000


namespace convex_polygon_perimeter_bound_l1330_133062

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool  -- We simplify the convexity check to a boolean for this statement

/-- A square in 2D space -/
structure Square where
  center : Real × Real
  side_length : Real

/-- Check if a point is inside or on the boundary of a square -/
def point_in_square (p : Real × Real) (s : Square) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  let half_side := s.side_length / 2
  x ≥ cx - half_side ∧ x ≤ cx + half_side ∧
  y ≥ cy - half_side ∧ y ≤ cy + half_side

/-- Check if a polygon is contained in a square -/
def polygon_in_square (p : ConvexPolygon) (s : Square) : Prop :=
  ∀ v ∈ p.vertices, point_in_square v s

/-- Calculate the perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : Real :=
  sorry  -- The actual calculation is omitted for brevity

/-- The main theorem -/
theorem convex_polygon_perimeter_bound (p : ConvexPolygon) (s : Square) :
  p.is_convex = true →
  s.side_length = 1 →
  polygon_in_square p s →
  perimeter p ≤ 4 := by
  sorry

end convex_polygon_perimeter_bound_l1330_133062


namespace city_B_sand_amount_l1330_133097

def total_sand : ℝ := 95
def city_A_sand : ℝ := 16.5
def city_C_sand : ℝ := 24.5
def city_D_sand : ℝ := 28

theorem city_B_sand_amount : 
  total_sand - city_A_sand - city_C_sand - city_D_sand = 26 := by
  sorry

end city_B_sand_amount_l1330_133097


namespace min_t_for_equations_l1330_133028

theorem min_t_for_equations (a b c d e : ℝ) 
  (h_non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) 
  (h_sum_pos : a + b + c + d + e > 0) :
  (∃ t : ℝ, t = Real.sqrt 2 ∧ 
    a + c = t * b ∧ 
    b + d = t * c ∧ 
    c + e = t * d) ∧
  (∀ s : ℝ, (a + c = s * b ∧ b + d = s * c ∧ c + e = s * d) → s ≥ Real.sqrt 2) :=
by sorry

end min_t_for_equations_l1330_133028


namespace total_marbles_is_240_l1330_133021

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * dozen

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 4 * jessica_marbles

/-- The number of red marbles Alex has -/
def alex_marbles : ℕ := jessica_marbles + 2 * dozen

/-- The total number of red marbles Jessica, Sandy, and Alex have -/
def total_marbles : ℕ := jessica_marbles + sandy_marbles + alex_marbles

theorem total_marbles_is_240 : total_marbles = 240 := by
  sorry

end total_marbles_is_240_l1330_133021


namespace quadratic_has_minimum_l1330_133012

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = b^2 / (9a) and a > 0,
    prove that the graph of y = f(x) has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + b^2 / (9 * a)
  ∃ x_min : ℝ, ∀ x : ℝ, f x_min ≤ f x :=
by sorry

end quadratic_has_minimum_l1330_133012


namespace train_speed_calculation_l1330_133050

/-- Calculates the speed of a train given the lengths of two trains, the speed of the second train, and the time taken for the first train to pass the second train. -/
theorem train_speed_calculation (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) :
  length1 = 250 →
  length2 = 300 →
  speed2 = 36 * (1000 / 3600) →
  time = 54.995600351971845 →
  ∃ (speed1 : ℝ), speed1 = 72 * (1000 / 3600) ∧
    (length1 + length2) / time = speed1 - speed2 :=
by sorry

end train_speed_calculation_l1330_133050


namespace time_sum_after_increment_l1330_133058

-- Define a type for time on a 12-hour digital clock
structure Time12Hour where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ
  is_pm : Bool

-- Function to add hours, minutes, and seconds to a given time
def addTime (start : Time12Hour) (hours minutes seconds : ℕ) : Time12Hour :=
  sorry

-- Function to calculate A + B + C for a given time
def sumTime (t : Time12Hour) : ℕ :=
  t.hours + t.minutes + t.seconds

-- Theorem statement
theorem time_sum_after_increment :
  let start_time := Time12Hour.mk 3 0 0 true
  let end_time := addTime start_time 190 45 30
  sumTime end_time = 76 := by sorry

end time_sum_after_increment_l1330_133058


namespace rays_dog_walks_66_blocks_l1330_133056

/-- Represents the number of blocks Ray walks in different segments of his route -/
structure RayWalk where
  to_park : Nat
  to_school : Nat
  to_home : Nat

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  route : RayWalk
  walks_per_day : Nat

/-- Calculates the total number of blocks Ray's dog walks in a day -/
def total_blocks_walked (daily : DailyWalk) : Nat :=
  (daily.route.to_park + daily.route.to_school + daily.route.to_home) * daily.walks_per_day

/-- Theorem stating that Ray's dog walks 66 blocks each day -/
theorem rays_dog_walks_66_blocks (daily : DailyWalk) 
  (h1 : daily.route.to_park = 4)
  (h2 : daily.route.to_school = 7)
  (h3 : daily.route.to_home = 11)
  (h4 : daily.walks_per_day = 3) : 
  total_blocks_walked daily = 66 := by
  sorry

end rays_dog_walks_66_blocks_l1330_133056


namespace subtraction_of_negative_l1330_133032

theorem subtraction_of_negative : 12.345 - (-3.256) = 15.601 := by
  sorry

end subtraction_of_negative_l1330_133032


namespace joan_dimes_l1330_133061

/-- The number of dimes Joan has after spending some -/
def remaining_dimes (initial : ℕ) (spent : ℕ) : ℕ := initial - spent

/-- Theorem: If Joan had 5 dimes initially and spent 2 dimes, she now has 3 dimes -/
theorem joan_dimes : remaining_dimes 5 2 = 3 := by
  sorry

end joan_dimes_l1330_133061


namespace field_trip_theorem_l1330_133081

/-- Represents the number of students participating in the field trip -/
def num_students : ℕ := 245

/-- Represents the number of 35-seat buses needed to exactly fit all students -/
def num_35_seat_buses : ℕ := 7

/-- Represents the number of 45-seat buses needed to fit all students with one less bus -/
def num_45_seat_buses : ℕ := 6

/-- Represents the rental fee for a 35-seat bus in yuan -/
def fee_35_seat : ℕ := 320

/-- Represents the rental fee for a 45-seat bus in yuan -/
def fee_45_seat : ℕ := 380

/-- Represents the total number of buses to be rented -/
def total_buses : ℕ := 6

/-- Theorem stating the number of students and the most cost-effective rental plan -/
theorem field_trip_theorem : 
  (num_students = 35 * num_35_seat_buses) ∧ 
  (num_students = 45 * (num_45_seat_buses - 1) - 25) ∧
  (∀ a b : ℕ, a + b = total_buses → 
    35 * a + 45 * b ≥ num_students →
    fee_35_seat * a + fee_45_seat * b ≥ fee_35_seat * 2 + fee_45_seat * 4) :=
by sorry

end field_trip_theorem_l1330_133081


namespace absolute_value_inequality_l1330_133090

theorem absolute_value_inequality (x : ℝ) :
  |x + 3| > 1 ↔ x < -4 ∨ x > -2 :=
sorry

end absolute_value_inequality_l1330_133090


namespace cookie_cutter_sides_l1330_133023

/-- The number of sides on a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides on a square -/
def square_sides : ℕ := 4

/-- The number of sides on a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := 4

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_hexagons * hexagon_sides

theorem cookie_cutter_sides : total_sides = 46 := by
  sorry

end cookie_cutter_sides_l1330_133023


namespace quadratic_function_properties_l1330_133078

/-- A quadratic function f(x) with specific properties -/
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + c

/-- The theorem statement -/
theorem quadratic_function_properties (a c : ℝ) :
  (∀ x : ℝ, x ≠ 1/a → f a c x > 0) →
  (∃ min_a min_c : ℝ, 
    (∀ a' c' : ℝ, f a' c' 2 ≥ f min_a min_c 2) ∧
    f min_a min_c 2 = 0 ∧
    min_a = 1/2 ∧ min_c = 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 2 → f (1/2) 2 x + 4 ≥ m * (x - 2)) → m ≤ 2 * Real.sqrt 2) :=
by sorry

end quadratic_function_properties_l1330_133078


namespace volleyball_match_probability_l1330_133063

-- Define the probability of Team A winning a single game
def p_win_game : ℚ := 2/3

-- Define the probability of Team A winning the match
def p_win_match : ℚ := 20/27

-- Theorem statement
theorem volleyball_match_probability :
  (p_win_game = 2/3) →  -- Probability of Team A winning a single game
  (p_win_match = p_win_game * p_win_game + 2 * p_win_game * (1 - p_win_game) * p_win_game) :=
by
  sorry

#check volleyball_match_probability

end volleyball_match_probability_l1330_133063


namespace ratio_is_five_l1330_133001

/-- The equation holds for all real x except -3, 0, and 6 -/
def equation_holds (P Q : ℤ) : Prop :=
  ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 6 →
    (P : ℝ) / (x + 3) + (Q : ℝ) / (x^2 - 6*x) = (x^2 - 4*x + 15) / (x^3 + x^2 - 18*x)

theorem ratio_is_five (P Q : ℤ) (h : equation_holds P Q) : (Q : ℚ) / P = 5 := by
  sorry

end ratio_is_five_l1330_133001


namespace binomial_expansion_coefficient_l1330_133098

/-- Given that in the expansion of (2x + a/x^2)^5, the coefficient of x^(-4) is 320, prove that a = 2 -/
theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ (c : ℝ), c = (Nat.choose 5 3) * 2^2 * a^3 ∧ c = 320) → a = 2 := by
sorry

end binomial_expansion_coefficient_l1330_133098


namespace two_digit_number_special_property_l1330_133026

theorem two_digit_number_special_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ x y : ℕ, n = 10 * x + y ∧ x < 10 ∧ y < 10 ∧ n = x^3 + y^2) ∧
  n = 24 := by
sorry

end two_digit_number_special_property_l1330_133026


namespace square_last_digits_l1330_133065

theorem square_last_digits (n : ℕ) :
  (n^2 % 10 % 2 = 1) → ((n^2 % 100) / 10 % 2 = 0) := by
  sorry

end square_last_digits_l1330_133065


namespace north_american_stamps_cost_is_91_cents_l1330_133095

/-- Represents a country --/
inductive Country
| China
| Japan
| Canada
| Mexico

/-- Represents a continent --/
inductive Continent
| Asia
| NorthAmerica

/-- Represents a decade --/
inductive Decade
| D1960s
| D1970s

/-- Maps a country to its continent --/
def country_continent : Country → Continent
| Country.China => Continent.Asia
| Country.Japan => Continent.Asia
| Country.Canada => Continent.NorthAmerica
| Country.Mexico => Continent.NorthAmerica

/-- Cost of stamps in cents for each country --/
def stamp_cost : Country → ℕ
| Country.China => 7
| Country.Japan => 7
| Country.Canada => 3
| Country.Mexico => 4

/-- Number of stamps for each country and decade --/
def stamp_count : Country → Decade → ℕ
| Country.China => fun
  | Decade.D1960s => 5
  | Decade.D1970s => 9
| Country.Japan => fun
  | Decade.D1960s => 6
  | Decade.D1970s => 7
| Country.Canada => fun
  | Decade.D1960s => 7
  | Decade.D1970s => 6
| Country.Mexico => fun
  | Decade.D1960s => 8
  | Decade.D1970s => 5

/-- Total cost of North American stamps from 1960s and 1970s --/
def north_american_stamps_cost : ℚ :=
  let north_american_countries := [Country.Canada, Country.Mexico]
  let decades := [Decade.D1960s, Decade.D1970s]
  (north_american_countries.map fun country =>
    (decades.map fun decade =>
      (stamp_count country decade) * (stamp_cost country)
    ).sum
  ).sum / 100

theorem north_american_stamps_cost_is_91_cents :
  north_american_stamps_cost = 91 / 100 := by sorry

end north_american_stamps_cost_is_91_cents_l1330_133095


namespace sufficient_but_not_necessary_condition_minimal_m_l1330_133071

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 10) ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Define the sufficient condition
def sufficient (m : ℝ) : Prop :=
  ∀ x, q x m → p x

-- Define the not necessary condition
def not_necessary (m : ℝ) : Prop :=
  ∃ x, p x ∧ ¬(q x m)

-- Main theorem
theorem sufficient_but_not_necessary_condition (m : ℝ) 
  (h1 : m ≥ 3) (h2 : m > 0) : 
  sufficient m ∧ not_necessary m := by
  sorry

-- Prove that this is the minimal value of m
theorem minimal_m :
  ∀ m < 3, ¬(sufficient m ∧ not_necessary m) := by
  sorry

end sufficient_but_not_necessary_condition_minimal_m_l1330_133071


namespace smallest_divisible_by_3_and_4_l1330_133042

theorem smallest_divisible_by_3_and_4 : 
  ∀ n : ℕ, n > 0 ∧ 3 ∣ n ∧ 4 ∣ n → n ≥ 12 :=
by sorry

end smallest_divisible_by_3_and_4_l1330_133042


namespace distance_A_to_B_l1330_133074

/-- The distance between points A(1, 0) and B(0, -1) is √2. -/
theorem distance_A_to_B : Real.sqrt 2 = Real.sqrt ((0 - 1)^2 + (-1 - 0)^2) := by sorry

end distance_A_to_B_l1330_133074


namespace c_months_is_six_l1330_133005

/-- Represents the rental scenario for a pasture -/
structure PastureRental where
  total_rent : ℕ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  b_payment : ℕ

/-- Calculates the number of months c put in the horses -/
def calculate_c_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that c put in the horses for 6 months -/
theorem c_months_is_six (rental : PastureRental)
  (h1 : rental.total_rent = 870)
  (h2 : rental.a_horses = 12)
  (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16)
  (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18)
  (h7 : rental.b_payment = 360) :
  calculate_c_months rental = 6 :=
sorry

end c_months_is_six_l1330_133005


namespace pascal_ratio_row_34_l1330_133068

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Check if three consecutive entries in a row are in the ratio 2:3:4 -/
def hasRatio234 (n : ℕ) (r : ℕ) : Prop :=
  4 * pascal n r = 3 * pascal n (r+1) ∧
  4 * pascal n (r+1) = 3 * pascal n (r+2)

theorem pascal_ratio_row_34 : ∃ r, hasRatio234 34 r := by
  sorry

#check pascal_ratio_row_34

end pascal_ratio_row_34_l1330_133068


namespace petrol_price_reduction_l1330_133060

/-- The original price of petrol in dollars per gallon -/
def P : ℝ := sorry

/-- The amount spent on petrol in dollars -/
def amount_spent : ℝ := 250

/-- The price reduction percentage as a decimal -/
def price_reduction : ℝ := 0.1

/-- The additional gallons that can be bought after the price reduction -/
def additional_gallons : ℝ := 5

/-- Theorem stating the relationship between the original price and the additional gallons that can be bought after the price reduction -/
theorem petrol_price_reduction (P : ℝ) (amount_spent : ℝ) (price_reduction : ℝ) (additional_gallons : ℝ) :
  amount_spent / ((1 - price_reduction) * P) - amount_spent / P = additional_gallons :=
sorry

end petrol_price_reduction_l1330_133060


namespace total_trees_on_farm_l1330_133085

def farm_trees (mango_trees : ℕ) (coconut_trees : ℕ) : ℕ :=
  mango_trees + coconut_trees

theorem total_trees_on_farm :
  let mango_trees : ℕ := 60
  let coconut_trees : ℕ := mango_trees / 2 - 5
  farm_trees mango_trees coconut_trees = 85 := by
  sorry

end total_trees_on_farm_l1330_133085


namespace system_solutions_l1330_133035

def is_solution (x y z u : ℤ) : Prop :=
  x + y + z + u = 12 ∧
  x^2 + y^2 + z^2 + u^2 = 170 ∧
  x^3 + y^3 + z^3 + u^3 = 1764 ∧
  x * y = z * u

def solutions : List (ℤ × ℤ × ℤ × ℤ) :=
  [(12, -1, 4, -3), (12, -1, -3, 4), (-1, 12, 4, -3), (-1, 12, -3, 4),
   (4, -3, 12, -1), (4, -3, -1, 12), (-3, 4, 12, -1), (-3, 4, -1, 12)]

theorem system_solutions :
  (∀ x y z u : ℤ, is_solution x y z u ↔ (x, y, z, u) ∈ solutions) ∧
  solutions.length = 8 := by
  sorry

end system_solutions_l1330_133035


namespace quadratic_equation_roots_l1330_133077

theorem quadratic_equation_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ 
   (a^2 - 1) * x^2 - 2*(5*a + 1)*x + 24 = 0 ∧
   (a^2 - 1) * y^2 - 2*(5*a + 1)*y + 24 = 0) → 
  a = -2 := by
sorry

end quadratic_equation_roots_l1330_133077


namespace perimeter_difference_rectangles_l1330_133044

/-- Calculate the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculate the positive difference between two natural numbers -/
def positiveDifference (a b : ℕ) : ℕ :=
  max a b - min a b

theorem perimeter_difference_rectangles :
  positiveDifference (rectanglePerimeter 3 4) (rectanglePerimeter 1 8) = 4 := by
  sorry

end perimeter_difference_rectangles_l1330_133044


namespace rational_fraction_equality_l1330_133091

theorem rational_fraction_equality (a b : ℚ) 
  (h1 : (a + 2*b) / (2*a - b) = 2)
  (h2 : 3*a - 2*b ≠ 0) :
  (3*a + 2*b) / (3*a - 2*b) = 3 := by
sorry

end rational_fraction_equality_l1330_133091


namespace x_plus_y_equals_two_l1330_133022

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : (x - 1)^2017 + 2013 * (x - 1) = -1)
  (hy : (y - 1)^2017 + 2013 * (y - 1) = 1) : 
  x + y = 2 := by
  sorry

end x_plus_y_equals_two_l1330_133022


namespace slope_of_tan_45_degrees_line_l1330_133029

theorem slope_of_tan_45_degrees_line (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.tan (45 * π / 180)
  (deriv f) x = 0 := by
  sorry

end slope_of_tan_45_degrees_line_l1330_133029


namespace equal_numbers_l1330_133075

theorem equal_numbers (a b c : ℝ) (h : |a - b| = 2*|b - c| ∧ |a - b| = 3*|c - a|) : a = b ∧ b = c := by
  sorry

end equal_numbers_l1330_133075


namespace binary_1011001_equals_base5_324_l1330_133008

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_1011001_equals_base5_324 : 
  decimal_to_base5 (binary_to_decimal [true, false, false, true, true, false, true]) = [3, 2, 4] := by
  sorry

end binary_1011001_equals_base5_324_l1330_133008


namespace y_not_between_l1330_133033

theorem y_not_between (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x y : ℝ, y = (a * Real.sin x + b) / (a * Real.sin x - b) →
  (a > b → (y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b))) :=
by sorry

end y_not_between_l1330_133033


namespace total_enjoyable_gameplay_l1330_133043

/-- Calculates the total enjoyable gameplay time given the conditions of the game, expansion, and mod. -/
theorem total_enjoyable_gameplay 
  (original_game_hours : ℝ)
  (original_game_boring_percent : ℝ)
  (expansion_hours : ℝ)
  (expansion_load_screen_percent : ℝ)
  (expansion_inventory_percent : ℝ)
  (mod_skip_percent : ℝ)
  (h1 : original_game_hours = 150)
  (h2 : original_game_boring_percent = 0.7)
  (h3 : expansion_hours = 50)
  (h4 : expansion_load_screen_percent = 0.25)
  (h5 : expansion_inventory_percent = 0.25)
  (h6 : mod_skip_percent = 0.15) :
  let original_enjoyable := original_game_hours * (1 - original_game_boring_percent)
  let expansion_enjoyable := expansion_hours * (1 - expansion_load_screen_percent) * (1 - expansion_inventory_percent)
  let total_tedious := original_game_hours * original_game_boring_percent + 
                       expansion_hours * (expansion_load_screen_percent + (1 - expansion_load_screen_percent) * expansion_inventory_percent)
  let mod_skipped := total_tedious * mod_skip_percent
  original_enjoyable + expansion_enjoyable + mod_skipped = 92.15625 := by
  sorry


end total_enjoyable_gameplay_l1330_133043


namespace octal_sum_equality_l1330_133073

/-- Represents a number in base 8 --/
def OctalNumber : Type := List Nat

/-- Converts an OctalNumber to a natural number --/
def octal_to_nat (n : OctalNumber) : Nat :=
  n.foldl (fun acc d => 8 * acc + d) 0

/-- Adds two OctalNumbers in base 8 --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_sum_equality : 
  octal_add [1, 4, 6, 3] [2, 7, 5] = [1, 7, 5, 0] :=
sorry

end octal_sum_equality_l1330_133073


namespace coffee_cheesecake_set_price_l1330_133015

/-- Calculates the discounted price of a coffee and cheesecake set --/
def discounted_set_price (coffee_price cheesecake_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_price := coffee_price + cheesecake_price
  let discount_amount := discount_rate * total_price
  total_price - discount_amount

/-- Proves that the final price of a coffee and cheesecake set with a 25% discount is $12 --/
theorem coffee_cheesecake_set_price :
  discounted_set_price 6 10 (25 / 100) = 12 :=
by sorry

end coffee_cheesecake_set_price_l1330_133015


namespace prime_p_cube_condition_l1330_133002

theorem prime_p_cube_condition (p : ℕ) : 
  Prime p → (∃ n : ℕ, 13 * p + 1 = n^3) → p = 2 ∨ p = 211 := by
sorry

end prime_p_cube_condition_l1330_133002


namespace joes_dad_marshmallows_joes_dad_marshmallows_proof_l1330_133019

theorem joes_dad_marshmallows : ℕ → Prop :=
  fun d : ℕ =>
    let joe_marshmallows : ℕ := 4 * d
    let dad_roasted : ℕ := d / 3
    let joe_roasted : ℕ := joe_marshmallows / 2
    dad_roasted + joe_roasted = 49 → d = 21

-- The proof goes here
theorem joes_dad_marshmallows_proof : joes_dad_marshmallows 21 := by
  sorry

end joes_dad_marshmallows_joes_dad_marshmallows_proof_l1330_133019


namespace equation_solution_set_l1330_133082

theorem equation_solution_set : 
  {x : ℝ | x^6 + x^2 = (2*x + 3)^3 + 2*x + 3} = {-1, 3} := by
  sorry

end equation_solution_set_l1330_133082


namespace sum_two_longest_altitudes_eq_14_l1330_133069

/-- A triangle with sides 6, 8, and 10 -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side1_eq : side1 = 6)
  (side2_eq : side2 = 8)
  (side3_eq : side3 = 10)

/-- The length of an altitude in a triangle -/
def altitude_length (t : Triangle) : ℝ → ℝ :=
  sorry

/-- The sum of the two longest altitudes in the triangle -/
def sum_two_longest_altitudes (t : Triangle) : ℝ :=
  sorry

/-- Theorem: The sum of the two longest altitudes in a triangle with sides 6, 8, and 10 is 14 -/
theorem sum_two_longest_altitudes_eq_14 (t : Triangle) :
  sum_two_longest_altitudes t = 14 := by
  sorry

end sum_two_longest_altitudes_eq_14_l1330_133069


namespace complementary_angles_difference_l1330_133066

theorem complementary_angles_difference (a b : ℝ) (h1 : a + b = 90) (h2 : a / b = 5 / 4) :
  |a - b| = 10 := by
  sorry

end complementary_angles_difference_l1330_133066


namespace absolute_value_simplification_l1330_133084

theorem absolute_value_simplification (a : ℝ) (h : a < 3) : |a - 3| = 3 - a := by
  sorry

end absolute_value_simplification_l1330_133084


namespace fraction_equality_implies_relationship_l1330_133051

theorem fraction_equality_implies_relationship (a b c d : ℝ) :
  (a + b + 1) / (b + c + 2) = (c + d + 1) / (d + a + 2) →
  (a - c) * (a + b + c + d + 2) = 0 := by
sorry

end fraction_equality_implies_relationship_l1330_133051


namespace y_value_l1330_133072

theorem y_value : (2023^2 - 1012) / 2023 = 2023 - 1012/2023 := by sorry

end y_value_l1330_133072


namespace race_participants_l1330_133045

theorem race_participants (total : ℕ) (finished : ℕ) : 
  finished = 52 →
  (3/4 : ℚ) * total * (1/3 : ℚ) + 
  (3/4 : ℚ) * total * (2/3 : ℚ) * (4/5 : ℚ) = finished →
  total = 130 := by
  sorry

end race_participants_l1330_133045


namespace mark_spent_40_l1330_133054

/-- The total amount Mark spent on tomatoes and apples -/
def total_spent (tomato_price : ℝ) (tomato_weight : ℝ) (apple_price : ℝ) (apple_weight : ℝ) : ℝ :=
  tomato_price * tomato_weight + apple_price * apple_weight

/-- Theorem stating that Mark spent $40 in total -/
theorem mark_spent_40 : 
  total_spent 5 2 6 5 = 40 := by sorry

end mark_spent_40_l1330_133054


namespace vishal_investment_percentage_l1330_133049

def total_investment : ℝ := 6358
def raghu_investment : ℝ := 2200
def trishul_investment_percentage : ℝ := 90  -- 100% - 10%

theorem vishal_investment_percentage (vishal_investment trishul_investment : ℝ) : 
  vishal_investment + trishul_investment + raghu_investment = total_investment →
  trishul_investment = raghu_investment * trishul_investment_percentage / 100 →
  (vishal_investment - trishul_investment) / trishul_investment * 100 = 10 := by
  sorry

end vishal_investment_percentage_l1330_133049


namespace hexagon_covers_ground_l1330_133018

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

def can_cover_ground (n : ℕ) : Prop :=
  ∃ k : ℕ, k * interior_angle n = 360

theorem hexagon_covers_ground :
  can_cover_ground 6 ∧
  ¬can_cover_ground 5 ∧
  ¬can_cover_ground 8 ∧
  ¬can_cover_ground 12 :=
sorry

end hexagon_covers_ground_l1330_133018


namespace simplest_quadratic_radical_l1330_133007

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℚ) : Prop :=
  ∃ m : ℚ, m * m = n

-- Define a function to check if a quadratic radical is in its simplest form
def is_simplest_quadratic_radical (n : ℚ) : Prop :=
  n > 0 ∧ ¬is_perfect_square n ∧ (∀ m : ℕ, m > 1 → ¬is_perfect_square (n / (m * m : ℚ)))

-- Theorem statement
theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical 7 ∧
  ¬is_simplest_quadratic_radical 9 ∧
  ¬is_simplest_quadratic_radical 20 ∧
  ¬is_simplest_quadratic_radical (1/3) :=
by sorry

end simplest_quadratic_radical_l1330_133007


namespace eagles_volleyball_games_l1330_133059

theorem eagles_volleyball_games :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
  initial_wins = (0.4 : ℝ) * initial_games →
  (initial_wins + 9 : ℝ) / (initial_games + 10) = 0.55 →
  initial_games + 10 = 33 :=
by
  sorry

end eagles_volleyball_games_l1330_133059


namespace no_increase_employees_l1330_133003

theorem no_increase_employees (total : ℕ) (salary_percent : ℚ) (travel_percent : ℚ) :
  total = 480 →
  salary_percent = 10 / 100 →
  travel_percent = 20 / 100 →
  total - (total * salary_percent).floor - (total * travel_percent).floor = 336 :=
by sorry

end no_increase_employees_l1330_133003


namespace geometric_sequence_property_l1330_133064

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence, if a_4 * a_6 = 10, then a_2 * a_8 = 10 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 4 * a 6 = 10) : a 2 * a 8 = 10 := by
  sorry

end geometric_sequence_property_l1330_133064


namespace base7_to_base4_conversion_l1330_133020

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- The given number in base 7 -/
def given_number : ℕ := 563

theorem base7_to_base4_conversion :
  base10ToBase4 (base7ToBase10 given_number) = 10202 := by sorry

end base7_to_base4_conversion_l1330_133020


namespace school_population_l1330_133094

theorem school_population (total_students : ℕ) : 
  (5 : ℚ)/8 * total_students + (3 : ℚ)/8 * total_students = total_students →  -- Girls + Boys = Total
  ((3 : ℚ)/10 * (5 : ℚ)/8 * total_students : ℚ) + 
  ((3 : ℚ)/5 * (3 : ℚ)/8 * total_students : ℚ) = 330 →                       -- Middle schoolers
  total_students = 800 := by
sorry

end school_population_l1330_133094


namespace stuarts_initial_marbles_l1330_133057

/-- Stuart's initial marble count problem -/
theorem stuarts_initial_marbles (betty_marbles : ℕ) (stuart_final : ℕ) 
  (h1 : betty_marbles = 60)
  (h2 : stuart_final = 80) :
  ∃ (stuart_initial : ℕ), 
    stuart_initial + (betty_marbles * 2/5 : ℕ) = stuart_final ∧ 
    stuart_initial = 56 := by
  sorry

end stuarts_initial_marbles_l1330_133057


namespace vector_decomposition_l1330_133017

theorem vector_decomposition (x p q r : ℝ × ℝ × ℝ) 
  (hx : x = (-9, -8, -3))
  (hp : p = (1, 4, 1))
  (hq : q = (-3, 2, 0))
  (hr : r = (1, -1, 2)) :
  ∃ (α β γ : ℝ), x = α • p + β • q + γ • r ∧ α = -3 ∧ β = 2 ∧ γ = 0 :=
by sorry

end vector_decomposition_l1330_133017


namespace opposite_of_a_is_smallest_positive_integer_l1330_133004

theorem opposite_of_a_is_smallest_positive_integer (a : ℤ) : 
  (∃ (x : ℤ), x > 0 ∧ ∀ (y : ℤ), y > 0 → x ≤ y) ∧ (-a = x) → 3*a - 2 = -5 := by
  sorry

end opposite_of_a_is_smallest_positive_integer_l1330_133004


namespace common_root_of_equations_l1330_133025

theorem common_root_of_equations : ∃ x : ℚ, 
  2 * x^3 - 5 * x^2 + 6 * x - 2 = 0 ∧ 
  6 * x^3 - 3 * x^2 - 2 * x + 1 = 0 := by
  use 1/2
  sorry

#eval (2 * (1/2)^3 - 5 * (1/2)^2 + 6 * (1/2) - 2 : ℚ)
#eval (6 * (1/2)^3 - 3 * (1/2)^2 - 2 * (1/2) + 1 : ℚ)

end common_root_of_equations_l1330_133025


namespace p_percentage_of_x_l1330_133034

theorem p_percentage_of_x (x y z w t u p : ℝ) 
  (h1 : 0.37 * z = 0.84 * y)
  (h2 : y = 0.62 * x)
  (h3 : 0.47 * w = 0.73 * z)
  (h4 : w = t - u)
  (h5 : u = 0.25 * t)
  (h6 : p = z + t + u) :
  p = 5.05675 * x := by sorry

end p_percentage_of_x_l1330_133034


namespace binary_multiplication_division_l1330_133037

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Theorem: The result of (101110₂) × (110100₂) ÷ (110₂) is 101011100₂ -/
theorem binary_multiplication_division :
  let a := binaryToNat [true, false, true, true, true, false]  -- 101110₂
  let b := binaryToNat [true, true, false, true, false, false] -- 110100₂
  let c := binaryToNat [true, true, false]                     -- 110₂
  let result := binaryToNat [true, false, true, false, true, true, true, false, false] -- 101011100₂
  a * b / c = result := by
  sorry


end binary_multiplication_division_l1330_133037


namespace corporation_employee_count_l1330_133096

/-- The number of employees at a corporation. -/
structure Corporation where
  female_employees : ℕ
  total_managers : ℕ
  male_associates : ℕ
  female_managers : ℕ

/-- The total number of employees in the corporation. -/
def Corporation.total_employees (c : Corporation) : ℕ :=
  c.female_employees + c.male_associates + (c.total_managers - c.female_managers)

/-- Theorem stating that the total number of employees is 250 given the specific conditions. -/
theorem corporation_employee_count (c : Corporation)
  (h1 : c.female_employees = 90)
  (h2 : c.total_managers = 40)
  (h3 : c.male_associates = 160)
  (h4 : c.female_managers = 40) :
  c.total_employees = 250 := by
  sorry

#check corporation_employee_count

end corporation_employee_count_l1330_133096
