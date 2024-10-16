import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3165_316523

theorem inequality_proof (a b : ℝ) (h : a * b > 0) : b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3165_316523


namespace NUMINAMATH_CALUDE_unique_cube_property_l3165_316559

theorem unique_cube_property :
  ∃! (n : ℕ), n > 0 ∧ n^3 / 1000 = n :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_property_l3165_316559


namespace NUMINAMATH_CALUDE_money_left_after_sale_l3165_316569

/-- Represents the total revenue from selling items in a store's inventory. -/
def total_revenue (
  category_a_items : ℕ)
  (category_b_items : ℕ)
  (category_c_items : ℕ)
  (category_a_price : ℚ)
  (category_b_price : ℚ)
  (category_c_price : ℚ)
  (category_a_discount : ℚ)
  (category_b_discount : ℚ)
  (category_c_discount : ℚ)
  (category_a_sold_percent : ℚ)
  (category_b_sold_percent : ℚ)
  (category_c_sold_percent : ℚ) : ℚ :=
  (category_a_items : ℚ) * category_a_price * (1 - category_a_discount) * category_a_sold_percent +
  (category_b_items : ℚ) * category_b_price * (1 - category_b_discount) * category_b_sold_percent +
  (category_c_items : ℚ) * category_c_price * (1 - category_c_discount) * category_c_sold_percent

/-- Theorem stating the amount of money left after the sale and paying creditors. -/
theorem money_left_after_sale : 
  total_revenue 1000 700 300 50 75 100 0.8 0.7 0.6 0.85 0.75 0.9 - 15000 = 16112.5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_sale_l3165_316569


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l3165_316521

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l3165_316521


namespace NUMINAMATH_CALUDE_inequality_proof_l3165_316570

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3165_316570


namespace NUMINAMATH_CALUDE_det_inequality_equiv_l3165_316522

/-- Definition of a second-order determinant -/
def secondOrderDet (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the equivalence of the determinant inequality and the simplified inequality -/
theorem det_inequality_equiv (x : ℝ) :
  secondOrderDet 2 (3 - x) 1 x > 0 ↔ 3 * x - 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_det_inequality_equiv_l3165_316522


namespace NUMINAMATH_CALUDE_new_person_weight_l3165_316531

/-- The weight of the new person in a group, given:
  * The initial number of people in the group
  * The average weight increase when a new person replaces one person
  * The weight of the person being replaced
-/
def weight_of_new_person (initial_count : ℕ) (avg_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + initial_count * avg_increase

theorem new_person_weight :
  weight_of_new_person 12 (37/10) 65 = 1094/10 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3165_316531


namespace NUMINAMATH_CALUDE_intersection_when_m_zero_necessary_not_sufficient_condition_l3165_316503

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | (x - m + 1)*(x - m - 1) > 0}

theorem intersection_when_m_zero :
  A ∩ B 0 = {x : ℝ | 1 < x ∧ x ≤ 3} := by sorry

theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ m < -2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_zero_necessary_not_sufficient_condition_l3165_316503


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3165_316527

theorem least_n_satisfying_inequality : ∃ n : ℕ, 
  (∀ k : ℕ, k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3165_316527


namespace NUMINAMATH_CALUDE_multiply_121_54_l3165_316548

theorem multiply_121_54 : 121 * 54 = 6534 := by
  sorry

end NUMINAMATH_CALUDE_multiply_121_54_l3165_316548


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3165_316552

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : ∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) :
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2) ∧
  (a₁ + a₃ + a₅ + a₇ = -1094) ∧
  (a₀ + a₂ + a₄ + a₆ = 1093) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3165_316552


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3165_316545

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3165_316545


namespace NUMINAMATH_CALUDE_square_side_length_l3165_316533

/-- Given a ribbon of length 78 cm used to make a triangle and a square,
    with the triangle having a perimeter of 46 cm,
    prove that the length of one side of the square is 8 cm. -/
theorem square_side_length (total_ribbon : ℝ) (triangle_perimeter : ℝ) (square_side : ℝ) :
  total_ribbon = 78 ∧ 
  triangle_perimeter = 46 ∧ 
  square_side * 4 = total_ribbon - triangle_perimeter → 
  square_side = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3165_316533


namespace NUMINAMATH_CALUDE_three_two_digit_multiples_l3165_316584

theorem three_two_digit_multiples :
  (∃! (s : Finset ℕ), 
    (∀ x ∈ s, x > 0 ∧ 
      (∃! (m : Finset ℕ), 
        (∀ y ∈ m, 10 ≤ y ∧ y < 100 ∧ ∃ k, y = k * x) ∧ 
        m.card = 3)) ∧ 
    s.card = 9) := by sorry

end NUMINAMATH_CALUDE_three_two_digit_multiples_l3165_316584


namespace NUMINAMATH_CALUDE_system_solution_l3165_316540

theorem system_solution : 
  ∃ (x y : ℝ), (x^4 - y^4 = 3 * Real.sqrt (abs y) - 3 * Real.sqrt (abs x)) ∧ 
                (x^2 - 2*x*y = 27) ↔ 
  ((x = 3 ∧ y = -3) ∨ (x = -3 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3165_316540


namespace NUMINAMATH_CALUDE_grass_seed_min_cost_l3165_316518

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat

/-- Finds the minimum cost to buy grass seed given the constraints -/
def minCostGrassSeed (bags : List GrassSeedBag) (minWeight maxWeight : Nat) : Rat :=
  sorry

/-- Theorem stating the minimum cost for the given problem -/
theorem grass_seed_min_cost :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 138/10 },
    { weight := 10, price := 2043/100 },
    { weight := 25, price := 3225/100 }
  ]
  minCostGrassSeed bags 65 80 = 9675/100 := by sorry

end NUMINAMATH_CALUDE_grass_seed_min_cost_l3165_316518


namespace NUMINAMATH_CALUDE_trig_problem_l3165_316564

theorem trig_problem (θ : Real) 
  (h1 : θ > 0) 
  (h2 : θ < Real.pi / 2) 
  (h3 : Real.cos (θ + Real.pi / 6) = 1 / 3) : 
  Real.sin θ = (2 * Real.sqrt 6 - 1) / 6 ∧ 
  Real.sin (2 * θ + Real.pi / 6) = (4 * Real.sqrt 6 + 7) / 18 := by
sorry

end NUMINAMATH_CALUDE_trig_problem_l3165_316564


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3165_316592

theorem least_subtraction_for_divisibility (n : ℕ) (primes : List ℕ) 
  (h_n : n = 899830)
  (h_primes : primes = [2, 3, 5, 7, 11]) : 
  ∃ (k : ℕ), 
    k = 2000 ∧ 
    (∀ m : ℕ, m < k → ¬((n - m) % (primes.prod) = 0)) ∧ 
    ((n - k) % (primes.prod) = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3165_316592


namespace NUMINAMATH_CALUDE_kate_age_l3165_316514

theorem kate_age (total_age maggie_age sue_age : ℕ) 
  (h1 : total_age = 48)
  (h2 : maggie_age = 17)
  (h3 : sue_age = 12) :
  total_age - maggie_age - sue_age = 19 :=
by sorry

end NUMINAMATH_CALUDE_kate_age_l3165_316514


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3165_316536

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 5 * x - 2 > 0) ↔ (1/2 < x ∧ x < b)) → 
  (a = -2 ∧ b = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3165_316536


namespace NUMINAMATH_CALUDE_mark_soup_donation_l3165_316581

theorem mark_soup_donation (shelters : ℕ) (people_per_shelter : ℕ) (cans_per_person : ℕ)
  (h1 : shelters = 6)
  (h2 : people_per_shelter = 30)
  (h3 : cans_per_person = 10) :
  shelters * people_per_shelter * cans_per_person = 1800 :=
by sorry

end NUMINAMATH_CALUDE_mark_soup_donation_l3165_316581


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3165_316541

theorem absolute_value_equation : ∀ x : ℝ, 
  (abs x) * (abs (-25) - abs 5) = 40 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3165_316541


namespace NUMINAMATH_CALUDE_ball_ratio_problem_l3165_316598

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 4 / 3 →
  white_balls = 12 →
  red_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_ball_ratio_problem_l3165_316598


namespace NUMINAMATH_CALUDE_isosceles_triangle_solution_l3165_316560

def isosceles_triangle_sides (perimeter : ℝ) (height_ratio : ℝ) : Prop :=
  let base := 130
  let leg := 169
  perimeter = base + 2 * leg ∧
  height_ratio = 10 / 13 ∧
  base * (13 : ℝ) = leg * (10 : ℝ)

theorem isosceles_triangle_solution :
  isosceles_triangle_sides 468 (10 / 13) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_solution_l3165_316560


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3165_316586

theorem sqrt_equation_solution : 
  Real.sqrt (2 + Real.sqrt (3 + Real.sqrt (81/256))) = (2 + Real.sqrt (81/256)) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3165_316586


namespace NUMINAMATH_CALUDE_transformation_matrix_correct_l3165_316574

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_factor : ℝ := 2

theorem transformation_matrix_correct :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]
  ∀ (v : Fin 2 → ℝ),
    M.mulVec v = scaling_factor • (rotation_matrix.mulVec v) :=
by sorry

end NUMINAMATH_CALUDE_transformation_matrix_correct_l3165_316574


namespace NUMINAMATH_CALUDE_min_value_implies_a_equals_9_l3165_316568

theorem min_value_implies_a_equals_9 (t a : ℝ) (h1 : 0 < t) (h2 : t < π / 2) (h3 : a > 0) :
  (∀ s, 0 < s ∧ s < π / 2 → (1 / Real.cos s + a / (1 - Real.cos s)) ≥ 16) ∧
  (∃ s, 0 < s ∧ s < π / 2 ∧ 1 / Real.cos s + a / (1 - Real.cos s) = 16) →
  a = 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_equals_9_l3165_316568


namespace NUMINAMATH_CALUDE_jace_initial_earnings_l3165_316588

theorem jace_initial_earnings (debt : ℕ) (remaining : ℕ) (h1 : debt = 358) (h2 : remaining = 642) :
  debt + remaining = 1000 := by
  sorry

end NUMINAMATH_CALUDE_jace_initial_earnings_l3165_316588


namespace NUMINAMATH_CALUDE_sequence_ratio_l3165_316562

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-1 : ℝ) - a₁ = a₁ - a₂) →  -- arithmetic sequence condition
  (a₂ - (-4 : ℝ) = a₁ - a₂) →  -- arithmetic sequence condition
  (b₁ / (-1 : ℝ) = b₂ / b₁) →  -- geometric sequence condition
  (b₂ / b₁ = b₃ / b₂) →        -- geometric sequence condition
  (b₃ / b₂ = (-4 : ℝ) / b₃) →  -- geometric sequence condition
  (a₂ - a₁) / b₂ = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3165_316562


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l3165_316501

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The GDP value in ten thousand yuan -/
def gdp : ℝ := 84300000

/-- The scientific notation representation of the GDP -/
def gdp_scientific : ScientificNotation := {
  coefficient := 8.43,
  exponent := 7,
  h1 := by sorry
}

/-- Theorem stating that the GDP in scientific notation is correct -/
theorem gdp_scientific_notation_correct : 
  gdp = gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l3165_316501


namespace NUMINAMATH_CALUDE_cube_with_cut_corners_has_44_edges_l3165_316578

/-- A cube with cut corners is a polyhedron obtained by cutting off each corner of a cube
    such that no two cutting planes intersect within the cube, and each corner cut
    removes a vertex and replaces it with a quadrilateral face. -/
structure CubeWithCutCorners where
  /-- The number of vertices in the original cube -/
  original_vertices : ℕ
  /-- The number of edges in the original cube -/
  original_edges : ℕ
  /-- The number of new edges introduced by each corner cut -/
  new_edges_per_cut : ℕ
  /-- The condition that the original shape is a cube -/
  is_cube : original_vertices = 8 ∧ original_edges = 12
  /-- The condition that each corner cut introduces 4 new edges -/
  corner_cut : new_edges_per_cut = 4

/-- The number of edges in the resulting figure after cutting off all corners of a cube -/
def num_edges_after_cuts (c : CubeWithCutCorners) : ℕ :=
  c.original_edges + c.original_vertices * c.new_edges_per_cut

/-- Theorem stating that a cube with cut corners has 44 edges -/
theorem cube_with_cut_corners_has_44_edges (c : CubeWithCutCorners) :
  num_edges_after_cuts c = 44 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cut_corners_has_44_edges_l3165_316578


namespace NUMINAMATH_CALUDE_order_of_expressions_l3165_316544

theorem order_of_expressions (x : ℝ) :
  let a : ℝ := -x^2 - 2*x
  let b : ℝ := -2*x^2 - 2
  let c : ℝ := Real.sqrt 5 - 1
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l3165_316544


namespace NUMINAMATH_CALUDE_video_game_expenditure_l3165_316519

theorem video_game_expenditure (total : ℝ) (books toys snacks : ℝ) : 
  total = 45 →
  books = (1/4) * total →
  toys = (1/3) * total →
  snacks = (2/9) * total →
  total - (books + toys + snacks) = 8.75 :=
by sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l3165_316519


namespace NUMINAMATH_CALUDE_cube_of_4_minus_3i_l3165_316551

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem cube_of_4_minus_3i :
  (4 - 3 * i) ^ 3 = -44 - 117 * i :=
by sorry

end NUMINAMATH_CALUDE_cube_of_4_minus_3i_l3165_316551


namespace NUMINAMATH_CALUDE_real_equal_roots_iff_k_values_l3165_316502

/-- The quadratic equation in question -/
def equation (k x : ℝ) : ℝ := 3 * x^2 - 2 * k * x + 3 * x + 12

/-- Condition for real and equal roots -/
def has_real_equal_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, equation k x = 0 ∧ 
  ∀ y : ℝ, equation k y = 0 → y = x

/-- Theorem stating the values of k for which the equation has real and equal roots -/
theorem real_equal_roots_iff_k_values :
  ∀ k : ℝ, has_real_equal_roots k ↔ (k = -9/2 ∨ k = 15/2) :=
sorry

end NUMINAMATH_CALUDE_real_equal_roots_iff_k_values_l3165_316502


namespace NUMINAMATH_CALUDE_red_balls_count_l3165_316509

/-- Given a bag of balls with some red and some white balls, prove the number of red balls. -/
theorem red_balls_count (total_balls : ℕ) (red_prob : ℝ) (h_total : total_balls = 50) (h_prob : red_prob = 0.7) :
  ⌊(total_balls : ℝ) * red_prob⌋ = 35 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3165_316509


namespace NUMINAMATH_CALUDE_inclined_line_properties_l3165_316537

/-- A line passing through a point with a given inclination angle -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the equation and triangle area of an inclined line -/
theorem inclined_line_properties (l : InclinedLine) 
  (h1 : l.point = (Real.sqrt 3, -2))
  (h2 : l.angle = π / 3) : 
  ∃ (eq : LineEquation) (area : ℝ),
    eq.a = Real.sqrt 3 ∧ 
    eq.b = -1 ∧ 
    eq.c = -5 ∧
    area = (25 * Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_inclined_line_properties_l3165_316537


namespace NUMINAMATH_CALUDE_student_number_exists_l3165_316530

theorem student_number_exists : ∃ x : ℝ, Real.sqrt (2 * x^2 - 138) = 9 := by
  sorry

end NUMINAMATH_CALUDE_student_number_exists_l3165_316530


namespace NUMINAMATH_CALUDE_equation_represents_point_l3165_316520

/-- The equation x^2 + 36y^2 - 12x - 72y + 36 = 0 represents a single point (6, 1) in the xy-plane -/
theorem equation_represents_point :
  ∀ x y : ℝ, x^2 + 36*y^2 - 12*x - 72*y + 36 = 0 ↔ x = 6 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_point_l3165_316520


namespace NUMINAMATH_CALUDE_team_points_l3165_316579

/-- Calculates the total points earned by a sports team based on their performance. -/
def total_points (wins losses ties : ℕ) : ℕ :=
  2 * wins + 0 * losses + 1 * ties

/-- Theorem stating that a team with 9 wins, 3 losses, and 4 ties earns 22 points. -/
theorem team_points : total_points 9 3 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_team_points_l3165_316579


namespace NUMINAMATH_CALUDE_system_solution_l3165_316546

theorem system_solution :
  ∃ (x y : ℚ), 
    (7 * x - 50 * y = 2) ∧ 
    (3 * y - x = 4) ∧ 
    (x = -206/29) ∧ 
    (y = -30/29) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3165_316546


namespace NUMINAMATH_CALUDE_no_non_zero_integer_solution_l3165_316507

theorem no_non_zero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_non_zero_integer_solution_l3165_316507


namespace NUMINAMATH_CALUDE_train_car_count_l3165_316593

/-- Calculates the number of cars in a train given the observed data -/
def train_cars (cars_observed : ℕ) (observation_time : ℕ) (total_time : ℕ) : ℕ :=
  (cars_observed * total_time) / observation_time

/-- Theorem stating the number of cars in the train -/
theorem train_car_count :
  let cars_observed : ℕ := 8
  let observation_time : ℕ := 12  -- in seconds
  let total_time : ℕ := 3 * 60    -- 3 minutes converted to seconds
  train_cars cars_observed observation_time total_time = 120 := by
  sorry

#eval train_cars 8 12 (3 * 60)

end NUMINAMATH_CALUDE_train_car_count_l3165_316593


namespace NUMINAMATH_CALUDE_max_equilateral_triangle_area_in_rectangle_l3165_316596

theorem max_equilateral_triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a = 10 ∧ b = 11 →
  ∃ (area : ℝ),
  area = 221 * Real.sqrt 3 - 330 ∧
  (∀ (triangle_area : ℝ),
    (∃ (x y : ℝ),
      0 ≤ x ∧ x ≤ a ∧
      0 ≤ y ∧ y ≤ b ∧
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ area) :=
by sorry

end NUMINAMATH_CALUDE_max_equilateral_triangle_area_in_rectangle_l3165_316596


namespace NUMINAMATH_CALUDE_factorization_theorem_1_l3165_316555

theorem factorization_theorem_1 (x : ℝ) : 
  4 * (x - 2)^2 - 1 = (2*x - 3) * (2*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_1_l3165_316555


namespace NUMINAMATH_CALUDE_betty_and_sister_book_ratio_l3165_316529

theorem betty_and_sister_book_ratio : 
  ∀ (betty_books sister_books : ℕ),
    betty_books = 20 →
    betty_books + sister_books = 45 →
    (sister_books : ℚ) / betty_books = 5 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_betty_and_sister_book_ratio_l3165_316529


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3165_316572

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 + 20 * x - 24 = (d * x + e)^2 + f) → d * e = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3165_316572


namespace NUMINAMATH_CALUDE_cycling_speeds_l3165_316516

/-- Represents the cycling speeds of four people -/
structure CyclingGroup where
  henry_speed : ℝ
  liz_speed : ℝ
  jack_speed : ℝ
  tara_speed : ℝ

/-- The cycling group satisfies the given conditions -/
def satisfies_conditions (g : CyclingGroup) : Prop :=
  g.henry_speed = 5 ∧
  g.liz_speed = 3/4 * g.henry_speed ∧
  g.jack_speed = 6/5 * g.liz_speed ∧
  g.tara_speed = 9/8 * g.jack_speed

/-- Theorem stating the cycling speeds of Jack and Tara -/
theorem cycling_speeds (g : CyclingGroup) 
  (h : satisfies_conditions g) : 
  g.jack_speed = 4.5 ∧ g.tara_speed = 5.0625 := by
  sorry

#check cycling_speeds

end NUMINAMATH_CALUDE_cycling_speeds_l3165_316516


namespace NUMINAMATH_CALUDE_tiles_difference_8th_7th_l3165_316571

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n^2

/-- The theorem stating the difference in tiles between the 8th and 7th squares -/
theorem tiles_difference_8th_7th : 
  tiles_in_square 8 - tiles_in_square 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_8th_7th_l3165_316571


namespace NUMINAMATH_CALUDE_integer_coloring_theorem_l3165_316583

/-- A color type with four colors -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A coloring function that assigns a color to each integer -/
def coloring : ℤ → Color := sorry

theorem integer_coloring_theorem 
  (m n : ℤ) 
  (h_odd_m : Odd m) 
  (h_odd_n : Odd n) 
  (h_distinct : m ≠ n) 
  (h_sum_nonzero : m + n ≠ 0) :
  ∃ (a b : ℤ), 
    coloring a = coloring b ∧ 
    (a - b = m ∨ a - b = n ∨ a - b = m + n ∨ a - b = m - n) := by
  sorry

end NUMINAMATH_CALUDE_integer_coloring_theorem_l3165_316583


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3165_316512

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3165_316512


namespace NUMINAMATH_CALUDE_polygon_sides_l3165_316505

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 1800 → n = 10 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3165_316505


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3165_316510

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 7 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 4 ∧ B = 3 ∧ C = -1 ∧ D = -1 ∧ E = 42 ∧ F = 10 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3165_316510


namespace NUMINAMATH_CALUDE_optimal_path_to_island_l3165_316556

/-- Represents the optimal path problem for Hagrid to reach Harry Potter --/
theorem optimal_path_to_island (island_distance : ℝ) (shore_distance : ℝ) 
  (shore_speed : ℝ) (sea_speed : ℝ) :
  island_distance = 9 →
  shore_distance = 15 →
  shore_speed = 50 →
  sea_speed = 40 →
  ∃ (x : ℝ), x = 3 ∧ 
    ∀ (y : ℝ), y ≥ 0 → 
      (x / shore_speed + (Real.sqrt ((island_distance^2) + (shore_distance - x)^2)) / sea_speed) ≤
      (y / shore_speed + (Real.sqrt ((island_distance^2) + (shore_distance - y)^2)) / sea_speed) :=
by sorry


end NUMINAMATH_CALUDE_optimal_path_to_island_l3165_316556


namespace NUMINAMATH_CALUDE_bird_watching_problem_l3165_316532

theorem bird_watching_problem (total_watchers : Nat) (average_birds : Nat) 
  (first_watcher_birds : Nat) (second_watcher_birds : Nat) :
  total_watchers = 3 →
  average_birds = 9 →
  first_watcher_birds = 7 →
  second_watcher_birds = 11 →
  (total_watchers * average_birds - first_watcher_birds - second_watcher_birds) = 9 := by
  sorry

end NUMINAMATH_CALUDE_bird_watching_problem_l3165_316532


namespace NUMINAMATH_CALUDE_line_x_intercept_l3165_316550

/-- Given a line passing through points (2, -2) and (6, 6), its x-intercept is 3 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (f 2 = -2) → 
  (f 6 = 6) → 
  (∀ x y : ℝ, f y - f x = (y - x) * ((6 - (-2)) / (6 - 2))) →
  (∃ x : ℝ, f x = 0 ∧ x = 3) := by
sorry

end NUMINAMATH_CALUDE_line_x_intercept_l3165_316550


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3165_316525

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + y^2 + 4 / (x + y)^2 = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3165_316525


namespace NUMINAMATH_CALUDE_congruent_triangles_equal_perimeter_l3165_316563

/-- Represents a triangle -/
structure Triangle where
  perimeter : ℝ

/-- Two triangles are congruent -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

theorem congruent_triangles_equal_perimeter (t1 t2 : Triangle) 
  (h1 : Congruent t1 t2) (h2 : t1.perimeter = 5) : t2.perimeter = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_equal_perimeter_l3165_316563


namespace NUMINAMATH_CALUDE_equation_solution_l3165_316575

theorem equation_solution : ∃! x : ℝ, (3 / (x - 3) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3165_316575


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3165_316526

theorem solution_set_abs_inequality (x : ℝ) :
  (|x - 3| < 2) ↔ (1 < x ∧ x < 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3165_316526


namespace NUMINAMATH_CALUDE_largest_guaranteed_divisor_l3165_316590

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def Q (s : Finset ℕ) : ℕ := s.prod id

theorem largest_guaranteed_divisor :
  ∀ s : Finset ℕ, s ⊆ die_faces → s.card = 7 → 960 ∣ Q s ∧
  ∀ n : ℕ, n > 960 → ∃ t : Finset ℕ, t ⊆ die_faces ∧ t.card = 7 ∧ ¬(n ∣ Q t) :=
by sorry

end NUMINAMATH_CALUDE_largest_guaranteed_divisor_l3165_316590


namespace NUMINAMATH_CALUDE_increasing_with_properties_is_geometric_l3165_316513

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define property 1
def Property1 (a : Sequence) : Prop :=
  ∀ i j, i > j → ∃ m, a i ^ 2 / a j = a m

-- Define property 2
def Property2 (a : Sequence) : Prop :=
  ∀ n, n ≥ 3 → ∃ k l, k > l ∧ a n = a k ^ 2 / a l

-- Define increasing sequence
def IncreasingSequence (a : Sequence) : Prop :=
  ∀ n m, n < m → a n < a m

-- Define geometric sequence
def GeometricSequence (a : Sequence) : Prop :=
  ∃ r, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

-- Main theorem
theorem increasing_with_properties_is_geometric (a : Sequence) :
  IncreasingSequence a → Property1 a → Property2 a → GeometricSequence a :=
by
  sorry


end NUMINAMATH_CALUDE_increasing_with_properties_is_geometric_l3165_316513


namespace NUMINAMATH_CALUDE_binary_representation_of_500_l3165_316567

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_representation_of_500 :
  to_binary 500 = [true, false, false, true, true, true, true, true, true] :=
by sorry

end NUMINAMATH_CALUDE_binary_representation_of_500_l3165_316567


namespace NUMINAMATH_CALUDE_brick_length_calculation_l3165_316543

/-- Calculates the length of a brick given wall dimensions and brick count --/
theorem brick_length_calculation (wall_length wall_height wall_thickness : ℝ)
                                 (brick_width brick_height : ℝ) (brick_count : ℕ) :
  wall_length = 750 ∧ wall_height = 600 ∧ wall_thickness = 22.5 ∧
  brick_width = 11.25 ∧ brick_height = 6 ∧ brick_count = 6000 →
  ∃ (brick_length : ℝ),
    brick_length = 25 ∧
    wall_length * wall_height * wall_thickness =
    brick_length * brick_width * brick_height * brick_count :=
by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l3165_316543


namespace NUMINAMATH_CALUDE_e_pow_pi_gt_pi_pow_e_l3165_316506

/-- Prove that e^π > π^e, given that π > e -/
theorem e_pow_pi_gt_pi_pow_e : Real.exp π > π ^ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_e_pow_pi_gt_pi_pow_e_l3165_316506


namespace NUMINAMATH_CALUDE_jason_balloon_count_l3165_316515

/-- Calculates the final number of balloons Jason has after a series of changes. -/
def final_balloon_count (initial_violet : ℕ) (initial_red : ℕ) 
  (violet_given : ℕ) (red_given : ℕ) (violet_acquired : ℕ) : ℕ :=
  let remaining_violet := initial_violet - violet_given + violet_acquired
  let remaining_red := (initial_red - red_given) * 3
  remaining_violet + remaining_red

/-- Proves that Jason ends up with 35 balloons given the initial quantities and changes. -/
theorem jason_balloon_count : 
  final_balloon_count 15 12 3 5 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_jason_balloon_count_l3165_316515


namespace NUMINAMATH_CALUDE_range_of_a_l3165_316539

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ici 1 ∧ |x - a| + x - 4 ≤ 0) → a ∈ Set.Icc (-2) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3165_316539


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3165_316594

theorem fraction_equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3165_316594


namespace NUMINAMATH_CALUDE_cos_diff_symmetric_angles_l3165_316597

/-- Two angles are symmetric with respect to the origin if their difference is an odd multiple of π -/
def symmetric_angles (α β : Real) : Prop :=
  ∃ k : Int, β = α + (2 * k - 1) * Real.pi

/-- 
If the terminal sides of angles α and β are symmetric with respect to the origin O,
then cos(α - β) = -1
-/
theorem cos_diff_symmetric_angles (α β : Real) 
  (h : symmetric_angles α β) : Real.cos (α - β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_diff_symmetric_angles_l3165_316597


namespace NUMINAMATH_CALUDE_multiply_decimals_l3165_316589

theorem multiply_decimals : (2.4 : ℝ) * 0.2 = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l3165_316589


namespace NUMINAMATH_CALUDE_molecular_weight_Al2S3_l3165_316577

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

-- Define the composition of Al2S3
def Al_atoms_in_Al2S3 : ℕ := 2
def S_atoms_in_Al2S3 : ℕ := 3

-- Define the number of moles
def moles_Al2S3 : ℝ := 3

-- Theorem statement
theorem molecular_weight_Al2S3 :
  let molecular_weight_one_mole := Al_atoms_in_Al2S3 * atomic_weight_Al + S_atoms_in_Al2S3 * atomic_weight_S
  moles_Al2S3 * molecular_weight_one_mole = 450.42 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_Al2S3_l3165_316577


namespace NUMINAMATH_CALUDE_min_value_fraction_l3165_316580

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 2*y^2 + z^2) / (x*y + 3*y*z) ≥ 2*Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3165_316580


namespace NUMINAMATH_CALUDE_value_of_a_l3165_316554

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 8) 
  (h3 : d = 4) : 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3165_316554


namespace NUMINAMATH_CALUDE_triangle_theorem_l3165_316599

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.c * Real.sin t.A = t.a * Real.cos t.C)
  (h2 : 0 < t.A ∧ t.A < Real.pi)
  (h3 : 0 < t.B ∧ t.B < Real.pi)
  (h4 : 0 < t.C ∧ t.C < Real.pi)
  (h5 : t.A + t.B + t.C = Real.pi) : 
  (t.C = Real.pi / 4) ∧ 
  (∃ (max : Real), ∀ (A B : Real), 
    (0 < A ∧ A < 3 * Real.pi / 4) → 
    (B = 3 * Real.pi / 4 - A) → 
    (Real.sqrt 3 * Real.sin A - Real.cos (B + Real.pi / 4) ≤ max) ∧
    (max = 2)) ∧
  (Real.sqrt 3 * Real.sin (Real.pi / 3) - Real.cos (5 * Real.pi / 12 + Real.pi / 4) = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3165_316599


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3165_316587

theorem negative_fraction_comparison : -3/5 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3165_316587


namespace NUMINAMATH_CALUDE_ninth_power_sum_l3165_316528

/-- Given two real numbers m and n satisfying specific conditions, prove that m⁹ + n⁹ = 76 -/
theorem ninth_power_sum (m n : ℝ) 
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) : 
  m^9 + n^9 = 76 := by
  sorry

#check ninth_power_sum

end NUMINAMATH_CALUDE_ninth_power_sum_l3165_316528


namespace NUMINAMATH_CALUDE_horner_method_operations_l3165_316547

def f (x : ℝ) : ℝ := x^6 + 1

def horner_eval (x : ℝ) : ℝ := ((((((x * x + 0) * x + 0) * x + 0) * x + 0) * x + 0) * x + 1)

theorem horner_method_operations (x : ℝ) :
  (∃ (exp_count mult_count add_count : ℕ),
    horner_eval x = f x ∧
    exp_count = 0 ∧
    mult_count = 6 ∧
    add_count = 6) :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l3165_316547


namespace NUMINAMATH_CALUDE_inequality_implies_bound_l3165_316557

theorem inequality_implies_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 3, x^2 - a*x + 4 ≥ 0) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_bound_l3165_316557


namespace NUMINAMATH_CALUDE_smallest_x_value_l3165_316582

theorem smallest_x_value (x : ℝ) : 
  (((15 * x^2 - 40 * x + 20) / (4 * x - 3) + 7 * x) = (8 * x - 3)) →
  x ≥ (25 - Real.sqrt 141) / 22 ∧ 
  ∃ (y : ℝ), y = (25 - Real.sqrt 141) / 22 ∧ 
    ((15 * y^2 - 40 * y + 20) / (4 * y - 3) + 7 * y) = (8 * y - 3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3165_316582


namespace NUMINAMATH_CALUDE_prop_a_false_prop_b_false_prop_c_true_prop_d_false_false_propositions_l3165_316585

-- Proposition A
theorem prop_a_false : ¬(∀ x : ℝ, x^2 + 3 < 0) := by sorry

-- Proposition B
theorem prop_b_false : ¬(∀ x : ℕ, x^2 > 1) := by sorry

-- Proposition C
theorem prop_c_true : ∃ x : ℤ, x^5 < 1 := by sorry

-- Proposition D
theorem prop_d_false : ¬(∃ x : ℚ, x^2 = 3) := by sorry

-- Combined theorem
theorem false_propositions :
  (¬(∀ x : ℝ, x^2 + 3 < 0)) ∧
  (¬(∀ x : ℕ, x^2 > 1)) ∧
  (∃ x : ℤ, x^5 < 1) ∧
  (¬(∃ x : ℚ, x^2 = 3)) := by sorry

end NUMINAMATH_CALUDE_prop_a_false_prop_b_false_prop_c_true_prop_d_false_false_propositions_l3165_316585


namespace NUMINAMATH_CALUDE_nth_equation_l3165_316561

theorem nth_equation (n : ℕ) : ((n + 1)^2 - n^2 - 1) / 2 = n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l3165_316561


namespace NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_square_sum_l3165_316534

/-- Given a right triangle with legs a and b, hypotenuse c, and altitude h drawn to the hypotenuse,
    prove that 1/h^2 = 1/a^2 + 1/b^2. -/
theorem right_triangle_altitude_reciprocal_square_sum 
  (a b c h : ℝ) 
  (h_positive : h > 0)
  (a_positive : a > 0)
  (b_positive : b > 0)
  (right_triangle : a^2 + b^2 = c^2)
  (altitude_formula : h * c = a * b) : 
  1 / h^2 = 1 / a^2 + 1 / b^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_square_sum_l3165_316534


namespace NUMINAMATH_CALUDE_ones_digit_largest_power_of_three_dividing_factorial_ones_digit_largest_power_of_three_dividing_27_factorial_l3165_316535

theorem ones_digit_largest_power_of_three_dividing_factorial : ℕ → Prop :=
  fun n => 
    let factorial := Nat.factorial n
    let largest_power := Nat.log 3 factorial
    (3^largest_power % 10 = 3 ∧ n = 27)

-- The proof
theorem ones_digit_largest_power_of_three_dividing_27_factorial :
  ones_digit_largest_power_of_three_dividing_factorial 27 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_largest_power_of_three_dividing_factorial_ones_digit_largest_power_of_three_dividing_27_factorial_l3165_316535


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l3165_316517

/-- Given a quadratic function y = (m-2)x^2 + 2mx - (3-m), prove that the range of m
    satisfying all conditions is 2 < m < 3 --/
theorem quadratic_function_m_range (m : ℝ) : 
  let f (x : ℝ) := (m - 2) * x^2 + 2 * m * x - (3 - m)
  let vertex_x := -m / (m - 2)
  let vertex_y := (-5 * m + 6) / (m - 2)
  (∀ x, (m - 2) * x^2 + 2 * m * x - (3 - m) = f x) →
  (vertex_x < 0 ∧ vertex_y < 0) →
  (m - 2 > 0) →
  (-(3 - m) < 0) →
  (2 < m ∧ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l3165_316517


namespace NUMINAMATH_CALUDE_gcd_of_44_33_55_l3165_316524

/-- The greatest common divisor of 44, 33, and 55 is 11. -/
theorem gcd_of_44_33_55 : Nat.gcd 44 (Nat.gcd 33 55) = 11 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_44_33_55_l3165_316524


namespace NUMINAMATH_CALUDE_number_of_nickels_l3165_316566

def pennies : ℕ := 123
def dimes : ℕ := 35
def quarters : ℕ := 26
def family_members : ℕ := 5
def ice_cream_cost_per_member : ℚ := 3
def leftover_cents : ℕ := 48

def total_ice_cream_cost : ℚ := family_members * ice_cream_cost_per_member

def total_without_nickels : ℚ := 
  (pennies : ℚ) / 100 + (dimes : ℚ) / 10 + (quarters : ℚ) / 4

theorem number_of_nickels : 
  ∃ (n : ℕ), total_without_nickels + (n : ℚ) / 20 = total_ice_cream_cost + (leftover_cents : ℚ) / 100 ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_number_of_nickels_l3165_316566


namespace NUMINAMATH_CALUDE_vector_independence_l3165_316558

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![m - 1, m + 3]

theorem vector_independence (m : ℝ) :
  LinearIndependent ℝ ![vector_a, vector_b m] ↔ m ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_independence_l3165_316558


namespace NUMINAMATH_CALUDE_fraction_simplification_l3165_316508

theorem fraction_simplification : 
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3165_316508


namespace NUMINAMATH_CALUDE_multiply_three_point_five_by_zero_point_twenty_five_l3165_316549

theorem multiply_three_point_five_by_zero_point_twenty_five : 3.5 * 0.25 = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_point_five_by_zero_point_twenty_five_l3165_316549


namespace NUMINAMATH_CALUDE_simplify_expression_l3165_316591

theorem simplify_expression (x : ℝ) : 1 - (2 + (1 - (1 + (2 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3165_316591


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l3165_316553

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l3165_316553


namespace NUMINAMATH_CALUDE_debbys_candy_l3165_316500

theorem debbys_candy (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ)
  (h1 : sister_candy = 42)
  (h2 : eaten_candy = 35)
  (h3 : remaining_candy = 39) :
  ∃ (debby_candy : ℕ), debby_candy + sister_candy - eaten_candy = remaining_candy ∧ debby_candy = 32 :=
by sorry

end NUMINAMATH_CALUDE_debbys_candy_l3165_316500


namespace NUMINAMATH_CALUDE_f_value_at_8pi_3_l3165_316511

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_8pi_3 :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, f (x + π) = f x) →  -- f has period π
  (∀ x ∈ Set.Icc 0 (π/2), f x = Real.sqrt 3 * Real.tan x - 1) →  -- definition of f on [0, π/2)
  f (8*π/3) = 2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_8pi_3_l3165_316511


namespace NUMINAMATH_CALUDE_y_derivative_l3165_316573

open Real

noncomputable def y (x : ℝ) : ℝ := 2 * (cos x / sin x ^ 4) + 3 * (cos x / sin x ^ 2)

theorem y_derivative (x : ℝ) (h : sin x ≠ 0) : 
  deriv y x = 3 * (1 / sin x) - 8 * (1 / sin x) ^ 5 := by
sorry

end NUMINAMATH_CALUDE_y_derivative_l3165_316573


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3165_316595

def is_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r

theorem geometric_sequence_property (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₃ ≠ 0) (h₄ : a₄ ≠ 0) :
  (is_geometric_sequence a₁ a₂ a₃ a₄ → a₁ * a₄ = a₂ * a₃) ∧
  (∃ b₁ b₂ b₃ b₄ : ℝ, b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ b₃ ≠ 0 ∧ b₄ ≠ 0 ∧
    b₁ * b₄ = b₂ * b₃ ∧ ¬is_geometric_sequence b₁ b₂ b₃ b₄) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3165_316595


namespace NUMINAMATH_CALUDE_dice_sides_for_given_probability_l3165_316504

theorem dice_sides_for_given_probability (n : ℕ+) : 
  (((6 : ℝ) / (n : ℝ)^2)^2 = 0.027777777777777776) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_dice_sides_for_given_probability_l3165_316504


namespace NUMINAMATH_CALUDE_calvins_bug_collection_l3165_316565

theorem calvins_bug_collection (roaches scorpions caterpillars crickets : ℕ) : 
  roaches = 12 →
  scorpions = 3 →
  caterpillars = 2 * scorpions →
  roaches + scorpions + caterpillars + crickets = 27 →
  crickets * 2 = roaches :=
by sorry

end NUMINAMATH_CALUDE_calvins_bug_collection_l3165_316565


namespace NUMINAMATH_CALUDE_charity_sale_result_l3165_316576

/-- Represents the number and prices of shirts in a charity sale --/
structure ShirtSale where
  total_shirts : ℕ
  total_cost : ℕ
  black_wholesale : ℕ
  black_retail : ℕ
  white_wholesale : ℕ
  white_retail : ℕ

/-- Calculates the number of black and white shirts and the total profit --/
def calculate_shirts_and_profit (sale : ShirtSale) : 
  (ℕ × ℕ × ℕ) := sorry

/-- Theorem stating the correct results for the given shirt sale --/
theorem charity_sale_result (sale : ShirtSale) 
  (h1 : sale.total_shirts = 200)
  (h2 : sale.total_cost = 3500)
  (h3 : sale.black_wholesale = 25)
  (h4 : sale.black_retail = 50)
  (h5 : sale.white_wholesale = 15)
  (h6 : sale.white_retail = 35) :
  calculate_shirts_and_profit sale = (50, 150, 4250) := by sorry

end NUMINAMATH_CALUDE_charity_sale_result_l3165_316576


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l3165_316538

theorem solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (x > a ∧ x > 1) ↔ x > 1) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l3165_316538


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l3165_316542

theorem pentagon_angle_measure :
  ∀ (a b c d e : ℝ),
  a + b + c + d + e = 540 →
  a = 111 →
  b = 113 →
  c = 92 →
  d = 128 →
  e = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l3165_316542
