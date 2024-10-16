import Mathlib

namespace NUMINAMATH_CALUDE_computer_accessories_cost_l3220_322034

/-- Proves that the amount spent on computer accessories is $12 -/
theorem computer_accessories_cost (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 48 →
  snack_cost = 8 →
  remaining_amount = initial_amount / 2 + 4 →
  initial_amount - (remaining_amount + snack_cost) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_computer_accessories_cost_l3220_322034


namespace NUMINAMATH_CALUDE_exam_score_difference_l3220_322014

theorem exam_score_difference (score_65 score_75 score_85 score_95 : ℝ)
  (percent_65 percent_75 percent_85 percent_95 : ℝ)
  (h1 : score_65 = 65)
  (h2 : score_75 = 75)
  (h3 : score_85 = 85)
  (h4 : score_95 = 95)
  (h5 : percent_65 = 0.15)
  (h6 : percent_75 = 0.40)
  (h7 : percent_85 = 0.20)
  (h8 : percent_95 = 0.25)
  (h9 : percent_65 + percent_75 + percent_85 + percent_95 = 1) :
  let mean := score_65 * percent_65 + score_75 * percent_75 + score_85 * percent_85 + score_95 * percent_95
  let median := score_75
  mean - median = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_difference_l3220_322014


namespace NUMINAMATH_CALUDE_inequality_solution_l3220_322088

theorem inequality_solution :
  let f (x : ℝ) := x^3 - 3*x - 3/x + 1/x^3 + 5
  ∀ x : ℝ, (202 * Real.sqrt (f x) ≤ 0) ↔
    (x = (-1 - Real.sqrt 21 + Real.sqrt (2 * Real.sqrt 21 + 6)) / 4 ∨
     x = (-1 - Real.sqrt 21 - Real.sqrt (2 * Real.sqrt 21 + 6)) / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3220_322088


namespace NUMINAMATH_CALUDE_inequality_proof_l3220_322052

theorem inequality_proof (a b c d e f : ℕ) 
  (h1 : (a : ℚ) / b > (c : ℚ) / d)
  (h2 : (c : ℚ) / d > (e : ℚ) / f)
  (h3 : a * f - b * e = 1) :
  d ≥ b + f := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3220_322052


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3220_322067

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 12) → cows = 6 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3220_322067


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3220_322000

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  x + 2*y ≥ 9 ∧ ∀ M : ℝ, ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 1/x' + 2/y' = 1 ∧ x' + 2*y' > M :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3220_322000


namespace NUMINAMATH_CALUDE_problem_solution_l3220_322017

theorem problem_solution : 
  (2023^0 + |1 - Real.sqrt 2| - Real.sqrt 3 * Real.sqrt 6 = -2 * Real.sqrt 2) ∧
  ((Real.sqrt 5 - 1)^2 + Real.sqrt 5 * (Real.sqrt 5 + 2) = 11) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3220_322017


namespace NUMINAMATH_CALUDE_athletes_on_second_floor_l3220_322077

/-- Proves that given a hotel with three floors housing 38 athletes, where 26 athletes are on the first and second floors, and 27 athletes are on the second and third floors, the number of athletes on the second floor is 15. -/
theorem athletes_on_second_floor 
  (total_athletes : ℕ) 
  (first_second : ℕ) 
  (second_third : ℕ) 
  (h1 : total_athletes = 38) 
  (h2 : first_second = 26) 
  (h3 : second_third = 27) : 
  ∃ (first second third : ℕ), 
    first + second + third = total_athletes ∧ 
    first + second = first_second ∧ 
    second + third = second_third ∧ 
    second = 15 :=
by sorry

end NUMINAMATH_CALUDE_athletes_on_second_floor_l3220_322077


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l3220_322082

/-- Parabola M: y^2 = 4x -/
def parabola_M (x y : ℝ) : Prop := y^2 = 4*x

/-- Circle N: (x-1)^2 + y^2 = r^2 -/
def circle_N (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

/-- Line l passing through (1, 0) -/
def line_l (m x y : ℝ) : Prop := x = m * y + 1

/-- Condition for |AC| = |BD| -/
def equal_distances (y₁ y₂ y₃ y₄ : ℝ) : Prop := |y₁ - y₃| = |y₂ - y₄|

/-- Main theorem -/
theorem parabola_circle_intersection (r : ℝ) :
  (r > 0) →
  (∃ (m₁ m₂ m₃ : ℝ),
    (∀ (m : ℝ), m ≠ m₁ ∧ m ≠ m₂ ∧ m ≠ m₃ →
      ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
        parabola_M x₁ y₁ ∧ parabola_M x₂ y₂ ∧
        circle_N x₃ y₃ r ∧ circle_N x₄ y₄ r ∧
        line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
        line_l m x₃ y₃ ∧ line_l m x₄ y₄ ∧
        equal_distances y₁ y₂ y₃ y₄))) →
  r ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l3220_322082


namespace NUMINAMATH_CALUDE_max_distance_theorem_l3220_322024

/-- Given points in a 2D Cartesian coordinate system, prove the maximum distance -/
theorem max_distance_theorem (x y : ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (0, Real.sqrt 3)
  let C : ℝ × ℝ := (3, 0)
  let D : ℝ × ℝ := (x, y)
  (x - 3)^2 + y^2 = 1 →
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + y₀^2 = 1 ∧ 
    ∀ (x' y' : ℝ), (x' - 3)^2 + y'^2 = 1 → 
      ((x' - 1)^2 + (y' + Real.sqrt 3)^2) ≤ ((x₀ - 1)^2 + (y₀ + Real.sqrt 3)^2)) ∧
  ((x₀ - 1)^2 + (y₀ + Real.sqrt 3)^2) = (Real.sqrt 7 + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_theorem_l3220_322024


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3220_322048

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3220_322048


namespace NUMINAMATH_CALUDE_cotton_collection_rate_l3220_322036

/-- The amount of cotton (in kg) that can be collected by a given number of workers in 2 days -/
def cotton_collected (w : ℕ) : ℝ := w * 8

theorem cotton_collection_rate 
  (h1 : 3 * (48 / 4) = 3 * 12)  -- 3 workers collect 48 kg in 4 days
  (h2 : 9 * 8 = 72) :  -- 9 workers collect 72 kg in 2 days
  ∀ w : ℕ, cotton_collected w = w * 8 := by
  sorry

#check cotton_collection_rate

end NUMINAMATH_CALUDE_cotton_collection_rate_l3220_322036


namespace NUMINAMATH_CALUDE_problem_solution_l3220_322025

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a * b = 2) 
  (h2 : a / (a + b^2) + b / (b + a^2) = 7/8) : 
  a^6 + b^6 = 128 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3220_322025


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3220_322071

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 3 * p.2 = 7}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = -1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1/2, 3/2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3220_322071


namespace NUMINAMATH_CALUDE_expression_evaluation_l3220_322028

/-- Given x = -2 and y = 1/2, prove that 2(x^2y + xy^2) - 2(x^2y - 1) - 3xy^2 - 2 evaluates to 1/2 -/
theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1/2) :
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3220_322028


namespace NUMINAMATH_CALUDE_intersection_polygon_exists_and_unique_l3220_322070

/- Define the cube and points -/
def cube_edge_length : ℝ := 30

/- Define points on cube edges -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨cube_edge_length, 0, 0⟩
def C : Point3D := ⟨cube_edge_length, 0, cube_edge_length⟩
def D : Point3D := ⟨cube_edge_length, cube_edge_length, cube_edge_length⟩

def P : Point3D := ⟨10, 0, 0⟩
def Q : Point3D := ⟨cube_edge_length, 0, 10⟩
def R : Point3D := ⟨cube_edge_length, 15, cube_edge_length⟩

/- Define the plane PQR -/
def plane_PQR (x y z : ℝ) : Prop := 2*x + y - 2*z = 15

/- Define the cube -/
def in_cube (p : Point3D) : Prop :=
  0 ≤ p.x ∧ p.x ≤ cube_edge_length ∧
  0 ≤ p.y ∧ p.y ≤ cube_edge_length ∧
  0 ≤ p.z ∧ p.z ≤ cube_edge_length

/- Theorem statement -/
theorem intersection_polygon_exists_and_unique :
  ∃! polygon : Set Point3D,
    (∀ p ∈ polygon, in_cube p ∧ plane_PQR p.x p.y p.z) ∧
    (∀ p, in_cube p ∧ plane_PQR p.x p.y p.z → p ∈ polygon) :=
sorry

end NUMINAMATH_CALUDE_intersection_polygon_exists_and_unique_l3220_322070


namespace NUMINAMATH_CALUDE_y_percent_of_x_l3220_322056

theorem y_percent_of_x (x y : ℝ) (h : 0.6 * (x - y) = 0.3 * (x + y)) : y / x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_x_l3220_322056


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l3220_322058

/-- If θ is an acute angle, then 2θ is a positive angle less than 180°. -/
theorem double_acute_angle_range (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  0 < 2 * θ ∧ 2 * θ < Real.pi := by sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l3220_322058


namespace NUMINAMATH_CALUDE_best_competitor_is_man_l3220_322030

-- Define the set of competitors
inductive Competitor
| Man
| Sister
| Son
| Niece

-- Define the gender type
inductive Gender
| Male
| Female

-- Define the function to get the gender of a competitor
def gender : Competitor → Gender
  | Competitor.Man => Gender.Male
  | Competitor.Sister => Gender.Female
  | Competitor.Son => Gender.Male
  | Competitor.Niece => Gender.Female

-- Define the function to get the twin of a competitor
def twin : Competitor → Competitor
  | Competitor.Man => Competitor.Sister
  | Competitor.Sister => Competitor.Man
  | Competitor.Son => Competitor.Niece
  | Competitor.Niece => Competitor.Son

-- Define the age equality relation
def sameAge : Competitor → Competitor → Prop := sorry

-- State the theorem
theorem best_competitor_is_man :
  ∃ (best worst : Competitor),
    (twin best ∈ [Competitor.Man, Competitor.Sister, Competitor.Son, Competitor.Niece]) ∧
    (gender (twin best) ≠ gender worst) ∧
    (sameAge best worst) →
    best = Competitor.Man :=
  sorry

end NUMINAMATH_CALUDE_best_competitor_is_man_l3220_322030


namespace NUMINAMATH_CALUDE_quadratic_equality_l3220_322084

-- Define the two quadratic functions
def f (a c x : ℝ) : ℝ := a * (x - 2)^2 + c
def g (b x : ℝ) : ℝ := (2*x - 5) * (x - b)

-- State the theorem
theorem quadratic_equality (a c b : ℝ) : 
  (∀ x, f a c x = g b x) → b = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equality_l3220_322084


namespace NUMINAMATH_CALUDE_number_of_red_balls_l3220_322029

theorem number_of_red_balls (blue : ℕ) (green : ℕ) (red : ℕ) : 
  blue = 3 → green = 1 → (red : ℚ) / (blue + green + red : ℚ) = 1 / 2 → red = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_red_balls_l3220_322029


namespace NUMINAMATH_CALUDE_product_coefficient_equality_l3220_322091

theorem product_coefficient_equality (m : ℝ) : 
  (∃ a b c d : ℝ, (x^2 - m*x + 2) * (2*x + 1) = a*x^3 + b*x^2 + b*x + d) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_coefficient_equality_l3220_322091


namespace NUMINAMATH_CALUDE_dwarf_truth_count_l3220_322096

/-- Represents the number of dwarfs who tell the truth -/
def truthful_dwarfs : ℕ := sorry

/-- Represents the number of dwarfs who lie -/
def lying_dwarfs : ℕ := sorry

/-- The total number of dwarfs -/
def total_dwarfs : ℕ := 10

/-- The number of dwarfs who raised their hands for vanilla ice cream -/
def vanilla_hands : ℕ := 10

/-- The number of dwarfs who raised their hands for chocolate ice cream -/
def chocolate_hands : ℕ := 5

/-- The number of dwarfs who raised their hands for fruit ice cream -/
def fruit_hands : ℕ := 1

theorem dwarf_truth_count :
  truthful_dwarfs + lying_dwarfs = total_dwarfs ∧
  truthful_dwarfs + 2 * lying_dwarfs = vanilla_hands + chocolate_hands + fruit_hands ∧
  truthful_dwarfs = 4 := by sorry

end NUMINAMATH_CALUDE_dwarf_truth_count_l3220_322096


namespace NUMINAMATH_CALUDE_polynomial_equality_l3220_322063

theorem polynomial_equality (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 = x^3) →
  (a = 1 ∧ a₂ = 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3220_322063


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3220_322006

/-- Given a circle intersected by three equally spaced parallel lines creating
    chords of lengths 40, 40, and 36, the distance between adjacent lines is √38. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 40 ∧ 
    chord3 = 36 ∧ 
    chord1^2 = 4 * (r^2 - (d/2)^2) ∧ 
    chord2^2 = 4 * (r^2 - (3*d/2)^2) ∧ 
    chord3^2 = 4 * (r^2 - d^2)) → 
  d = Real.sqrt 38 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3220_322006


namespace NUMINAMATH_CALUDE_base_five_last_digit_l3220_322064

theorem base_five_last_digit (n : ℕ) (h : n = 89) : n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_five_last_digit_l3220_322064


namespace NUMINAMATH_CALUDE_square_difference_equality_l3220_322008

theorem square_difference_equality : 1007^2 - 995^2 - 1005^2 + 997^2 = 8008 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3220_322008


namespace NUMINAMATH_CALUDE_original_number_proof_l3220_322042

theorem original_number_proof : 
  ∃ x : ℝ, x * 16 = 3408 ∧ x = 213 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3220_322042


namespace NUMINAMATH_CALUDE_cannot_form_right_triangle_l3220_322093

theorem cannot_form_right_triangle : ¬ (9^2 + 16^2 = 25^2) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_right_triangle_l3220_322093


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l3220_322085

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for the union of complement of B and A
theorem union_complement_B_A : (𝒰 \ B) ∪ A = {x | x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l3220_322085


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3220_322005

/-- Given planar vectors a and b satisfying the conditions,
    prove that the cosine of the angle between them is 1/2 -/
theorem cosine_of_angle_between_vectors
  (a b : ℝ × ℝ)  -- Planar vectors represented as pairs of real numbers
  (h1 : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5)  -- a · (a + b) = 5
  (h2 : a.1^2 + a.2^2 = 4)  -- |a| = 2
  (h3 : b.1^2 + b.2^2 = 1)  -- |b| = 1
  : (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3220_322005


namespace NUMINAMATH_CALUDE_condition_A_sufficient_not_necessary_l3220_322027

/-- Condition A: a > 1 and b > 1 -/
def condition_A (a b : ℝ) : Prop := a > 1 ∧ b > 1

/-- Condition B: a + b > 2 and ab > 1 -/
def condition_B (a b : ℝ) : Prop := a + b > 2 ∧ a * b > 1

theorem condition_A_sufficient_not_necessary :
  (∀ a b : ℝ, condition_A a b → condition_B a b) ∧
  (∃ a b : ℝ, condition_B a b ∧ ¬condition_A a b) :=
by sorry

end NUMINAMATH_CALUDE_condition_A_sufficient_not_necessary_l3220_322027


namespace NUMINAMATH_CALUDE_shoes_sold_l3220_322072

theorem shoes_sold (large medium small left : ℕ) 
  (h1 : large = 22)
  (h2 : medium = 50)
  (h3 : small = 24)
  (h4 : left = 13) : 
  large + medium + small - left = 83 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_l3220_322072


namespace NUMINAMATH_CALUDE_circle_equation_l3220_322022

/-- The equation of a circle with center (0, 4) passing through (3, 0) is x² + (y - 4)² = 25 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ x^2 + (y - 4)^2 = r^2) ∧ 
  (3^2 + (0 - 4)^2 = x^2 + (y - 4)^2) → 
  x^2 + (y - 4)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l3220_322022


namespace NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l3220_322033

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13]

def remove_element (l : List ℕ) (n : ℕ) : List ℕ :=
  l.filter (λ x => x ≠ n)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem remove_one_gives_average_seven_point_five :
  average (remove_element original_list 1) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l3220_322033


namespace NUMINAMATH_CALUDE_number_relationship_l3220_322059

theorem number_relationship (n : ℚ) : n = 25 / 3 → (6 * n - 10) - 3 * n = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l3220_322059


namespace NUMINAMATH_CALUDE_population_difference_l3220_322098

/-- The population difference between thrice Willowdale and Roseville -/
theorem population_difference (willowdale roseville suncity : ℕ) : 
  willowdale = 2000 →
  suncity = 12000 →
  suncity = 2 * roseville + 1000 →
  roseville < 3 * willowdale →
  3 * willowdale - roseville = 500 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_l3220_322098


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3220_322011

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3220_322011


namespace NUMINAMATH_CALUDE_beth_bought_ten_cans_of_corn_l3220_322074

/-- The number of cans of corn Beth bought -/
def cans_of_corn : ℕ := sorry

/-- The number of cans of peas Beth bought -/
def cans_of_peas : ℕ := 35

/-- The relationship between cans of peas and cans of corn -/
axiom peas_corn_relation : cans_of_peas = 15 + 2 * cans_of_corn

theorem beth_bought_ten_cans_of_corn : cans_of_corn = 10 := by sorry

end NUMINAMATH_CALUDE_beth_bought_ten_cans_of_corn_l3220_322074


namespace NUMINAMATH_CALUDE_periodic_sequence_property_l3220_322043

def is_odd (n : Int) : Prop := ∃ k, n = 2 * k + 1

def sequence_property (a : ℕ → Int) : Prop :=
  ∀ n : ℕ, ∀ α : ℕ, ∀ k : Int,
    (a n = 2^α * k ∧ is_odd k) → a (n + 1) = 2^α - k

def is_periodic (a : ℕ → Int) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + d) = a n

theorem periodic_sequence_property (a : ℕ → Int) 
  (h1 : ∀ n : ℕ, a n ≠ 0)
  (h2 : sequence_property a)
  (h3 : is_periodic a) :
  ∀ n : ℕ, a (n + 2) = a n :=
sorry

end NUMINAMATH_CALUDE_periodic_sequence_property_l3220_322043


namespace NUMINAMATH_CALUDE_coefficient_of_quadratic_term_l3220_322032

/-- The coefficient of the quadratic term in a quadratic equation ax^2 + bx + c = 0 -/
def quadratic_coefficient (a b c : ℝ) : ℝ := a

theorem coefficient_of_quadratic_term :
  quadratic_coefficient (-5) 5 6 = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_quadratic_term_l3220_322032


namespace NUMINAMATH_CALUDE_king_high_school_teachers_l3220_322038

/-- The number of students at King High School -/
def num_students : ℕ := 1500

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 3

/-- The number of students in each class -/
def students_per_class : ℕ := 35

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- The number of teachers at King High School -/
def num_teachers : ℕ := 86

theorem king_high_school_teachers : 
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = num_teachers := by
  sorry

end NUMINAMATH_CALUDE_king_high_school_teachers_l3220_322038


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3220_322044

theorem min_value_expression (x y : ℝ) (hx : x > 4) (hy : y > 5) :
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5) ≥ 71 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 4) (hy : y > 5) :
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5) = 71 ↔ x = 9/2 ∧ y = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3220_322044


namespace NUMINAMATH_CALUDE_b2f_to_decimal_l3220_322066

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a hexadecimal digit to its decimal value --/
def hexToDecimal (d : HexDigit) : ℕ :=
  match d with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Converts a list of hexadecimal digits to its decimal value --/
def hexListToDecimal (l : List HexDigit) : ℕ :=
  l.enum.foldr (fun (i, d) acc => acc + (hexToDecimal d) * (16 ^ i)) 0

theorem b2f_to_decimal :
  hexListToDecimal [HexDigit.B, HexDigit.D2, HexDigit.F] = 2863 := by
  sorry

end NUMINAMATH_CALUDE_b2f_to_decimal_l3220_322066


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_radius_l3220_322095

/-- Given a quadrilateral ABCD inscribed in a circle with diagonals intersecting at M,
    where AB = a, CD = b, and ∠AMB = α, the radius R of the circle is as follows. -/
theorem inscribed_quadrilateral_radius 
  (a b : ℝ) (α : ℝ) (ha : a > 0) (hb : b > 0) (hα : 0 < α ∧ α < π) :
  ∃ (R : ℝ), R = (Real.sqrt (a^2 + b^2 + 2*a*b*(Real.cos α))) / (2 * Real.sin α) :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_radius_l3220_322095


namespace NUMINAMATH_CALUDE_min_value_of_f_l3220_322092

/-- The quadratic function f(x) = 8x^2 - 32x + 2023 -/
def f (x : ℝ) : ℝ := 8 * x^2 - 32 * x + 2023

/-- Theorem stating that the minimum value of f(x) is 1991 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = 1991 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3220_322092


namespace NUMINAMATH_CALUDE_total_players_is_51_l3220_322045

/-- The number of cricket players -/
def cricket_players : ℕ := 10

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 16

/-- The number of softball players -/
def softball_players : ℕ := 13

/-- Theorem stating that the total number of players is 51 -/
theorem total_players_is_51 :
  cricket_players + hockey_players + football_players + softball_players = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_players_is_51_l3220_322045


namespace NUMINAMATH_CALUDE_some_number_value_l3220_322051

theorem some_number_value (x : ℝ) : 40 + x * 12 / (180 / 3) = 41 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3220_322051


namespace NUMINAMATH_CALUDE_inequality_solution_l3220_322003

-- Define the inequality function
def f (x : ℝ) : ℝ := (x^2 - 4) * (x - 6)^2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

-- Theorem stating that the solution set is correct
theorem inequality_solution : 
  {x : ℝ | f x ≤ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3220_322003


namespace NUMINAMATH_CALUDE_f_sum_2006_2007_l3220_322054

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom f_1 : f 1 = 2

-- State the theorem
theorem f_sum_2006_2007 : f 2006 + f 2007 = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_2006_2007_l3220_322054


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3220_322062

theorem quadratic_one_solution (m : ℚ) : 
  (∃! y, 3 * y^2 - 7 * y + m = 0) ↔ m = 49/12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3220_322062


namespace NUMINAMATH_CALUDE_sin_15_and_tan_75_l3220_322065

theorem sin_15_and_tan_75 :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.tan (75 * π / 180) = 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_15_and_tan_75_l3220_322065


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_two_l3220_322039

theorem no_solution_iff_m_eq_neg_two (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (x - 5) / (x - 3) ≠ m / (x - 3) + 2) ↔ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_two_l3220_322039


namespace NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_plane_implies_parallel_l3220_322073

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)

-- Axiom for transitivity of parallel lines
axiom parallel_trans (a b c : Line) : parallel a b → parallel b c → parallel a c

-- Axiom for perpendicular lines to the same plane being parallel
axiom perpendicular_plane_parallel (a b : Line) (γ : Plane) : 
  perpendicularToPlane a γ → perpendicularToPlane b γ → parallel a b

-- Theorem 1: If two lines are parallel to a third line, then they are parallel to each other
theorem parallel_transitivity (a b c : Line) : 
  parallel a b → parallel b c → parallel a c :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel to each other
theorem perpendicular_plane_implies_parallel (a b : Line) (γ : Plane) :
  perpendicularToPlane a γ → perpendicularToPlane b γ → parallel a b :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_plane_implies_parallel_l3220_322073


namespace NUMINAMATH_CALUDE_number_ordering_l3220_322080

theorem number_ordering (a b c : ℝ) : 
  a = 9^(1/3) → b = 3^(2/5) → c = 4^(1/5) → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l3220_322080


namespace NUMINAMATH_CALUDE_competition_result_l3220_322046

/-- Represents the scores of contestants in a mathematics competition. -/
structure Scores where
  ann : ℝ
  bill : ℝ
  carol : ℝ
  dick : ℝ
  nonnegative : 0 ≤ ann ∧ 0 ≤ bill ∧ 0 ≤ carol ∧ 0 ≤ dick

/-- Conditions of the mathematics competition. -/
def CompetitionConditions (s : Scores) : Prop :=
  s.bill + s.dick = 2 * s.ann ∧
  s.ann + s.carol < s.bill + s.dick ∧
  s.ann < s.bill + s.carol

/-- The order of contestants from highest to lowest score. -/
def CorrectOrder (s : Scores) : Prop :=
  s.dick > s.bill ∧ s.bill > s.ann ∧ s.ann > s.carol

theorem competition_result (s : Scores) (h : CompetitionConditions s) : CorrectOrder s := by
  sorry

end NUMINAMATH_CALUDE_competition_result_l3220_322046


namespace NUMINAMATH_CALUDE_M_on_y_axis_MN_parallel_to_y_axis_l3220_322041

-- Define the point M
def M (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

-- Define the point N
def N : ℝ × ℝ := (-3, 2)

-- Theorem 1: If M lies on the y-axis, then m = 1
theorem M_on_y_axis (m : ℝ) : M m = (0, M m).2 → m = 1 := by sorry

-- Theorem 2: If MN is parallel to the y-axis, then the length of MN is 3
theorem MN_parallel_to_y_axis (m : ℝ) : 
  (M m).1 = N.1 → abs ((M m).2 - N.2) = 3 := by sorry

end NUMINAMATH_CALUDE_M_on_y_axis_MN_parallel_to_y_axis_l3220_322041


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_two_alpha_l3220_322026

theorem cos_two_pi_thirds_minus_two_alpha (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.cos ((2 * π) / 3 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_two_alpha_l3220_322026


namespace NUMINAMATH_CALUDE_parallel_lines_length_l3220_322037

/-- Given parallel lines AB, CD, and GH, where AB = 240 cm and CD = 160 cm, prove that the length of GH is 320/3 cm. -/
theorem parallel_lines_length (AB CD GH : ℝ) : 
  AB = 240 → CD = 160 → (GH / CD = CD / AB) → GH = 320 / 3 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l3220_322037


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l3220_322075

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

-- State the theorem
theorem even_increasing_inequality (h1 : is_even f) (h2 : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l3220_322075


namespace NUMINAMATH_CALUDE_line_perpendicular_theorem_l3220_322002

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem line_perpendicular_theorem
  (m n : Line3D) (α β : Plane3D)
  (h1 : ¬ parallel_planes α β)
  (h2 : perpendicular m α)
  (h3 : ¬ parallel n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_theorem_l3220_322002


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l3220_322020

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem factorial_sum_remainder (n : ℕ) : 
  n ≥ 50 → sum_factorials n % 25 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 25 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l3220_322020


namespace NUMINAMATH_CALUDE_intersection_condition_subset_complement_condition_l3220_322012

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 2 ≤ x ∧ x ≤ m + 2}

theorem intersection_condition (m : ℝ) :
  A ∩ B m = {x : ℝ | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

theorem subset_complement_condition (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_subset_complement_condition_l3220_322012


namespace NUMINAMATH_CALUDE_pitchers_prepared_is_six_l3220_322001

/-- Represents the number of glasses of lemonade a single pitcher can serve. -/
def glasses_per_pitcher : ℕ := 5

/-- Represents the total number of glasses of lemonade served. -/
def total_glasses_served : ℕ := 30

/-- Calculates the number of pitchers needed to serve the given number of glasses. -/
def pitchers_needed (total_glasses : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  total_glasses / glasses_per_pitcher

/-- Proves that the number of pitchers prepared is 6. -/
theorem pitchers_prepared_is_six :
  pitchers_needed total_glasses_served glasses_per_pitcher = 6 := by
  sorry

end NUMINAMATH_CALUDE_pitchers_prepared_is_six_l3220_322001


namespace NUMINAMATH_CALUDE_second_term_is_three_l3220_322076

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- The second term of the sequence is 3 -/
theorem second_term_is_three (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 2) (a 5) →
  a 2 = 3 := by sorry

end NUMINAMATH_CALUDE_second_term_is_three_l3220_322076


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_and_4_l3220_322099

theorem smallest_five_digit_divisible_by_3_and_4 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 3 = 0 ∧ 
  n % 4 = 0 ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 3 = 0 ∧ m % 4 = 0 → m ≥ n) ∧
  n = 10008 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_and_4_l3220_322099


namespace NUMINAMATH_CALUDE_perfect_square_expression_l3220_322013

theorem perfect_square_expression : ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.4792 + 0.02 * 0.02) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l3220_322013


namespace NUMINAMATH_CALUDE_flight_time_theorem_l3220_322078

/-- Represents the flight time between two towns -/
structure FlightTime where
  against_wind : ℝ
  with_wind : ℝ
  no_wind : ℝ

/-- The flight time satisfies the given conditions -/
def satisfies_conditions (ft : FlightTime) : Prop :=
  ft.against_wind = 84 ∧ ft.with_wind = ft.no_wind - 9

/-- The theorem to be proved -/
theorem flight_time_theorem (ft : FlightTime) 
  (h : satisfies_conditions ft) : 
  ft.with_wind = 63 ∨ ft.with_wind = 12 := by
  sorry

end NUMINAMATH_CALUDE_flight_time_theorem_l3220_322078


namespace NUMINAMATH_CALUDE_infinite_nonprime_powers_l3220_322019

theorem infinite_nonprime_powers (k : ℕ) : ∃ n : ℕ, n ≥ k ∧
  (¬ Nat.Prime (2^(2^n) + 1) ∨ ¬ Nat.Prime (2018^(2^n) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_infinite_nonprime_powers_l3220_322019


namespace NUMINAMATH_CALUDE_triangle_side_length_l3220_322016

theorem triangle_side_length (AB : ℝ) (sinA sinC : ℝ) :
  AB = 30 →
  sinA = 3/5 →
  sinC = 1/4 →
  ∃ (BD BC DC : ℝ),
    BD = AB * sinA ∧
    BC = BD / sinC ∧
    DC ^ 2 = BC ^ 2 - BD ^ 2 ∧
    DC = 18 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3220_322016


namespace NUMINAMATH_CALUDE_intercepts_correct_l3220_322068

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Theorem stating that the x-intercept and y-intercept are correct for the given line equation -/
theorem intercepts_correct : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept :=
sorry

end NUMINAMATH_CALUDE_intercepts_correct_l3220_322068


namespace NUMINAMATH_CALUDE_max_value_abc_l3220_322060

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 2048/19683 ∧ ∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ a^2 * b^3 * c^4 = 2048/19683 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l3220_322060


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3220_322018

theorem inequality_system_solution (x : ℝ) : 
  (2 * x - 1 < x + 5) → ((x + 1) / 3 < x - 1) → (2 < x ∧ x < 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3220_322018


namespace NUMINAMATH_CALUDE_mosquito_lethal_feedings_l3220_322023

/-- The number of mosquito feedings required to reach lethal blood loss -/
def lethal_feedings (drops_per_feeding : ℕ) (drops_per_liter : ℕ) (lethal_liters : ℕ) : ℕ :=
  (lethal_liters * drops_per_liter) / drops_per_feeding

theorem mosquito_lethal_feedings :
  lethal_feedings 20 5000 3 = 750 := by
  sorry

#eval lethal_feedings 20 5000 3

end NUMINAMATH_CALUDE_mosquito_lethal_feedings_l3220_322023


namespace NUMINAMATH_CALUDE_power_of_128_l3220_322021

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by sorry

end NUMINAMATH_CALUDE_power_of_128_l3220_322021


namespace NUMINAMATH_CALUDE_train_distance_problem_l3220_322087

theorem train_distance_problem (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : distance_diff = 55)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0) :
  ∃ (time distance1 distance2 : ℝ),
    time > 0 ∧
    distance1 = speed1 * time ∧
    distance2 = speed2 * time ∧
    distance2 = distance1 + distance_diff ∧
    distance1 + distance2 = 495 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3220_322087


namespace NUMINAMATH_CALUDE_solve_for_q_l3220_322007

theorem solve_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l3220_322007


namespace NUMINAMATH_CALUDE_office_age_problem_l3220_322049

theorem office_age_problem (total_persons : Nat) (avg_age_all : Nat) (group1_size : Nat) 
  (group1_avg_age : Nat) (group2_size : Nat) (group2_avg_age : Nat) :
  total_persons = 19 →
  avg_age_all = 15 →
  group1_size = 5 →
  group1_avg_age = 14 →
  group2_size = 9 →
  group2_avg_age = 16 →
  (total_persons * avg_age_all) - (group1_size * group1_avg_age + group2_size * group2_avg_age) = 71 := by
  sorry

#check office_age_problem

end NUMINAMATH_CALUDE_office_age_problem_l3220_322049


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3220_322061

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 8 * x * y) : 1 / x + 1 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3220_322061


namespace NUMINAMATH_CALUDE_modulo_23_equivalence_l3220_322097

theorem modulo_23_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 58294 ≡ n [ZMOD 23] ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_modulo_23_equivalence_l3220_322097


namespace NUMINAMATH_CALUDE_expansion_coefficient_equals_negative_eighty_l3220_322057

/-- The coefficient of the term containing x in the expansion of (2√x - 1/x)^n -/
def coefficient (n : ℕ) : ℤ :=
  (-1)^((n-2)/3) * 2^((2*n+2)/3) * (n.choose ((n-2)/3))

theorem expansion_coefficient_equals_negative_eighty (n : ℕ) :
  coefficient n = -80 → n = 5 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_equals_negative_eighty_l3220_322057


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3220_322040

/-- Given a quadratic equation x^2 + (m - 3)x + m = 0 where m is a real number,
    if one root is greater than 1 and the other root is less than 1,
    then m < 1 -/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 1 ∧ r₂ < 1 ∧ 
    r₁^2 + (m - 3) * r₁ + m = 0 ∧ 
    r₂^2 + (m - 3) * r₂ + m = 0) → 
  m < 1 := by
sorry


end NUMINAMATH_CALUDE_quadratic_root_condition_l3220_322040


namespace NUMINAMATH_CALUDE_volunteer_assignment_problem_l3220_322031

/-- The number of ways to assign n volunteers to k venues with at least one volunteer at each venue -/
def assignment_count (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_assignment_problem :
  assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_problem_l3220_322031


namespace NUMINAMATH_CALUDE_different_color_pairs_count_l3220_322009

/-- Represents the number of socks of each color -/
structure SockDrawer :=
  (white : ℕ)
  (brown : ℕ)
  (blue : ℕ)
  (black : ℕ)

/-- Calculates the number of ways to choose a pair of socks of different colors -/
def differentColorPairs (drawer : SockDrawer) : ℕ :=
  drawer.white * drawer.brown +
  drawer.white * drawer.blue +
  drawer.white * drawer.black +
  drawer.brown * drawer.blue +
  drawer.brown * drawer.black +
  drawer.blue * drawer.black

/-- The specific sock drawer described in the problem -/
def myDrawer : SockDrawer :=
  { white := 4
  , brown := 4
  , blue := 5
  , black := 4 }

theorem different_color_pairs_count :
  differentColorPairs myDrawer = 108 := by
  sorry

end NUMINAMATH_CALUDE_different_color_pairs_count_l3220_322009


namespace NUMINAMATH_CALUDE_unique_solution_l3220_322035

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem unique_solution :
  ∃! n : ℕ, n + S n = 1964 ∧ n = 1945 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3220_322035


namespace NUMINAMATH_CALUDE_closest_to_half_at_seven_dips_l3220_322094

/-- The number of unit cubes --/
def num_cubes : ℕ := 1729

/-- The number of faces per cube --/
def faces_per_cube : ℕ := 6

/-- The total number of faces --/
def total_faces : ℕ := num_cubes * faces_per_cube

/-- The expected number of painted faces per dip --/
def painted_per_dip : ℚ := 978

/-- The recurrence relation for painted faces --/
def painted_faces (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n+1 => painted_faces n * (1566 / 1729) + painted_per_dip

/-- The theorem to prove --/
theorem closest_to_half_at_seven_dips :
  ∀ k : ℕ, k ≠ 7 →
  |painted_faces 7 - (total_faces / 2)| < |painted_faces k - (total_faces / 2)| :=
sorry

end NUMINAMATH_CALUDE_closest_to_half_at_seven_dips_l3220_322094


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3220_322047

-- Define the circle and its properties
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the chord length
def chord_length : ℝ := 10

-- Define the internal segment of the secant
def secant_internal : ℝ := 12

-- Theorem statement
theorem circle_radius_proof (r : ℝ) (h1 : r > 0) :
  ∃ (A B C : ℝ × ℝ),
    A ∈ Circle r ∧ 
    B ∈ Circle r ∧ 
    C ∈ Circle r ∧
    ‖A - B‖ = chord_length ∧
    ‖B - C‖ = secant_internal ∧
    (∃ (D : ℝ × ℝ), D ∈ Circle r ∧ (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0) →
    r = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3220_322047


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3220_322053

-- Define a structure for a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a : ℝ) → (b : ℝ) → (c : ℝ) → (x : ℝ) → (y : ℝ) → Prop
  mk_eq : ∀ x y, eq a b c x y ↔ a * x + b * y + c = 0

-- Define a function to check if a line passes through a point
def passes_through (l : Line) (x₀ y₀ : ℝ) : Prop :=
  l.eq l.a l.b l.c x₀ y₀

-- Define a function to check if a line has equal absolute intercepts
def has_equal_abs_intercepts (l : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ l.a = k ∧ l.b = k ∨ l.a = k ∧ l.b = -k

-- Define the three lines in question
def line1 : Line :=
  { a := 2, b := 3, c := 0, eq := λ a b c x y => a * x + b * y + c = 0,
    mk_eq := λ x y => Iff.rfl }

def line2 : Line :=
  { a := 1, b := 1, c := -1, eq := λ a b c x y => a * x + b * y + c = 0,
    mk_eq := λ x y => Iff.rfl }

def line3 : Line :=
  { a := 1, b := -1, c := -5, eq := λ a b c x y => a * x + b * y + c = 0,
    mk_eq := λ x y => Iff.rfl }

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  (passes_through line1 3 (-2) ∧ has_equal_abs_intercepts line1) ∨
  (passes_through line2 3 (-2) ∧ has_equal_abs_intercepts line2) ∨
  (passes_through line3 3 (-2) ∧ has_equal_abs_intercepts line3) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3220_322053


namespace NUMINAMATH_CALUDE_max_distance_point_to_circle_l3220_322086

/-- The maximum distance between a point and a circle --/
theorem max_distance_point_to_circle :
  let M : ℝ × ℝ := (2, 0)
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2*p.2 = 0}
  ∀ N ∈ C, dist M N ≤ Real.sqrt 5 + 1 ∧ 
  ∃ N' ∈ C, dist M N' = Real.sqrt 5 + 1 := by
  sorry

#check max_distance_point_to_circle

end NUMINAMATH_CALUDE_max_distance_point_to_circle_l3220_322086


namespace NUMINAMATH_CALUDE_exponent_problem_l3220_322004

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = -2) : x^(m+2*n) = 20 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l3220_322004


namespace NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_g_behavior_l3220_322090

/-- The polynomial function g(x) -/
def g (x : ℝ) : ℝ := 3*x^4 - 2*x^3 + x - 9

/-- Theorem stating that g(x) approaches infinity as x approaches positive infinity -/
theorem g_limit_pos_infinity : 
  Filter.Tendsto g Filter.atTop Filter.atTop :=
sorry

/-- Theorem stating that g(x) approaches infinity as x approaches negative infinity -/
theorem g_limit_neg_infinity : 
  Filter.Tendsto g Filter.atBot Filter.atTop :=
sorry

/-- Main theorem combining both limits to show the behavior of g(x) -/
theorem g_behavior : 
  (Filter.Tendsto g Filter.atTop Filter.atTop) ∧ 
  (Filter.Tendsto g Filter.atBot Filter.atTop) :=
sorry

end NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_g_behavior_l3220_322090


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3220_322015

theorem negative_fraction_comparison : -2/3 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3220_322015


namespace NUMINAMATH_CALUDE_magic_sum_order_8_l3220_322083

def magic_sum (n : ℕ) : ℕ :=
  let total_sum := n^2 * (n^2 + 1) / 2
  total_sum / n

theorem magic_sum_order_8 :
  magic_sum 8 = 260 :=
by sorry

end NUMINAMATH_CALUDE_magic_sum_order_8_l3220_322083


namespace NUMINAMATH_CALUDE_min_triangles_for_points_l3220_322010

/-- A square with points and a triangular division -/
structure SquareWithPoints where
  k : ℕ
  points : Finset (ℝ × ℝ)
  triangles : Finset (Finset (ℝ × ℝ))

/-- The property that each triangle contains at most one point -/
def ValidDivision (s : SquareWithPoints) : Prop :=
  ∀ t ∈ s.triangles, (s.points ∩ t).card ≤ 1

/-- The theorem stating the minimum number of triangles needed -/
theorem min_triangles_for_points (s : SquareWithPoints) 
  (h1 : s.k > 2) 
  (h2 : s.points.card = s.k) 
  (h3 : ValidDivision s) : 
  s.triangles.card ≥ s.k + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_for_points_l3220_322010


namespace NUMINAMATH_CALUDE_rowing_speed_in_still_water_l3220_322079

/-- The speed of a man rowing in still water, given downstream conditions -/
theorem rowing_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 8.5)
  (h2 : distance = 45.5)
  (h3 : time = 9.099272058235341)
  : ∃ (still_water_speed : ℝ), still_water_speed = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_in_still_water_l3220_322079


namespace NUMINAMATH_CALUDE_ball_weight_problem_l3220_322055

theorem ball_weight_problem (R W : ℚ) 
  (eq1 : 7 * R + 5 * W = 43)
  (eq2 : 5 * R + 7 * W = 47) :
  4 * R + 8 * W = 49 := by
  sorry

end NUMINAMATH_CALUDE_ball_weight_problem_l3220_322055


namespace NUMINAMATH_CALUDE_peters_remaining_money_l3220_322050

/-- Peter's shopping trip to the market -/
theorem peters_remaining_money 
  (initial_amount : ℕ) 
  (potato_price potato_quantity : ℕ)
  (tomato_price tomato_quantity : ℕ)
  (cucumber_price cucumber_quantity : ℕ)
  (banana_price banana_quantity : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_price = 2 ∧ potato_quantity = 6)
  (h3 : tomato_price = 3 ∧ tomato_quantity = 9)
  (h4 : cucumber_price = 4 ∧ cucumber_quantity = 5)
  (h5 : banana_price = 5 ∧ banana_quantity = 3) :
  initial_amount - 
  (potato_price * potato_quantity + 
   tomato_price * tomato_quantity + 
   cucumber_price * cucumber_quantity + 
   banana_price * banana_quantity) = 426 := by
  sorry

end NUMINAMATH_CALUDE_peters_remaining_money_l3220_322050


namespace NUMINAMATH_CALUDE_tan_graph_product_l3220_322069

theorem tan_graph_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = 3 → x = π / 8) →
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) →
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_tan_graph_product_l3220_322069


namespace NUMINAMATH_CALUDE_class_size_l3220_322089

def is_valid_total_score (n : ℕ) : Prop :=
  n ≥ 4460 ∧ n < 4470 ∧ n % 100 = 64 ∧ n % 8 = 0 ∧ n % 9 = 0

theorem class_size (total_score : ℕ) (h1 : is_valid_total_score total_score) 
  (h2 : (total_score : ℚ) / 72 = 62) : 
  ∃ (num_students : ℕ), (num_students : ℚ) = total_score / 72 := by
sorry

end NUMINAMATH_CALUDE_class_size_l3220_322089


namespace NUMINAMATH_CALUDE_eighth_result_value_l3220_322081

theorem eighth_result_value (total_count : Nat) (total_avg : ℝ)
  (first_7_count : Nat) (first_7_avg : ℝ)
  (next_5_count : Nat) (next_5_avg : ℝ)
  (last_5_count : Nat) (last_5_avg : ℝ)
  (h1 : total_count = 17)
  (h2 : total_avg = 24)
  (h3 : first_7_count = 7)
  (h4 : first_7_avg = 18)
  (h5 : next_5_count = 5)
  (h6 : next_5_avg = 23)
  (h7 : last_5_count = 5)
  (h8 : last_5_avg = 32) :
  (total_count : ℝ) * total_avg - 
  ((first_7_count : ℝ) * first_7_avg + (next_5_count : ℝ) * next_5_avg + (last_5_count : ℝ) * last_5_avg) = 7 := by
  sorry

end NUMINAMATH_CALUDE_eighth_result_value_l3220_322081
