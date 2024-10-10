import Mathlib

namespace parabola_c_value_l2578_257887

-- Define the parabola
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex condition
def vertex_condition (a b c : ℝ) : Prop :=
  parabola a b c 3 = -5

-- Define the point condition
def point_condition (a b c : ℝ) : Prop :=
  parabola a b c 1 = -3

theorem parabola_c_value :
  ∀ a b c : ℝ,
  vertex_condition a b c →
  point_condition a b c →
  c = -0.5 := by sorry

end parabola_c_value_l2578_257887


namespace max_consecutive_sum_under_1000_l2578_257835

theorem max_consecutive_sum_under_1000 : 
  (∀ k : ℕ, k ≤ 44 → k * (k + 1) ≤ 2000) ∧ 
  45 * 46 > 2000 := by
  sorry

end max_consecutive_sum_under_1000_l2578_257835


namespace function_characterization_l2578_257822

/-- A function is strictly increasing if for all x < y, f(x) < f(y) -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- g is the composition inverse of f if f(g(x)) = x and g(f(x)) = x for all real x -/
def CompositionInverse (f g : ℝ → ℝ) : Prop :=
  (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x)

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : StrictlyIncreasing f)
  (h2 : ∃ g : ℝ → ℝ, CompositionInverse f g ∧ ∀ x, f x + g x = 2 * x) :
  ∃ c : ℝ, ∀ x, f x = x + c :=
sorry

end function_characterization_l2578_257822


namespace beaver_count_correct_l2578_257821

/-- The number of beavers in the first scenario -/
def num_beavers : ℕ := 20

/-- The time taken by the first group of beavers to build the dam -/
def time_first : ℕ := 18

/-- The number of beavers in the second scenario -/
def num_beavers_second : ℕ := 12

/-- The time taken by the second group of beavers to build the dam -/
def time_second : ℕ := 30

/-- The theorem stating that the calculated number of beavers is correct -/
theorem beaver_count_correct :
  num_beavers * time_first = num_beavers_second * time_second :=
by sorry

end beaver_count_correct_l2578_257821


namespace grandfather_age_relationship_l2578_257884

/-- Represents the ages and relationships in the family problem -/
structure FamilyAges where
  fatherCurrentAge : ℕ
  sonCurrentAge : ℕ
  grandfatherAgeFiveYearsAgo : ℕ
  fatherAgeSameAsSonAtBirth : fatherCurrentAge = sonCurrentAge + sonCurrentAge
  fatherCurrentAge58 : fatherCurrentAge = 58
  sonAgeFiveYearsAgoHalfGrandfather : sonCurrentAge - 5 = (grandfatherAgeFiveYearsAgo - 5) / 2

/-- Theorem stating the relationship between the grandfather's age 5 years ago and the son's current age -/
theorem grandfather_age_relationship (f : FamilyAges) : 
  f.grandfatherAgeFiveYearsAgo = 2 * f.sonCurrentAge - 5 := by
  sorry

#check grandfather_age_relationship

end grandfather_age_relationship_l2578_257884


namespace cosine_equality_condition_l2578_257818

theorem cosine_equality_condition (x y : ℝ) : 
  (x = y → Real.cos x = Real.cos y) ∧ 
  ∃ a b : ℝ, Real.cos a = Real.cos b ∧ a ≠ b :=
by sorry

end cosine_equality_condition_l2578_257818


namespace trapezoid_area_proof_l2578_257891

/-- The area of a trapezoid bounded by the lines y = x + 1, y = 15, y = 8, and the y-axis -/
def trapezoid_area : ℝ := 73.5

/-- The line y = x + 1 -/
def line1 (x : ℝ) : ℝ := x + 1

/-- The line y = 15 -/
def line2 : ℝ := 15

/-- The line y = 8 -/
def line3 : ℝ := 8

theorem trapezoid_area_proof :
  let x1 := (line2 - 1 : ℝ)  -- x-coordinate where y = x + 1 intersects y = 15
  let x2 := (line3 - 1 : ℝ)  -- x-coordinate where y = x + 1 intersects y = 8
  let base1 := x1
  let base2 := x2
  let height := line2 - line3
  (base1 + base2) * height / 2 = trapezoid_area := by sorry

end trapezoid_area_proof_l2578_257891


namespace student_weight_l2578_257864

theorem student_weight (student_weight sister_weight : ℝ) :
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 104 →
  student_weight = 71 := by
sorry

end student_weight_l2578_257864


namespace sector_area_l2578_257860

/-- Given a sector with central angle α and arc length l, 
    the area S of the sector is (l * l) / (2 * α) -/
theorem sector_area (α : ℝ) (l : ℝ) (S : ℝ) 
  (h1 : α = 2) 
  (h2 : l = 3 * Real.pi) 
  (h3 : S = (l * l) / (2 * α)) : 
  S = (9 * Real.pi^2) / 4 := by
  sorry

#check sector_area

end sector_area_l2578_257860


namespace bench_seating_l2578_257844

theorem bench_seating (N : ℕ) : (∃ x : ℕ, 7 * N = x ∧ 11 * N = x) ↔ N ≥ 77 :=
sorry

end bench_seating_l2578_257844


namespace smallest_perfect_square_divisible_by_5_and_7_l2578_257827

theorem smallest_perfect_square_divisible_by_5_and_7 : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (m : ℕ), n = m ^ 2) ∧ 
  5 ∣ n ∧ 
  7 ∣ n ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (l : ℕ), k = l ^ 2) → 5 ∣ k → 7 ∣ k → k ≥ n) ∧
  n = 1225 :=
by sorry

end smallest_perfect_square_divisible_by_5_and_7_l2578_257827


namespace quadratic_equation_solution_l2578_257892

/-- Given a quadratic function f(x) = x^2 + bx - 5 with symmetric axis x = 2,
    prove that the solutions to f(x) = 2x - 13 are x₁ = 2 and x₂ = 4. -/
theorem quadratic_equation_solution (b : ℝ) :
  (∀ x, x^2 + b*x - 5 = x^2 + b*x - 5) →  -- f(x) is a well-defined function
  (-b/2 = 2) →                            -- symmetric axis is x = 2
  (∃ x₁ x₂, x₁ = 2 ∧ x₂ = 4 ∧
    (∀ x, x^2 + b*x - 5 = 2*x - 13 ↔ x = x₁ ∨ x = x₂)) :=
by sorry

end quadratic_equation_solution_l2578_257892


namespace sqrt_decimal_movement_l2578_257836

theorem sqrt_decimal_movement (a b : ℝ) (n : ℤ) (h : Real.sqrt a = b) :
  Real.sqrt (a * (10 : ℝ)^(2*n)) = b * (10 : ℝ)^n := by sorry

end sqrt_decimal_movement_l2578_257836


namespace max_gcd_of_sequence_l2578_257802

theorem max_gcd_of_sequence (n : ℕ+) : Nat.gcd (101 + n^3) (101 + (n + 1)^3) = 1 := by
  sorry

end max_gcd_of_sequence_l2578_257802


namespace quadratic_always_positive_l2578_257888

theorem quadratic_always_positive (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + (4 + m) > 0) ↔ 
  (-Real.sqrt 55 / 2 < m ∧ m < Real.sqrt 55 / 2) := by
  sorry

end quadratic_always_positive_l2578_257888


namespace standard_form_conversion_theta_range_phi_range_l2578_257849

/-- Converts spherical coordinates to standard form -/
def to_standard_spherical (ρ : ℝ) (θ : ℝ) (φ : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The standard form of (5, 3π/4, 9π/4) is (5, 7π/4, π/4) -/
theorem standard_form_conversion :
  let (ρ, θ, φ) := to_standard_spherical 5 (3 * Real.pi / 4) (9 * Real.pi / 4)
  ρ = 5 ∧ θ = 7 * Real.pi / 4 ∧ φ = Real.pi / 4 :=
by
  sorry

/-- The range of θ in standard spherical coordinates -/
theorem theta_range (θ : ℝ) : 0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

/-- The range of φ in standard spherical coordinates -/
theorem phi_range (φ : ℝ) : 0 ≤ φ ∧ φ ≤ Real.pi :=
by
  sorry

end standard_form_conversion_theta_range_phi_range_l2578_257849


namespace fraction_difference_l2578_257838

theorem fraction_difference (n d : ℤ) : 
  d = 5 → n > d → n + 6 = 3 * d → n - d = 4 := by sorry

end fraction_difference_l2578_257838


namespace cassidy_profit_l2578_257893

/-- Cassidy's bread baking and selling scenario --/
def bread_scenario (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (midday_price : ℚ) (evening_price : ℚ) : Prop :=
  let morning_sold := total_loaves / 3
  let midday_remaining := total_loaves - morning_sold
  let midday_sold := midday_remaining / 2
  let evening_sold := midday_remaining - midday_sold
  let total_revenue := morning_sold * morning_price + midday_sold * midday_price + evening_sold * evening_price
  let total_cost := total_loaves * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 70

/-- Theorem stating Cassidy's profit is $70 --/
theorem cassidy_profit : 
  bread_scenario 60 1 3 2 (3/2) :=
sorry

end cassidy_profit_l2578_257893


namespace constant_fifth_term_implies_n_six_l2578_257819

/-- 
Given a positive integer n, and considering the binomial expansion of (x^2 + 1/x)^n,
if the fifth term is a constant (i.e., the exponent of x is 0), then n must equal 6.
-/
theorem constant_fifth_term_implies_n_six (n : ℕ+) : 
  (∃ k : ℕ, k > 0 ∧ 2*n - 3*(k+1) = 0) → n = 6 := by
  sorry

end constant_fifth_term_implies_n_six_l2578_257819


namespace missing_digit_divisible_by_9_l2578_257861

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def five_digit_number (x : ℕ) : ℕ := 24600 + 10 * x + 8

theorem missing_digit_divisible_by_9 :
  ∀ x : ℕ, x < 10 →
    (is_divisible_by_9 (five_digit_number x) ↔ x = 7) :=
by sorry

end missing_digit_divisible_by_9_l2578_257861


namespace student_count_l2578_257820

theorem student_count (average_student_age : ℝ) (teacher_age : ℝ) (new_average_age : ℝ) :
  average_student_age = 15 →
  teacher_age = 26 →
  new_average_age = 16 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * average_student_age + teacher_age = (n + 1 : ℝ) * new_average_age ∧
    n = 10 := by
  sorry

end student_count_l2578_257820


namespace childrens_ticket_cost_l2578_257886

/-- Calculates the cost of a children's ticket given the total cost and other ticket prices --/
theorem childrens_ticket_cost 
  (adult_price : ℕ) 
  (senior_price : ℕ) 
  (total_cost : ℕ) 
  (num_adults : ℕ) 
  (num_seniors : ℕ) 
  (num_children : ℕ) :
  adult_price = 11 →
  senior_price = 9 →
  num_adults = 2 →
  num_seniors = 2 →
  num_children = 3 →
  total_cost = 64 →
  (total_cost - (num_adults * adult_price + num_seniors * senior_price)) / num_children = 8 :=
by
  sorry

#check childrens_ticket_cost

end childrens_ticket_cost_l2578_257886


namespace rectangle_opposite_vertices_distance_sum_equal_l2578_257894

/-- Theorem: The sums of the squares of the distances from any point in space to opposite vertices of a rectangle are equal to each other. -/
theorem rectangle_opposite_vertices_distance_sum_equal 
  (a b x y z : ℝ) : 
  (x^2 + y^2 + z^2) + ((x - a)^2 + (y - b)^2 + z^2) = 
  ((x - a)^2 + y^2 + z^2) + (x^2 + (y - b)^2 + z^2) := by
  sorry

end rectangle_opposite_vertices_distance_sum_equal_l2578_257894


namespace cos_equality_with_large_angle_l2578_257807

theorem cos_equality_with_large_angle (n : ℕ) :
  0 ≤ n ∧ n ≤ 200 →
  n = 166 →
  Real.cos (n * π / 180) = Real.cos (1274 * π / 180) := by
sorry

end cos_equality_with_large_angle_l2578_257807


namespace x_value_l2578_257842

theorem x_value (m : ℕ) (x : ℝ) 
  (h1 : m = 34) 
  (h2 : ((x ^ (m + 1)) / (5 ^ (m + 1))) * ((x ^ 18) / (4 ^ 18)) = 1 / (2 * (10 ^ 35))) :
  x = 1 := by
sorry

end x_value_l2578_257842


namespace kate_wands_proof_l2578_257897

/-- The number of wands Kate bought -/
def total_wands : ℕ := 3

/-- The cost of each wand -/
def wand_cost : ℕ := 60

/-- The selling price of each wand -/
def selling_price : ℕ := wand_cost + 5

/-- The total amount collected from sales -/
def total_collected : ℕ := 130

/-- Kate keeps one wand for herself -/
def kept_wands : ℕ := 1

theorem kate_wands_proof : 
  total_wands = (total_collected / selling_price) + kept_wands :=
by sorry

end kate_wands_proof_l2578_257897


namespace bus_seating_solution_l2578_257839

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  back_seat_capacity : Nat
  total_capacity : Nat

/-- Calculates the number of people each regular seat can hold -/
def seats_capacity (bus : BusSeating) : Nat :=
  let regular_seats := bus.left_seats + bus.right_seats
  let regular_capacity := bus.total_capacity - bus.back_seat_capacity
  regular_capacity / regular_seats

/-- Theorem stating the solution to the bus seating problem -/
theorem bus_seating_solution :
  let bus := BusSeating.mk 15 12 11 92
  seats_capacity bus = 3 := by sorry

end bus_seating_solution_l2578_257839


namespace inequality_system_solution_set_l2578_257826

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by sorry

end inequality_system_solution_set_l2578_257826


namespace fraction_product_simplification_l2578_257817

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 = 3 / 7 := by
  sorry

end fraction_product_simplification_l2578_257817


namespace exam_questions_count_l2578_257809

/-- Exam scoring system and student performance -/
structure ExamScoring where
  correct_score : Int
  incorrect_penalty : Int
  total_score : Int
  correct_answers : Int

/-- Calculate the total number of questions in the exam -/
def total_questions (exam : ExamScoring) : Int :=
  exam.correct_answers + (exam.total_score - exam.correct_score * exam.correct_answers) / (-exam.incorrect_penalty)

/-- Theorem: The total number of questions in the exam is 150 -/
theorem exam_questions_count (exam : ExamScoring) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.incorrect_penalty = 2)
  (h3 : exam.total_score = 420)
  (h4 : exam.correct_answers = 120) : 
  total_questions exam = 150 := by
  sorry


end exam_questions_count_l2578_257809


namespace average_population_is_1000_l2578_257834

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

theorem average_population_is_1000 : 
  (village_populations.sum / village_populations.length : ℚ) = 1000 := by
  sorry

end average_population_is_1000_l2578_257834


namespace oak_grove_library_books_l2578_257847

/-- The number of books in Oak Grove's school libraries -/
def school_books : ℕ := 5106

/-- The total number of books in all Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's public library -/
def public_books : ℕ := total_books - school_books

theorem oak_grove_library_books : public_books = 1986 := by
  sorry

end oak_grove_library_books_l2578_257847


namespace tetrahedron_inscribed_circle_centers_intersection_l2578_257851

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

def inscribedCircleCenter (p q r : Point) : Point := sorry

def intersect (a b c d : Point) : Prop := sorry

theorem tetrahedron_inscribed_circle_centers_intersection 
  (ABCD : Tetrahedron) 
  (E : Point) 
  (F : Point) 
  (h1 : E = inscribedCircleCenter ABCD.B ABCD.C ABCD.D) 
  (h2 : F = inscribedCircleCenter ABCD.A ABCD.C ABCD.D) 
  (h3 : intersect ABCD.A E ABCD.B F) :
  ∃ (G H : Point), 
    G = inscribedCircleCenter ABCD.A ABCD.B ABCD.D ∧ 
    H = inscribedCircleCenter ABCD.A ABCD.B ABCD.C ∧ 
    intersect ABCD.C G ABCD.D H :=
sorry

end tetrahedron_inscribed_circle_centers_intersection_l2578_257851


namespace initial_classes_l2578_257862

theorem initial_classes (initial_classes : ℕ) : 
  (20 * initial_classes : ℕ) + (20 * 5 : ℕ) = 400 → initial_classes = 15 :=
by sorry

end initial_classes_l2578_257862


namespace max_planks_from_trunk_l2578_257846

/-- Represents a cylindrical tree trunk -/
structure Trunk :=
  (diameter : ℝ)

/-- Represents a plank with thickness and width -/
structure Plank :=
  (thickness : ℝ)
  (width : ℝ)

/-- Calculates the maximum number of planks that can be cut from a trunk -/
def max_planks (t : Trunk) (p : Plank) : ℕ :=
  sorry

/-- Theorem stating the maximum number of planks that can be cut -/
theorem max_planks_from_trunk (t : Trunk) (p : Plank) :
  t.diameter = 46 → p.thickness = 4 → p.width = 12 → max_planks t p = 29 := by
  sorry

end max_planks_from_trunk_l2578_257846


namespace quotient_of_arctangents_eq_one_l2578_257867

theorem quotient_of_arctangents_eq_one :
  (π - Real.arctan (8/15)) / (2 * Real.arctan 4) = 1 := by sorry

end quotient_of_arctangents_eq_one_l2578_257867


namespace distance_between_circle_centers_l2578_257808

-- Define the isosceles triangle
structure IsoscelesTriangle where
  vertex_angle : Real
  side_length : Real

-- Define the circles
structure CircumscribedCircle where
  radius : Real

structure InscribedCircle where
  radius : Real

structure SecondCircle where
  radius : Real
  distance_to_vertex : Real

-- Main theorem
theorem distance_between_circle_centers
  (triangle : IsoscelesTriangle)
  (circum_circle : CircumscribedCircle)
  (in_circle : InscribedCircle)
  (second_circle : SecondCircle)
  (h1 : triangle.vertex_angle = 45)
  (h2 : second_circle.distance_to_vertex = 4)
  (h3 : second_circle.radius = circum_circle.radius - 4)
  (h4 : second_circle.radius > 0)
  (h5 : in_circle.radius > 0) :
  ∃ (distance : Real), distance = 4 ∧ 
    distance = circum_circle.radius - in_circle.radius + 4 * Real.sin (45 * π / 180) :=
by sorry


end distance_between_circle_centers_l2578_257808


namespace parabola_theorem_l2578_257880

/-- Parabola with parameter p and a tangent line -/
structure Parabola where
  p : ℝ
  tangent_x_intercept : ℝ
  tangent_y_intercept : ℝ

/-- Properties of the parabola -/
def parabola_properties (para : Parabola) : Prop :=
  -- Tangent line equation matches the given form
  para.tangent_x_intercept = -75 ∧ para.tangent_y_intercept = 15 ∧
  -- Parameter p is 6
  para.p = 6 ∧
  -- Focus coordinates are (3, 0)
  (3 : ℝ) = para.p / 2 ∧
  -- Directrix equation is x = -3
  (-3 : ℝ) = -para.p / 2

/-- Theorem stating the properties of the parabola -/
theorem parabola_theorem (para : Parabola) :
  parabola_properties para :=
sorry

end parabola_theorem_l2578_257880


namespace circumcircle_of_triangle_ABC_l2578_257895

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (4, 2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Theorem statement
theorem circumcircle_of_triangle_ABC :
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  circle_equation C.1 C.2 ∧
  ∀ (x y : ℝ), circle_equation x y →
    (x - A.1)^2 + (y - A.2)^2 =
    (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 =
    (x - C.1)^2 + (y - C.2)^2 :=
sorry


end circumcircle_of_triangle_ABC_l2578_257895


namespace computer_price_increase_l2578_257890

theorem computer_price_increase (d : ℝ) : 
  2 * d = 560 → (d * 1.3 : ℝ) = 364 := by
  sorry

end computer_price_increase_l2578_257890


namespace probability_black_second_draw_l2578_257814

theorem probability_black_second_draw 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (h1 : total_balls = 5) 
  (h2 : white_balls = 3) 
  (h3 : black_balls = 2) 
  (h4 : total_balls = white_balls + black_balls) 
  (h5 : white_balls > 0) : 
  (black_balls : ℚ) / (total_balls - 1 : ℚ) = 1/2 := by
sorry

end probability_black_second_draw_l2578_257814


namespace unique_solution_l2578_257898

/-- A pair of natural numbers (r, x) where r is the base and x is a number in that base -/
structure BaseNumber :=
  (r : ℕ)
  (x : ℕ)

/-- Check if a BaseNumber satisfies the given conditions -/
def satisfiesConditions (bn : BaseNumber) : Prop :=
  -- r is at most 70
  bn.r ≤ 70 ∧
  -- x is represented by repeating a pair of digits
  ∃ (n : ℕ) (a b : ℕ), 
    a < bn.r ∧ b < bn.r ∧
    bn.x = (a * bn.r + b) * (bn.r^(2*n) - 1) / (bn.r^2 - 1) ∧
  -- x^2 in base r consists of 4n ones
  ∃ (n : ℕ), bn.x^2 = (bn.r^(4*n) - 1) / (bn.r - 1)

/-- The theorem stating that (7, 26₇) is the only solution -/
theorem unique_solution : 
  ∀ (bn : BaseNumber), satisfiesConditions bn ↔ bn.r = 7 ∧ bn.x = 26 :=
sorry

end unique_solution_l2578_257898


namespace range_of_m_l2578_257803

-- Define the function f
def f (x : ℝ) := x^3 + x

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m ≤ 1 :=
sorry

end range_of_m_l2578_257803


namespace factorial_sum_equality_l2578_257878

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + Nat.factorial 6 = Nat.factorial 8 := by
  sorry

end factorial_sum_equality_l2578_257878


namespace sqrt_equation_solution_l2578_257879

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 8) / Real.sqrt (8 * x + 8) = 2 / Real.sqrt 5) → x = -1/3 := by
  sorry

end sqrt_equation_solution_l2578_257879


namespace solution_and_uniqueness_l2578_257881

def equation (x : ℤ) : Prop :=
  (x + 1)^3 + (x + 2)^3 + (x + 3)^3 = (x + 4)^3

theorem solution_and_uniqueness :
  equation 2 ∧ ∀ x : ℤ, x ≠ 2 → ¬(equation x) := by
  sorry

end solution_and_uniqueness_l2578_257881


namespace quadratic_expression_value_l2578_257877

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  3*x^2 - 6*x + 9 = 15 := by
sorry

end quadratic_expression_value_l2578_257877


namespace hcd_8580_330_minus_12_l2578_257829

theorem hcd_8580_330_minus_12 : Nat.gcd 8580 330 - 12 = 318 := by
  sorry

end hcd_8580_330_minus_12_l2578_257829


namespace polynomial_coefficient_sum_l2578_257848

theorem polynomial_coefficient_sum :
  ∀ A B C D E : ℝ,
  (∀ x : ℝ, (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E) →
  A + B + C + D + E = 16 := by
sorry

end polynomial_coefficient_sum_l2578_257848


namespace combined_average_age_l2578_257833

theorem combined_average_age (x_count y_count z_count : ℕ) 
  (x_avg y_avg z_avg : ℝ) : 
  x_count = 5 → 
  y_count = 3 → 
  z_count = 2 → 
  x_avg = 35 → 
  y_avg = 30 → 
  z_avg = 45 → 
  (x_count * x_avg + y_count * y_avg + z_count * z_avg) / (x_count + y_count + z_count) = 35.5 := by
  sorry

end combined_average_age_l2578_257833


namespace ratio_equality_l2578_257801

theorem ratio_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4 ∧ a/2 ≠ 0) : 
  (a - 2*c) / (a - 2*b) = 3/2 := by
  sorry

end ratio_equality_l2578_257801


namespace no_integer_solution_for_175_l2578_257813

theorem no_integer_solution_for_175 :
  ∀ x y : ℤ, x^2 + y^2 ≠ 175 := by
  sorry

end no_integer_solution_for_175_l2578_257813


namespace decimal_division_division_result_l2578_257883

theorem decimal_division (x y : ℚ) : x / y = (x * 1000) / (y * 1000) := by sorry

theorem division_result : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by sorry

end decimal_division_division_result_l2578_257883


namespace arithmetic_mean_special_set_l2578_257852

theorem arithmetic_mean_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate (n - 1) 1 ++ [1 + 1 / n]
  (set.sum / n : ℚ) = 1 + 1 / n^2 := by
  sorry

end arithmetic_mean_special_set_l2578_257852


namespace quadratic_roots_imply_m_l2578_257899

theorem quadratic_roots_imply_m (m : ℝ) : 
  (∀ x : ℂ, 8 * x^2 + 4 * x + m = 0 ↔ x = (-2 + Complex.I * Real.sqrt 88) / 8 ∨ x = (-2 - Complex.I * Real.sqrt 88) / 8) → 
  m = 13 / 4 := by
sorry

end quadratic_roots_imply_m_l2578_257899


namespace basil_plants_theorem_l2578_257806

/-- Calculates the number of basil plants sold given the costs, selling price, and net profit -/
def basil_plants_sold (seed_cost potting_soil_cost selling_price net_profit : ℚ) : ℚ :=
  (net_profit + seed_cost + potting_soil_cost) / selling_price

/-- Theorem stating that the number of basil plants sold is 20 -/
theorem basil_plants_theorem :
  basil_plants_sold 2 8 5 90 = 20 := by
  sorry

end basil_plants_theorem_l2578_257806


namespace subway_scenarios_l2578_257815

/-- Represents the fare structure for the subway -/
def fare (x : ℕ) : ℕ :=
  if x ≤ 4 then 2
  else if x ≤ 9 then 4
  else if x ≤ 15 then 6
  else 0

/-- The maximum number of stations -/
def max_stations : ℕ := 15

/-- Calculates the number of scenarios where two passengers pay a total fare -/
def scenarios_for_total_fare (total_fare : ℕ) : ℕ := sorry

/-- Calculates the number of scenarios where passenger A gets off before passenger B -/
def scenarios_a_before_b (total_fare : ℕ) : ℕ := sorry

theorem subway_scenarios :
  (scenarios_for_total_fare 6 = 40) ∧
  (scenarios_a_before_b 8 = 34) := by sorry

end subway_scenarios_l2578_257815


namespace g_of_two_value_l2578_257857

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3 * g(1 - x) = 4 * x^3 - 5 * x for all real x, 
    prove that g(2) = -19/6 -/
theorem g_of_two_value (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x) : 
  g 2 = -19/6 := by
  sorry

end g_of_two_value_l2578_257857


namespace hotel_rooms_l2578_257875

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost total_revenue : ℚ) 
  (h1 : total_rooms = 260)
  (h2 : single_cost = 35)
  (h3 : double_cost = 60)
  (h4 : total_revenue = 14000) :
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    double_rooms = 196 :=
by sorry

end hotel_rooms_l2578_257875


namespace no_real_solutions_l2578_257882

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 10 ≠ 0 :=
by
  sorry

end no_real_solutions_l2578_257882


namespace arrangement_counts_l2578_257855

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 4

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

theorem arrangement_counts :
  (∃ (n₁ n₂ n₃ n₄ : ℕ),
    /- (1) Person A and Person B at the ends -/
    n₁ = 240 ∧
    /- (2) All male students grouped together -/
    n₂ = 720 ∧
    /- (3) No male students next to each other -/
    n₃ = 1440 ∧
    /- (4) Exactly one person between Person A and Person B -/
    n₄ = 1200 ∧
    /- The numbers represent valid arrangement counts -/
    n₁ > 0 ∧ n₂ > 0 ∧ n₃ > 0 ∧ n₄ > 0) :=
by
  sorry

end arrangement_counts_l2578_257855


namespace large_rectangle_perimeter_l2578_257858

/-- Given five identical rectangles each with an area of 8 cm², 
    when arranged into a large rectangle, 
    the perimeter of the large rectangle is 32 cm. -/
theorem large_rectangle_perimeter (small_rectangle_area : ℝ) 
  (h1 : small_rectangle_area = 8) 
  (h2 : ∃ (w h : ℝ), w * h = small_rectangle_area ∧ h = 2 * w) : 
  ∃ (W H : ℝ), W * H = 5 * small_rectangle_area ∧ 2 * (W + H) = 32 :=
by sorry

end large_rectangle_perimeter_l2578_257858


namespace triangle_side_angle_relation_l2578_257843

/-- Given a triangle ABC with side lengths a, b, and c opposite to angles A, B, and C respectively,
    the sum of the squares of the side lengths equals twice the sum of the products of pairs of
    side lengths and the cosine of their opposite angles. -/
theorem triangle_side_angle_relation (a b c : ℝ) (A B C : ℝ) :
  (a ≥ 0) → (b ≥ 0) → (c ≥ 0) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  a^2 + b^2 + c^2 = 2 * (b * c * Real.cos A + a * c * Real.cos B + a * b * Real.cos C) :=
by sorry

end triangle_side_angle_relation_l2578_257843


namespace inverse_proportion_problem_l2578_257896

/-- Given that x² is inversely proportional to y⁴, prove that x² = 2.25 when y = 4,
    given that x = 6 when y = 2. -/
theorem inverse_proportion_problem (k : ℝ) (x y : ℝ → ℝ) :
  (∀ t, t > 0 → x t ^ 2 * y t ^ 4 = k) →  -- x² is inversely proportional to y⁴
  x 2 = 6 →                               -- x = 6 when y = 2
  y 2 = 2 →                               -- y = 2 at this point
  y 4 = 4 →                               -- y = 4 at the point we're calculating
  x 4 ^ 2 = 2.25 :=                       -- x² = 2.25 when y = 4
by sorry

end inverse_proportion_problem_l2578_257896


namespace platform_length_theorem_l2578_257823

def train_length : ℝ := 250

theorem platform_length_theorem (X Y : ℝ) (platform_time signal_time : ℝ) 
  (h1 : platform_time = 40)
  (h2 : signal_time = 20)
  (h3 : Y * signal_time = train_length) :
  Y = 12.5 ∧ ∃ L, L = X * platform_time - train_length := by
  sorry

#check platform_length_theorem

end platform_length_theorem_l2578_257823


namespace determinant_of_specific_matrix_l2578_257804

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 2, 3]
  Matrix.det A = 2 := by
sorry

end determinant_of_specific_matrix_l2578_257804


namespace hua_optimal_selection_uses_golden_ratio_l2578_257854

/-- The mathematical concept used in Hua Luogeng's optimal selection method -/
inductive OptimalSelectionConcept
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- Hua Luogeng's optimal selection method -/
def huaOptimalSelectionMethod : OptimalSelectionConcept := OptimalSelectionConcept.GoldenRatio

/-- Theorem: The mathematical concept used in Hua Luogeng's optimal selection method is the Golden ratio -/
theorem hua_optimal_selection_uses_golden_ratio :
  huaOptimalSelectionMethod = OptimalSelectionConcept.GoldenRatio := by
  sorry

end hua_optimal_selection_uses_golden_ratio_l2578_257854


namespace fraction_equality_l2578_257828

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 1 / x + 1 / y = 2) : 
  (x * y + 3 * x + 3 * y) / (x * y) = 7 := by
  sorry

end fraction_equality_l2578_257828


namespace sqrt_one_minus_sqrt_three_squared_l2578_257870

theorem sqrt_one_minus_sqrt_three_squared : 
  Real.sqrt ((1 - Real.sqrt 3) ^ 2) = Real.sqrt 3 - 1 := by sorry

end sqrt_one_minus_sqrt_three_squared_l2578_257870


namespace shorter_can_radius_l2578_257868

/-- Represents a cylindrical can with radius and height -/
structure Can where
  radius : ℝ
  height : ℝ

/-- Given two cans with equal volume, one with 4 times the height of the other,
    and the taller can having a radius of 5, prove the radius of the shorter can is 10 -/
theorem shorter_can_radius (can1 can2 : Can) 
  (h_volume : π * can1.radius^2 * can1.height = π * can2.radius^2 * can2.height)
  (h_height : can2.height = 4 * can1.height)
  (h_taller_radius : can2.radius = 5) :
  can1.radius = 10 := by
  sorry

end shorter_can_radius_l2578_257868


namespace mod_power_minus_three_l2578_257811

theorem mod_power_minus_three (m : ℕ) : 
  0 ≤ m ∧ m < 37 ∧ (4 * m) % 37 = 1 → 
  (((3 ^ m) ^ 4 - 3) : ℤ) % 37 = 25 := by
  sorry

end mod_power_minus_three_l2578_257811


namespace fraction_addition_l2578_257831

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 11 = (37 : ℚ) / 55 := by
  sorry

end fraction_addition_l2578_257831


namespace drainage_pipes_count_l2578_257810

/-- The number of initial drainage pipes -/
def n : ℕ := 5

/-- The time (in days) it takes n pipes to drain the pool -/
def initial_time : ℕ := 12

/-- The time (in days) it takes (n + 10) pipes to drain the pool -/
def faster_time : ℕ := 4

/-- The number of additional pipes -/
def additional_pipes : ℕ := 10

theorem drainage_pipes_count :
  (n : ℚ) * faster_time = (n + additional_pipes) * initial_time :=
sorry

end drainage_pipes_count_l2578_257810


namespace sequence_matches_given_terms_l2578_257865

/-- The general term of the sequence -/
def a (n : ℕ) : ℚ := n + n^2 / (n^2 + 1)

/-- The first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  (a 1 = 3/2) ∧ 
  (a 2 = 14/5) ∧ 
  (a 3 = 39/10) ∧ 
  (a 4 = 84/17) := by
  sorry

end sequence_matches_given_terms_l2578_257865


namespace paul_juice_bottles_l2578_257853

/-- 
Given that Donald drinks 3 more than twice the number of juice bottles Paul drinks in one day,
and Donald drinks 9 bottles of juice per day, prove that Paul drinks 3 bottles of juice per day.
-/
theorem paul_juice_bottles (paul_bottles : ℕ) (donald_bottles : ℕ) : 
  donald_bottles = 2 * paul_bottles + 3 →
  donald_bottles = 9 →
  paul_bottles = 3 := by
  sorry

end paul_juice_bottles_l2578_257853


namespace speed_in_fifth_hour_l2578_257816

def speed_hour1 : ℝ := 90
def speed_hour2 : ℝ := 60
def speed_hour3 : ℝ := 120
def speed_hour4 : ℝ := 72
def avg_speed : ℝ := 80
def total_time : ℝ := 5

def total_distance : ℝ := avg_speed * total_time

def distance_first_four_hours : ℝ := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4

def speed_hour5 : ℝ := total_distance - distance_first_four_hours

theorem speed_in_fifth_hour :
  speed_hour5 = 58 := by sorry

end speed_in_fifth_hour_l2578_257816


namespace crayon_distribution_l2578_257830

theorem crayon_distribution (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  total_crayons = num_people * crayons_per_person →
  crayons_per_person = 8 := by
  sorry

end crayon_distribution_l2578_257830


namespace prob_same_heads_m_plus_n_l2578_257840

def fair_coin_prob : ℚ := 1/2
def biased_coin_prob : ℚ := 3/5

def same_heads_prob : ℚ :=
  (1 - fair_coin_prob) * (1 - biased_coin_prob) +
  fair_coin_prob * biased_coin_prob +
  fair_coin_prob * (1 - biased_coin_prob) * biased_coin_prob * (1 - fair_coin_prob)

theorem prob_same_heads :
  same_heads_prob = 19/50 := by sorry

#eval Nat.gcd 19 50  -- To verify that 19 and 50 are relatively prime

def m : ℕ := 19
def n : ℕ := 50

theorem m_plus_n : m + n = 69 := by sorry

end prob_same_heads_m_plus_n_l2578_257840


namespace sum_reciprocal_plus_one_bounds_l2578_257837

theorem sum_reciprocal_plus_one_bounds (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < (1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z)) ∧ 
  (1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z)) < 2 := by
  sorry

end sum_reciprocal_plus_one_bounds_l2578_257837


namespace A_run_time_l2578_257889

/-- The time it takes for A to run 160 meters -/
def time_A : ℝ := 28

/-- The time it takes for B to run 160 meters -/
def time_B : ℝ := 32

/-- The distance A runs -/
def distance_A : ℝ := 160

/-- The distance B runs when A finishes -/
def distance_B : ℝ := 140

theorem A_run_time :
  (distance_A / time_A = distance_B / time_B) ∧
  (distance_A - distance_B = 20) →
  time_A = 28 := by sorry

end A_run_time_l2578_257889


namespace remainder_problem_l2578_257876

theorem remainder_problem (N : ℤ) (h : ∃ k : ℤ, N = 39 * k + 18) : 
  ∃ m : ℤ, N = 13 * m + 5 := by
sorry

end remainder_problem_l2578_257876


namespace wind_speed_calculation_l2578_257856

/-- The speed of the wind that satisfies the given conditions -/
def wind_speed : ℝ := 20

/-- The speed of the plane in still air -/
def plane_speed : ℝ := 180

/-- The distance flown with the wind -/
def distance_with_wind : ℝ := 400

/-- The distance flown against the wind -/
def distance_against_wind : ℝ := 320

theorem wind_speed_calculation :
  (distance_with_wind / (plane_speed + wind_speed) = 
   distance_against_wind / (plane_speed - wind_speed)) ∧
  wind_speed = 20 := by
  sorry

end wind_speed_calculation_l2578_257856


namespace inequality_solution_l2578_257805

theorem inequality_solution (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x < 4/3 ∨ x > 2 := by sorry

end inequality_solution_l2578_257805


namespace expression_value_l2578_257866

theorem expression_value (a b : ℤ) (h : a - b = 1) : 2*b - (2*a + 6) = -8 := by
  sorry

end expression_value_l2578_257866


namespace factor_expression_l2578_257873

theorem factor_expression (x : ℝ) : 54 * x^6 - 231 * x^13 = 3 * x^6 * (18 - 77 * x^7) := by
  sorry

end factor_expression_l2578_257873


namespace sqrt_15_minus_1_range_l2578_257863

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  sorry

end sqrt_15_minus_1_range_l2578_257863


namespace factorization_identities_l2578_257871

theorem factorization_identities (a b x m : ℝ) :
  (2 * a^2 - 2*a*b = 2*a*(a-b)) ∧
  (2 * x^2 - 18 = 2*(x+3)*(x-3)) ∧
  (-3*m*a^3 + 6*m*a^2 - 3*m*a = -3*m*a*(a-1)^2) := by
  sorry

end factorization_identities_l2578_257871


namespace badminton_cost_theorem_l2578_257824

/-- Represents the cost of badminton equipment under different purchasing options -/
def BadmintonCost (x : ℕ) : Prop :=
  let racket_price : ℕ := 40
  let shuttlecock_price : ℕ := 10
  let racket_quantity : ℕ := 10
  let shuttlecock_quantity : ℕ := x
  let option1_cost : ℕ := 10 * x + 300
  let option2_cost : ℕ := 9 * x + 360
  x > 10 ∧
  option1_cost = racket_price * racket_quantity + shuttlecock_price * (shuttlecock_quantity - racket_quantity) ∧
  option2_cost = (racket_price * racket_quantity + shuttlecock_price * shuttlecock_quantity) * 9 / 10 ∧
  (x = 30 → option1_cost < option2_cost) ∧
  ∃ (better_cost : ℕ), x = 30 → better_cost < option1_cost ∧ better_cost < option2_cost

theorem badminton_cost_theorem : 
  ∀ x : ℕ, BadmintonCost x :=
sorry

end badminton_cost_theorem_l2578_257824


namespace sum_g_15_neg_15_l2578_257845

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 - e * x^6 + f * x^2 + 5

-- Theorem statement
theorem sum_g_15_neg_15 (d e f : ℝ) (h : g d e f 15 = 7) :
  g d e f 15 + g d e f (-15) = 14 := by
  sorry

end sum_g_15_neg_15_l2578_257845


namespace cloth_trimming_l2578_257874

/-- The number of feet trimmed from opposite edges of a square cloth -/
def x : ℝ := 15

/-- The original side length of the square cloth in feet -/
def original_side : ℝ := 22

/-- The remaining area of cloth after trimming in square feet -/
def remaining_area : ℝ := 120

/-- The number of feet trimmed from the other two edges -/
def other_edge_trim : ℝ := 5

theorem cloth_trimming :
  round ((original_side - x) * (original_side - other_edge_trim) - remaining_area) = 0 :=
sorry

end cloth_trimming_l2578_257874


namespace quadratic_function_behavior_l2578_257800

def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 4

theorem quadratic_function_behavior (b : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ ∧ x₂ ≤ -1 → f b x₁ ≥ f b x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ ≤ x₂ → f b x₁ ≤ f b x₂) →
  b > 0 := by
  sorry

end quadratic_function_behavior_l2578_257800


namespace conference_duration_theorem_l2578_257812

/-- Calculates the duration of a conference excluding breaks -/
def conference_duration_excluding_breaks (total_hours : ℕ) (total_minutes : ℕ) (break_duration : ℕ) : ℕ :=
  let total_duration := total_hours * 60 + total_minutes
  let total_breaks := total_hours * break_duration
  total_duration - total_breaks

/-- Proves that a conference lasting 14 hours and 20 minutes with 15-minute breaks after each hour has a duration of 650 minutes excluding breaks -/
theorem conference_duration_theorem :
  conference_duration_excluding_breaks 14 20 15 = 650 := by
  sorry

end conference_duration_theorem_l2578_257812


namespace special_rectangle_side_gt_12_l2578_257872

/-- A rectangle with sides a and b, where the area is three times the perimeter --/
structure SpecialRectangle where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a ≠ b
  h4 : a * b = 3 * (2 * (a + b))

/-- Theorem: For a SpecialRectangle, one of its sides is greater than 12 --/
theorem special_rectangle_side_gt_12 (rect : SpecialRectangle) : max rect.a rect.b > 12 := by
  sorry

end special_rectangle_side_gt_12_l2578_257872


namespace money_distribution_l2578_257841

/-- Given that A, B, and C have a total of 250 Rs., and A and C together have 200 Rs.,
    prove that B has 50 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 250)  -- Total money of A, B, and C
  (h2 : A + C = 200)      -- Money of A and C together
  : B = 50 := by
  sorry

end money_distribution_l2578_257841


namespace passing_percentage_l2578_257850

theorem passing_percentage (max_marks : ℕ) (pradeep_marks : ℕ) (failed_by : ℕ) 
  (h1 : max_marks = 925)
  (h2 : pradeep_marks = 160)
  (h3 : failed_by = 25) :
  (((pradeep_marks + failed_by : ℚ) / max_marks) * 100 : ℚ) = 20 := by
  sorry

end passing_percentage_l2578_257850


namespace product_evaluation_l2578_257885

theorem product_evaluation : (2.5 : ℝ) * (50.5 + 0.15) = 126.625 := by
  sorry

end product_evaluation_l2578_257885


namespace range_of_m_for_empty_intersection_l2578_257832

/-- The set A defined by the quadratic equation mx^2 + x + m = 0 -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + x + m = 0}

/-- Theorem stating the range of m for which A has no real solutions -/
theorem range_of_m_for_empty_intersection :
  (∀ m : ℝ, (A m ∩ Set.univ = ∅) ↔ (m < -1/2 ∨ m > 1/2)) := by sorry

end range_of_m_for_empty_intersection_l2578_257832


namespace parabola_properties_l2578_257859

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

-- Define the conditions
theorem parabola_properties :
  ∃ (a b : ℝ),
    (parabola a b 3 = 0) ∧
    (parabola a b 4 = 3) ∧
    (∀ x, parabola a b x = x^2 - 4*x + 3) ∧
    (a > 0) ∧
    (∀ x, parabola a b x ≥ parabola a b 2) ∧
    (parabola a b 2 = -1) :=
by sorry

end parabola_properties_l2578_257859


namespace eighteen_mangoes_yield_fortyeight_lassis_l2578_257869

/-- Given that 3 mangoes make 8 lassis, this function calculates
    the number of lassis that can be made from a given number of mangoes. -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (mangoes * 8) / 3

/-- Theorem stating that 18 mangoes will yield 48 lassis, 
    given the ratio of 8 lassis to 3 mangoes. -/
theorem eighteen_mangoes_yield_fortyeight_lassis :
  lassis_from_mangoes 18 = 48 := by
  sorry

#eval lassis_from_mangoes 18

end eighteen_mangoes_yield_fortyeight_lassis_l2578_257869


namespace books_left_over_l2578_257825

theorem books_left_over (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1335 →
  books_per_initial_box = 39 →
  books_per_new_box = 40 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 25 := by
  sorry

end books_left_over_l2578_257825
