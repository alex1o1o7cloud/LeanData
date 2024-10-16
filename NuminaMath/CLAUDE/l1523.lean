import Mathlib

namespace NUMINAMATH_CALUDE_negation_existential_real_l1523_152374

theorem negation_existential_real (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_real_l1523_152374


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1523_152381

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + (k - 1) * x + 2 > 0) ↔ k ∈ Set.Icc 1 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1523_152381


namespace NUMINAMATH_CALUDE_mikes_bills_l1523_152387

theorem mikes_bills (total_amount : ℕ) (bill_value : ℕ) (num_bills : ℕ) :
  total_amount = 45 →
  bill_value = 5 →
  total_amount = bill_value * num_bills →
  num_bills = 9 := by
sorry

end NUMINAMATH_CALUDE_mikes_bills_l1523_152387


namespace NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l1523_152320

theorem triangle_arithmetic_angle_sequence (A B C : Real) : 
  -- The angles form a triangle
  A + B + C = Real.pi →
  -- The angles form an arithmetic sequence
  A + C = 2 * B →
  -- Prove that sin B = √3/2
  Real.sin B = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l1523_152320


namespace NUMINAMATH_CALUDE_total_arrangements_with_at_least_one_girl_l1523_152373

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def choose (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_people : ℕ := num_boys + num_girls
def num_selected : ℕ := 3
def num_tasks : ℕ := 3

theorem total_arrangements_with_at_least_one_girl : 
  (choose num_people num_selected - choose num_boys num_selected) * factorial num_tasks = 186 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_with_at_least_one_girl_l1523_152373


namespace NUMINAMATH_CALUDE_tree_height_equation_l1523_152325

/-- Represents the height of a tree over time -/
def tree_height (initial_height growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_height + growth_rate * months

/-- Theorem stating the relationship between tree height and time -/
theorem tree_height_equation (h x : ℝ) :
  h = tree_height 80 2 x ↔ h = 80 + 2 * x :=
by sorry

end NUMINAMATH_CALUDE_tree_height_equation_l1523_152325


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1523_152356

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x < -1 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1523_152356


namespace NUMINAMATH_CALUDE_range_a_theorem_l1523_152331

open Set

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a : Set ℝ := Ioo (-2) (-1) ∪ Ici 1

-- Theorem statement
theorem range_a_theorem (h1 : ∀ a : ℝ, p a ∨ q a) (h2 : ¬ ∃ a : ℝ, p a ∧ q a) :
  ∀ a : ℝ, a ∈ range_of_a ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a) :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1523_152331


namespace NUMINAMATH_CALUDE_coffee_expense_theorem_l1523_152327

/-- Calculates the weekly coffee expense for a household -/
def weekly_coffee_expense (
  num_people : ℕ
) (cups_per_person_per_day : ℕ
) (ounces_per_cup : ℚ
) (price_per_ounce : ℚ
) : ℚ :=
  (num_people * cups_per_person_per_day : ℚ) *
  ounces_per_cup *
  price_per_ounce *
  7

/-- Proves that the weekly coffee expense for the given conditions is $35 -/
theorem coffee_expense_theorem :
  weekly_coffee_expense 4 2 (1/2) (5/4) = 35 := by
  sorry

end NUMINAMATH_CALUDE_coffee_expense_theorem_l1523_152327


namespace NUMINAMATH_CALUDE_rectangular_field_fencing_costs_l1523_152371

/-- Given a rectangular field with sides in the ratio of 3:4 and an area of 8112 sq.m,
    prove the perimeter and fencing costs for different materials. -/
theorem rectangular_field_fencing_costs 
  (ratio : ℚ) 
  (area : ℝ) 
  (wrought_iron_cost : ℝ) 
  (wooden_cost : ℝ) 
  (chain_link_cost : ℝ) :
  ratio = 3 / 4 →
  area = 8112 →
  wrought_iron_cost = 45 →
  wooden_cost = 35 →
  chain_link_cost = 25 →
  ∃ (perimeter : ℝ) 
    (wrought_iron_total : ℝ) 
    (wooden_total : ℝ) 
    (chain_link_total : ℝ),
    perimeter = 364 ∧
    wrought_iron_total = 16380 ∧
    wooden_total = 12740 ∧
    chain_link_total = 9100 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_fencing_costs_l1523_152371


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1523_152385

theorem quadratic_inequality_equivalence (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1523_152385


namespace NUMINAMATH_CALUDE_customer_total_cost_l1523_152368

-- Define the quantities and prices of items
def riqing_quantity : ℕ := 24
def riqing_price : ℚ := 1.80
def riqing_discount : ℚ := 0.8

def kangshifu_quantity : ℕ := 6
def kangshifu_price : ℚ := 1.70
def kangshifu_discount : ℚ := 0.8

def shanlin_quantity : ℕ := 5
def shanlin_price : ℚ := 3.40
def shanlin_discount : ℚ := 1  -- No discount

def shuanghui_quantity : ℕ := 3
def shuanghui_price : ℚ := 11.20
def shuanghui_discount : ℚ := 0.9

-- Define the total cost function
def total_cost : ℚ :=
  riqing_quantity * riqing_price * riqing_discount +
  kangshifu_quantity * kangshifu_price * kangshifu_discount +
  shanlin_quantity * shanlin_price * shanlin_discount +
  shuanghui_quantity * shuanghui_price * shuanghui_discount

-- Theorem statement
theorem customer_total_cost : total_cost = 89.96 := by
  sorry

end NUMINAMATH_CALUDE_customer_total_cost_l1523_152368


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_dot_product_l1523_152388

/-- The ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- A line with inclination angle 45° passing through a focus of the ellipse -/
def Line (f : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 - f.1}

/-- The dot product of two points in ℝ² -/
def dotProduct (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

theorem ellipse_line_intersection_dot_product :
  ∀ f A B : ℝ × ℝ,
  f ∈ Ellipse →
  f.2 = 0 →
  A ∈ Ellipse →
  B ∈ Ellipse →
  A ∈ Line f →
  B ∈ Line f →
  A ≠ B →
  dotProduct A B = -1/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_dot_product_l1523_152388


namespace NUMINAMATH_CALUDE_projective_transformation_existence_l1523_152315

-- Define a projective plane
class ProjectivePlane (P : Type*) :=
  (Line : Type*)
  (incidence : P → Line → Prop)
  (axiom_existence : ∀ (A B : P), ∃ (l : Line), incidence A l ∧ incidence B l)
  (axiom_uniqueness : ∀ (A B : P) (l m : Line), incidence A l → incidence B l → incidence A m → incidence B m → l = m)
  (axiom_nondegeneracy : ∃ (A B C : P), ¬∃ (l : Line), incidence A l ∧ incidence B l ∧ incidence C l)

-- Define a projective transformation
def ProjectiveTransformation (P : Type*) [ProjectivePlane P] := P → P

-- Define the property of four points being non-collinear
def NonCollinear {P : Type*} [ProjectivePlane P] (A B C D : P) : Prop :=
  ¬∃ (l : ProjectivePlane.Line P), ProjectivePlane.incidence A l ∧ ProjectivePlane.incidence B l ∧ ProjectivePlane.incidence C l ∧ ProjectivePlane.incidence D l

-- State the theorem
theorem projective_transformation_existence
  {P : Type*} [ProjectivePlane P]
  (A B C D A₁ B₁ C₁ D₁ : P)
  (h1 : NonCollinear A B C D)
  (h2 : NonCollinear A₁ B₁ C₁ D₁) :
  ∃ (f : ProjectiveTransformation P),
    f A = A₁ ∧ f B = B₁ ∧ f C = C₁ ∧ f D = D₁ :=
sorry

end NUMINAMATH_CALUDE_projective_transformation_existence_l1523_152315


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l1523_152336

/-- The number of ways to arrange 5 people in a row with two specific people having exactly one person between them -/
def arrangement_count : ℕ := 36

/-- The number of people in the row -/
def total_people : ℕ := 5

/-- The number of people that can be placed between the two specific people -/
def middle_choices : ℕ := 3

/-- The number of ways to arrange the two specific people with one person between them -/
def specific_arrangement : ℕ := 2

/-- The number of ways to arrange the group of three (two specific people and the one between them) with the other two people -/
def group_arrangement : ℕ := 6

theorem correct_arrangement_count :
  arrangement_count = middle_choices * specific_arrangement * group_arrangement :=
sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l1523_152336


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1523_152399

/-- An arithmetic sequence -/
def ArithmeticSequence (b : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_k_value
  (b : ℕ → ℚ)
  (h_arith : ArithmeticSequence b)
  (h_sum1 : b 5 + b 8 + b 11 = 21)
  (h_sum2 : (Finset.range 11).sum (fun i => b (i + 5)) = 121)
  (h_bk : ∃ k : ℕ, b k = 23) :
  ∃ k : ℕ, b k = 23 ∧ k = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1523_152399


namespace NUMINAMATH_CALUDE_equation_solution_l1523_152311

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 3/5 ∧ 
  (∀ x : ℝ, (x - 3)^2 + 4*x*(x - 3) = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1523_152311


namespace NUMINAMATH_CALUDE_isabella_book_purchase_l1523_152372

/-- The number of hardcover volumes bought by Isabella --/
def num_hardcovers : ℕ := 6

/-- The number of paperback volumes bought by Isabella --/
def num_paperbacks : ℕ := 12 - num_hardcovers

/-- The cost of a paperback volume in dollars --/
def paperback_cost : ℕ := 20

/-- The cost of a hardcover volume in dollars --/
def hardcover_cost : ℕ := 30

/-- The total number of volumes --/
def total_volumes : ℕ := 12

/-- The total cost of all volumes in dollars --/
def total_cost : ℕ := 300

theorem isabella_book_purchase :
  num_hardcovers = 6 ∧
  num_hardcovers + num_paperbacks = total_volumes ∧
  num_hardcovers * hardcover_cost + num_paperbacks * paperback_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_isabella_book_purchase_l1523_152372


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1523_152375

theorem perfect_square_condition (a b : ℤ) : 
  (∀ m n : ℕ, ∃ k : ℤ, a * m^2 + b * n^2 = k^2) → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1523_152375


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1523_152326

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1523_152326


namespace NUMINAMATH_CALUDE_five_lines_max_sections_l1523_152307

/-- The maximum number of sections created by drawing n line segments through a rectangle -/
def max_sections (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem five_lines_max_sections : max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_max_sections_l1523_152307


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1523_152301

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_3 + a_9 = 15 - a_6, prove that a_6 = 5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_equation : a 3 + a 9 = 15 - a 6) : 
  a 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1523_152301


namespace NUMINAMATH_CALUDE_bcm_hens_percentage_l1523_152346

/-- Given a farm with chickens, calculate the percentage of Black Copper Marans hens -/
theorem bcm_hens_percentage 
  (total_chickens : ℕ) 
  (bcm_percentage : ℚ) 
  (bcm_hens : ℕ) 
  (h1 : total_chickens = 100) 
  (h2 : bcm_percentage = 1/5) 
  (h3 : bcm_hens = 16) : 
  (bcm_hens : ℚ) / (bcm_percentage * total_chickens) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_bcm_hens_percentage_l1523_152346


namespace NUMINAMATH_CALUDE_angle_three_measure_l1523_152305

def minutes_to_degrees (m : ℕ) : ℚ := m / 60

def angle_measure (degrees : ℕ) (minutes : ℕ) : ℚ := degrees + minutes_to_degrees minutes

theorem angle_three_measure 
  (angle1 angle2 angle3 : ℚ) 
  (h1 : angle1 + angle2 = 90) 
  (h2 : angle2 + angle3 = 180) 
  (h3 : angle1 = angle_measure 67 12) : 
  angle3 = angle_measure 157 12 := by
sorry

end NUMINAMATH_CALUDE_angle_three_measure_l1523_152305


namespace NUMINAMATH_CALUDE_ebook_count_l1523_152395

def total_ebooks (anna_bought : ℕ) (john_diff : ℕ) (john_lost : ℕ) (mary_factor : ℕ) (mary_gave : ℕ) : ℕ :=
  let john_bought := anna_bought - john_diff
  let john_has := john_bought - john_lost
  let mary_bought := mary_factor * john_bought
  let mary_has := mary_bought - mary_gave
  anna_bought + john_has + mary_has

theorem ebook_count :
  total_ebooks 50 15 3 2 7 = 145 := by
  sorry

end NUMINAMATH_CALUDE_ebook_count_l1523_152395


namespace NUMINAMATH_CALUDE_decagon_diagonals_l1523_152308

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l1523_152308


namespace NUMINAMATH_CALUDE_cars_meeting_time_l1523_152341

/-- Two cars traveling from opposite ends of a highway meet after a certain time. -/
theorem cars_meeting_time
  (highway_length : ℝ)
  (car1_speed : ℝ)
  (car2_speed : ℝ)
  (h1 : highway_length = 175)
  (h2 : car1_speed = 25)
  (h3 : car2_speed = 45) :
  (highway_length / (car1_speed + car2_speed)) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_cars_meeting_time_l1523_152341


namespace NUMINAMATH_CALUDE_cube_construction_condition_l1523_152376

/-- A brick is composed of twelve unit cubes arranged in a three-step staircase of width 2. -/
def Brick : Type := Unit

/-- Predicate indicating whether it's possible to build a cube of side length n using Bricks. -/
def CanBuildCube (n : ℕ) : Prop := sorry

theorem cube_construction_condition (n : ℕ) : 
  CanBuildCube n ↔ 12 ∣ n :=
sorry

end NUMINAMATH_CALUDE_cube_construction_condition_l1523_152376


namespace NUMINAMATH_CALUDE_geometric_sum_seven_halves_l1523_152390

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_seven_halves :
  geometric_sum (1/2) (1/2) 7 = 127/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_seven_halves_l1523_152390


namespace NUMINAMATH_CALUDE_vector_sum_length_one_l1523_152349

theorem vector_sum_length_one (x : Real) :
  let a := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
  let b := (Real.cos (x / 2), -Real.sin (x / 2))
  (0 ≤ x) ∧ (x ≤ Real.pi) →
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1 →
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_length_one_l1523_152349


namespace NUMINAMATH_CALUDE_franks_books_l1523_152380

theorem franks_books (days_per_book : ℕ) (total_days : ℕ) (h1 : days_per_book = 12) (h2 : total_days = 492) :
  total_days / days_per_book = 41 := by
  sorry

end NUMINAMATH_CALUDE_franks_books_l1523_152380


namespace NUMINAMATH_CALUDE_boxes_problem_l1523_152343

theorem boxes_problem (stan jules joseph john : ℕ) : 
  stan = 100 →
  joseph = stan / 5 →
  jules = joseph + 5 →
  john = jules + jules / 5 →
  john = 30 := by
sorry

end NUMINAMATH_CALUDE_boxes_problem_l1523_152343


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l1523_152367

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The set of points satisfying the given inequalities -/
def SatisfyingPoints : Set Point :=
  {p : Point | p.y > 3 * p.x ∧ p.y > 5 - 2 * p.x}

/-- A point is in Quadrant I if both x and y are positive -/
def InQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- A point is in Quadrant II if x is negative and y is positive -/
def InQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: All points satisfying the inequalities are in Quadrants I or II -/
theorem points_in_quadrants_I_and_II :
  ∀ p ∈ SatisfyingPoints, InQuadrantI p ∨ InQuadrantII p :=
by sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l1523_152367


namespace NUMINAMATH_CALUDE_car_price_difference_car_price_difference_proof_l1523_152354

/-- The price difference between a new car and an old car, given specific conditions --/
theorem car_price_difference : ℝ → Prop :=
  fun price_difference =>
    ∃ (old_car_price : ℝ),
      -- New car costs $30,000
      let new_car_price : ℝ := 30000
      -- Down payment is 25% of new car price
      let down_payment : ℝ := 0.25 * new_car_price
      -- Old car sold at 80% of original price
      let old_car_sale_price : ℝ := 0.8 * old_car_price
      -- After selling old car and making down payment, $4000 more is needed
      old_car_sale_price + down_payment + 4000 = new_car_price ∧
      -- Price difference is the difference between new and old car prices
      price_difference = new_car_price - old_car_price ∧
      -- The price difference is $6875
      price_difference = 6875

/-- Proof of the car price difference theorem --/
theorem car_price_difference_proof : car_price_difference 6875 := by
  sorry

end NUMINAMATH_CALUDE_car_price_difference_car_price_difference_proof_l1523_152354


namespace NUMINAMATH_CALUDE_inverse_proportional_solution_l1523_152384

def inverse_proportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportional_solution (x y : ℝ) 
  (h1 : inverse_proportional x y) 
  (h2 : x + y = 30) 
  (h3 : x - y = 10) : 
  (∃ y' : ℝ, inverse_proportional 8 y' ∧ y' = 25) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportional_solution_l1523_152384


namespace NUMINAMATH_CALUDE_smallest_solution_5x2_eq_3y5_l1523_152330

theorem smallest_solution_5x2_eq_3y5 :
  ∃! (x y : ℕ), 
    (5 * x^2 = 3 * y^5) ∧ 
    (∀ (a b : ℕ), (5 * a^2 = 3 * b^5) → (x ≤ a ∧ y ≤ b)) ∧
    x = 675 ∧ y = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_solution_5x2_eq_3y5_l1523_152330


namespace NUMINAMATH_CALUDE_total_matches_proof_l1523_152391

/-- Represents the number of matches for a team -/
structure MatchRecord where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Calculate the total number of matches played by a team -/
def totalMatches (record : MatchRecord) : ℕ :=
  record.wins + record.draws + record.losses

theorem total_matches_proof
  (home : MatchRecord)
  (rival : MatchRecord)
  (h1 : rival.wins = 2 * home.wins)
  (h2 : home.wins = 3)
  (h3 : home.draws = 4)
  (h4 : rival.draws = 4)
  (h5 : home.losses = 0)
  (h6 : rival.losses = 0) :
  totalMatches home + totalMatches rival = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_proof_l1523_152391


namespace NUMINAMATH_CALUDE_friday_price_calculation_l1523_152321

theorem friday_price_calculation (tuesday_price : ℝ) : 
  tuesday_price = 50 →
  let wednesday_price := tuesday_price * (1 + 0.2)
  let friday_price := wednesday_price * (1 - 0.15)
  friday_price = 51 := by
sorry

end NUMINAMATH_CALUDE_friday_price_calculation_l1523_152321


namespace NUMINAMATH_CALUDE_tshirts_per_package_l1523_152323

theorem tshirts_per_package (total_packages : ℕ) (total_tshirts : ℕ) 
  (h1 : total_packages = 71) 
  (h2 : total_tshirts = 426) : 
  total_tshirts / total_packages = 6 := by
  sorry

end NUMINAMATH_CALUDE_tshirts_per_package_l1523_152323


namespace NUMINAMATH_CALUDE_original_selling_price_l1523_152322

/-- The original selling price given the profit rates and price difference -/
theorem original_selling_price 
  (original_profit_rate : ℝ)
  (reduced_purchase_rate : ℝ)
  (new_profit_rate : ℝ)
  (price_difference : ℝ)
  (h1 : original_profit_rate = 0.1)
  (h2 : reduced_purchase_rate = 0.1)
  (h3 : new_profit_rate = 0.3)
  (h4 : price_difference = 49) :
  ∃ (purchase_price : ℝ),
    (1 + original_profit_rate) * purchase_price = 770 ∧
    ((1 - reduced_purchase_rate) * (1 + new_profit_rate) - (1 + original_profit_rate)) * purchase_price = price_difference :=
by sorry

end NUMINAMATH_CALUDE_original_selling_price_l1523_152322


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l1523_152398

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number -/
def is_3digit_base8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), is_3digit_base8 n ∧ 
             base8_to_base10 n % 7 = 0 ∧
             ∀ (m : ℕ), is_3digit_base8 m ∧ base8_to_base10 m % 7 = 0 → m ≤ n :=
by
  use 777
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l1523_152398


namespace NUMINAMATH_CALUDE_star_property_l1523_152317

def star (m n : ℝ) : ℝ := (3 * m - 2 * n)^2

theorem star_property (x y : ℝ) : star ((3 * x - 2 * y)^2) ((2 * y - 3 * x)^2) = (3 * x - 2 * y)^4 := by
  sorry

end NUMINAMATH_CALUDE_star_property_l1523_152317


namespace NUMINAMATH_CALUDE_family_brownie_consumption_percentage_l1523_152355

theorem family_brownie_consumption_percentage
  (total_brownies : ℕ)
  (children_consumption_percentage : ℚ)
  (lorraine_extra_consumption : ℕ)
  (leftover_brownies : ℕ)
  (h1 : total_brownies = 16)
  (h2 : children_consumption_percentage = 1/4)
  (h3 : lorraine_extra_consumption = 1)
  (h4 : leftover_brownies = 5) :
  let remaining_after_children := total_brownies - (children_consumption_percentage * total_brownies).num
  let family_consumption := remaining_after_children - leftover_brownies - lorraine_extra_consumption
  (family_consumption : ℚ) / remaining_after_children = 1/2 :=
sorry

end NUMINAMATH_CALUDE_family_brownie_consumption_percentage_l1523_152355


namespace NUMINAMATH_CALUDE_randy_initial_money_l1523_152377

/-- Randy's initial amount of money -/
def initial_money : ℕ := sorry

/-- Amount Smith gave to Randy -/
def smith_gave : ℕ := 200

/-- Amount Randy gave to Sally -/
def sally_received : ℕ := 1200

/-- Amount Randy kept after giving money to Sally -/
def randy_kept : ℕ := 2000

theorem randy_initial_money :
  initial_money + smith_gave - sally_received = randy_kept ∧
  initial_money = 3000 := by sorry

end NUMINAMATH_CALUDE_randy_initial_money_l1523_152377


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l1523_152370

def has_three_integer_solutions (b : ℤ) : Prop :=
  ∃ x y z : ℤ, x < y ∧ y < z ∧
    (x^2 + b*x - 2 ≤ 0) ∧
    (y^2 + b*y - 2 ≤ 0) ∧
    (z^2 + b*z - 2 ≤ 0) ∧
    ∀ w : ℤ, (w^2 + b*w - 2 ≤ 0) → (w = x ∨ w = y ∨ w = z)

theorem quadratic_inequality_solutions :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ b : ℤ, b ∈ s ↔ has_three_integer_solutions b :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l1523_152370


namespace NUMINAMATH_CALUDE_greatest_fraction_l1523_152358

theorem greatest_fraction : 
  let f1 := (3 : ℚ) / 10
  let f2 := (4 : ℚ) / 7
  let f3 := (5 : ℚ) / 23
  let f4 := (2 : ℚ) / 3
  let f5 := (1 : ℚ) / 2
  f4 > f1 ∧ f4 > f2 ∧ f4 > f3 ∧ f4 > f5 := by sorry

end NUMINAMATH_CALUDE_greatest_fraction_l1523_152358


namespace NUMINAMATH_CALUDE_calculation_proof_l1523_152334

theorem calculation_proof :
  (13 + (-7) + (-6) = 0) ∧
  ((-8) * (-4/3) * (-0.125) * (5/4) = -5/3) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1523_152334


namespace NUMINAMATH_CALUDE_courtyard_length_is_20_l1523_152383

/-- The width of the courtyard in meters -/
def courtyard_width : ℝ := 16.5

/-- The number of paving stones required to cover the courtyard -/
def num_paving_stones : ℕ := 66

/-- The length of a paving stone in meters -/
def paving_stone_length : ℝ := 2.5

/-- The width of a paving stone in meters -/
def paving_stone_width : ℝ := 2

/-- The theorem stating that the length of the courtyard is 20 meters -/
theorem courtyard_length_is_20 : 
  (courtyard_width * (num_paving_stones * paving_stone_length * paving_stone_width) / courtyard_width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_is_20_l1523_152383


namespace NUMINAMATH_CALUDE_three_Z_five_l1523_152337

/-- The operation Z defined on real numbers -/
def Z (a b : ℝ) : ℝ := b + 7*a - 3*a^2

/-- Theorem stating that 3 Z 5 = -1 -/
theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_l1523_152337


namespace NUMINAMATH_CALUDE_view_characteristics_l1523_152309

/-- Represents the view described in the problem -/
structure View where
  endless_progress : Bool
  unlimited_capacity : Bool
  unlimited_resources : Bool

/-- Represents the characteristics of the view -/
structure ViewCharacteristics where
  emphasizes_subjective_initiative : Bool
  ignores_objective_conditions : Bool

/-- Theorem stating that the given view unilaterally emphasizes subjective initiative
    while ignoring objective conditions and laws -/
theorem view_characteristics (v : View) 
  (h1 : v.endless_progress = true)
  (h2 : v.unlimited_capacity = true)
  (h3 : v.unlimited_resources = true) :
  ∃ (c : ViewCharacteristics), 
    c.emphasizes_subjective_initiative ∧ c.ignores_objective_conditions := by
  sorry


end NUMINAMATH_CALUDE_view_characteristics_l1523_152309


namespace NUMINAMATH_CALUDE_brooke_jacks_eight_days_l1523_152318

/-- Represents the number of jumping jacks Sidney does on a given day -/
def sidney_jacks : Nat → Nat
  | 0 => 20  -- Monday
  | 1 => 36  -- Tuesday
  | n + 2 => sidney_jacks (n + 1) + (16 + 2 * n)  -- Following days

/-- The total number of jumping jacks Sidney does over 8 days -/
def sidney_total : Nat := (List.range 8).map sidney_jacks |>.sum

/-- The number of jumping jacks Brooke does is four times Sidney's -/
def brooke_total : Nat := 4 * sidney_total

theorem brooke_jacks_eight_days : brooke_total = 2880 := by
  sorry

end NUMINAMATH_CALUDE_brooke_jacks_eight_days_l1523_152318


namespace NUMINAMATH_CALUDE_tshirt_shop_profit_l1523_152342

theorem tshirt_shop_profit : 
  let profit_per_shirt : ℚ := 9
  let cost_per_shirt : ℚ := 4
  let num_shirts : ℕ := 245
  let discount_rate : ℚ := 1/5

  let original_price : ℚ := profit_per_shirt + cost_per_shirt
  let discounted_price : ℚ := original_price * (1 - discount_rate)
  let total_revenue : ℚ := (discounted_price * num_shirts : ℚ)
  let total_cost : ℚ := (cost_per_shirt * num_shirts : ℚ)
  let total_profit : ℚ := total_revenue - total_cost

  total_profit = 1568 := by sorry

end NUMINAMATH_CALUDE_tshirt_shop_profit_l1523_152342


namespace NUMINAMATH_CALUDE_committee_seating_arrangements_l1523_152357

/-- The number of Democrats on the committee -/
def num_democrats : ℕ := 6

/-- The number of Republicans on the committee -/
def num_republicans : ℕ := 4

/-- The total number of politicians on the committee -/
def total_politicians : ℕ := num_democrats + num_republicans

/-- Represents that all politicians are distinguishable -/
axiom politicians_distinguishable : True

/-- Represents the constraint that no two Republicans can sit next to each other -/
axiom no_adjacent_republicans : True

/-- The number of ways to arrange the politicians around a circular table -/
def arrangement_count : ℕ := 43200

/-- Theorem stating that the number of valid arrangements is 43,200 -/
theorem committee_seating_arrangements :
  arrangement_count = 43200 :=
sorry

end NUMINAMATH_CALUDE_committee_seating_arrangements_l1523_152357


namespace NUMINAMATH_CALUDE_melanie_yard_sale_books_l1523_152379

/-- The number of books Melanie bought at a yard sale -/
def books_bought (initial_books final_books : ℝ) : ℝ :=
  final_books - initial_books

/-- Proof that Melanie bought 87 books at the yard sale -/
theorem melanie_yard_sale_books : books_bought 41.0 128 = 87 := by
  sorry

end NUMINAMATH_CALUDE_melanie_yard_sale_books_l1523_152379


namespace NUMINAMATH_CALUDE_number_product_l1523_152393

theorem number_product (x : ℝ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_product_l1523_152393


namespace NUMINAMATH_CALUDE_small_forward_duration_l1523_152314

/-- Represents the duration of footage for each player in seconds. -/
structure PlayerFootage where
  pointGuard : ℕ
  shootingGuard : ℕ
  smallForward : ℕ
  powerForward : ℕ
  center : ℕ

/-- Calculates the total duration of all players' footage in seconds. -/
def totalDuration (pf : PlayerFootage) : ℕ :=
  pf.pointGuard + pf.shootingGuard + pf.smallForward + pf.powerForward + pf.center

/-- The number of players in the team. -/
def numPlayers : ℕ := 5

/-- The average duration per player in seconds. -/
def avgDurationPerPlayer : ℕ := 120 -- 2 minutes = 120 seconds

theorem small_forward_duration (pf : PlayerFootage) 
    (h1 : pf.pointGuard = 130)
    (h2 : pf.shootingGuard = 145)
    (h3 : pf.powerForward = 60)
    (h4 : pf.center = 180)
    (h5 : totalDuration pf = numPlayers * avgDurationPerPlayer) :
    pf.smallForward = 85 := by
  sorry

end NUMINAMATH_CALUDE_small_forward_duration_l1523_152314


namespace NUMINAMATH_CALUDE_mark_milk_purchase_l1523_152360

def problem (soup_price : ℕ) (soup_quantity : ℕ) (bread_price : ℕ) (bread_quantity : ℕ) 
             (cereal_price : ℕ) (cereal_quantity : ℕ) (milk_price : ℕ) (bill_value : ℕ) 
             (bill_quantity : ℕ) : ℕ :=
  let total_paid := bill_value * bill_quantity
  let other_items_cost := soup_price * soup_quantity + bread_price * bread_quantity + cereal_price * cereal_quantity
  let milk_total_cost := total_paid - other_items_cost
  milk_total_cost / milk_price

theorem mark_milk_purchase :
  problem 2 6 5 2 3 2 4 10 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_milk_purchase_l1523_152360


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1523_152345

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  -- The proof would go here, but we'll skip it
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1523_152345


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1523_152353

theorem cube_root_of_negative_eight :
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1523_152353


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1523_152319

/-- Proves that given a bicycle sold twice with profits of 20% and 25% respectively,
    and a final selling price of 225, the original cost price was 150. -/
theorem bicycle_cost_price 
  (profit_A : Real) 
  (profit_B : Real)
  (final_price : Real)
  (h1 : profit_A = 0.20)
  (h2 : profit_B = 0.25)
  (h3 : final_price = 225) :
  ∃ (initial_price : Real),
    initial_price * (1 + profit_A) * (1 + profit_B) = final_price ∧ 
    initial_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l1523_152319


namespace NUMINAMATH_CALUDE_no_high_grades_l1523_152340

/-- Represents the test scenario with given conditions -/
structure TestScenario where
  n : ℕ  -- number of students excluding Peter
  k : ℕ  -- number of problems solved by each student except Peter
  total_problems_solved : ℕ  -- total number of problems solved by all students

/-- The conditions of the test scenario -/
def valid_scenario (s : TestScenario) : Prop :=
  s.total_problems_solved = 25 ∧
  s.n * s.k + (s.k + 1) = s.total_problems_solved ∧
  s.k ≤ 5

/-- The theorem stating that no student received a grade of 4 or 5 -/
theorem no_high_grades (s : TestScenario) (h : valid_scenario s) : 
  s.k < 4 ∧ s.k + 1 < 5 := by
  sorry

#check no_high_grades

end NUMINAMATH_CALUDE_no_high_grades_l1523_152340


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l1523_152396

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, |n^3 - 250| ≥ |6^3 - 250| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l1523_152396


namespace NUMINAMATH_CALUDE_factor_polynomial_l1523_152324

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1523_152324


namespace NUMINAMATH_CALUDE_scout_troop_profit_l1523_152335

/-- The profit calculation for a scout troop selling candy bars -/
theorem scout_troop_profit :
  -- Number of candy bars
  let n : ℕ := 1500
  -- Buy price (in cents) for 3 bars
  let buy_price : ℕ := 150
  -- Sell price (in cents) for 3 bars
  let sell_price : ℕ := 200
  -- All candy bars are sold (implied in the problem)
  -- Profit calculation (in cents)
  let profit : ℚ := n * sell_price / 3 - n * buy_price / 3
  -- The theorem: profit equals 25050 cents (250.50 dollars)
  profit = 25050 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l1523_152335


namespace NUMINAMATH_CALUDE_max_dot_product_l1523_152378

/-- The ellipse with equation x^2/4 + y^2/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The dot product of vectors OP and FP -/
def dotProduct (P : ℝ × ℝ) : ℝ :=
  (P.1 * (P.1 + 1)) + (P.2 * P.2)

theorem max_dot_product :
  ∃ (M : ℝ), M = 6 ∧ ∀ P ∈ Ellipse, dotProduct P ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l1523_152378


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l1523_152392

/-- Proves that Maxwell walks for 8 hours before meeting Brad given the specified conditions -/
theorem maxwell_brad_meeting_time :
  let distance_between_homes : ℝ := 74
  let maxwell_speed : ℝ := 4
  let brad_speed : ℝ := 6
  let brad_delay : ℝ := 1

  let meeting_time : ℝ := 
    (distance_between_homes - maxwell_speed * brad_delay) / (maxwell_speed + brad_speed)

  let maxwell_total_time : ℝ := meeting_time + brad_delay

  maxwell_total_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l1523_152392


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l1523_152312

/-- 
Given an equation of the form x²/(4-m) + y²/(m-3) = 1 representing an ellipse with foci on the y-axis,
prove that the range of m is (7/2, 4).
-/
theorem ellipse_foci_y_axis_m_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2/(4-m) + y^2/(m-3) = 1) ∧ 
  (∀ (x y : ℝ), x^2/(4-m) + y^2/(m-3) = 1 → (0 : ℝ) < 4-m ∧ (0 : ℝ) < m-3 ∧ m-3 < 4-m) 
  → 7/2 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l1523_152312


namespace NUMINAMATH_CALUDE_greatest_power_of_three_dividing_fifteen_factorial_l1523_152329

theorem greatest_power_of_three_dividing_fifteen_factorial : 
  (∃ k : ℕ, k > 0 ∧ 3^k ∣ Nat.factorial 15 ∧ ∀ m : ℕ, m > k → ¬(3^m ∣ Nat.factorial 15)) → 
  (∃ k : ℕ, k > 0 ∧ 3^k ∣ Nat.factorial 15 ∧ ∀ m : ℕ, m > k → ¬(3^m ∣ Nat.factorial 15) ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_dividing_fifteen_factorial_l1523_152329


namespace NUMINAMATH_CALUDE_train_journey_time_l1523_152366

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  (4/5 * usual_speed) * (usual_time + 3/4) = usual_speed * usual_time → 
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l1523_152366


namespace NUMINAMATH_CALUDE_bhanu_petrol_expense_l1523_152339

theorem bhanu_petrol_expense (income : ℝ) (petrol_percent house_rent_percent : ℝ) 
  (house_rent : ℝ) : 
  petrol_percent = 0.3 →
  house_rent_percent = 0.1 →
  house_rent = 70 →
  house_rent_percent * (income - petrol_percent * income) = house_rent →
  petrol_percent * income = 300 :=
by sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expense_l1523_152339


namespace NUMINAMATH_CALUDE_soccer_preference_and_goals_l1523_152352

/-- Chi-square test statistic for 2x2 contingency table -/
def chi_square (a b c d : ℕ) : ℚ :=
  (200 * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical value for chi-square test at α = 0.001 -/
def critical_value : ℚ := 10828 / 1000

/-- Probability of scoring a goal for male students -/
def p_male : ℚ := 2 / 3

/-- Probability of scoring a goal for female student -/
def p_female : ℚ := 1 / 2

/-- Expected value of goals scored by 2 male and 1 female student -/
def expected_goals : ℚ := 11 / 6

theorem soccer_preference_and_goals (a b c d : ℕ) 
  (h1 : a + b = 100) (h2 : c + d = 100) (h3 : a + c = 90) (h4 : b + d = 110) :
  chi_square a b c d > critical_value ∧ 
  2 * p_male + p_female = expected_goals :=
sorry

end NUMINAMATH_CALUDE_soccer_preference_and_goals_l1523_152352


namespace NUMINAMATH_CALUDE_differential_savings_proof_l1523_152363

def annual_income : ℕ := 45000
def retirement_contribution : ℕ := 4000
def mortgage_interest : ℕ := 5000
def charitable_donations : ℕ := 2000
def previous_tax_rate : ℚ := 40 / 100

def taxable_income : ℕ := annual_income - retirement_contribution - mortgage_interest - charitable_donations

def tax_bracket_1 : ℕ := 10000
def tax_bracket_2 : ℕ := 25000
def tax_bracket_3 : ℕ := 50000

def tax_rate_1 : ℚ := 0 / 100
def tax_rate_2 : ℚ := 10 / 100
def tax_rate_3 : ℚ := 25 / 100
def tax_rate_4 : ℚ := 35 / 100

def new_tax (income : ℕ) : ℚ :=
  if income ≤ tax_bracket_1 then
    income * tax_rate_1
  else if income ≤ tax_bracket_2 then
    tax_bracket_1 * tax_rate_1 + (income - tax_bracket_1) * tax_rate_2
  else if income ≤ tax_bracket_3 then
    tax_bracket_1 * tax_rate_1 + (tax_bracket_2 - tax_bracket_1) * tax_rate_2 + (income - tax_bracket_2) * tax_rate_3
  else
    tax_bracket_1 * tax_rate_1 + (tax_bracket_2 - tax_bracket_1) * tax_rate_2 + (tax_bracket_3 - tax_bracket_2) * tax_rate_3 + (income - tax_bracket_3) * tax_rate_4

theorem differential_savings_proof :
  (annual_income * previous_tax_rate - new_tax taxable_income) = 14250 := by
  sorry

end NUMINAMATH_CALUDE_differential_savings_proof_l1523_152363


namespace NUMINAMATH_CALUDE_thomas_leftover_money_l1523_152361

theorem thomas_leftover_money (num_books : ℕ) (book_price : ℚ) (record_price : ℚ) (num_records : ℕ) :
  num_books = 200 →
  book_price = 3/2 →
  record_price = 3 →
  num_records = 75 →
  (num_books : ℚ) * book_price - (num_records : ℚ) * record_price = 75 :=
by sorry

end NUMINAMATH_CALUDE_thomas_leftover_money_l1523_152361


namespace NUMINAMATH_CALUDE_parallelogram_not_always_axisymmetric_and_centrally_symmetric_l1523_152359

-- Define a parallelogram
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : ∀ (i : Fin 4), 
    vertices i - vertices ((i + 1) % 4) = vertices ((i + 2) % 4) - vertices ((i + 3) % 4)

-- Define an axisymmetric figure
def IsAxisymmetric (vertices : Fin 4 → ℝ × ℝ) : Prop :=
  ∃ (axis : ℝ × ℝ → ℝ × ℝ), ∀ (i : Fin 4), 
    axis (vertices i) = vertices ((4 - i) % 4)

-- Define a centrally symmetric figure
def IsCentrallySymmetric (vertices : Fin 4 → ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ (i : Fin 4), 
    vertices i - center = center - vertices ((i + 2) % 4)

-- Theorem statement
theorem parallelogram_not_always_axisymmetric_and_centrally_symmetric :
  ¬(∀ (p : Parallelogram), IsAxisymmetric p.vertices ∧ IsCentrallySymmetric p.vertices) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_not_always_axisymmetric_and_centrally_symmetric_l1523_152359


namespace NUMINAMATH_CALUDE_complex_sum_equal_negative_three_l1523_152386

theorem complex_sum_equal_negative_three (w : ℂ) 
  (h1 : w = Complex.exp (6 * Real.pi * Complex.I / 11))
  (h2 : w^11 = 1) :
  w / (1 + w^2) + w^2 / (1 + w^4) + w^3 / (1 + w^6) + w^4 / (1 + w^8) = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equal_negative_three_l1523_152386


namespace NUMINAMATH_CALUDE_probability_at_least_6_consecutive_heads_l1523_152369

def coin_flip_sequence := Fin 9 → Bool

def has_at_least_6_consecutive_heads (s : coin_flip_sequence) : Prop :=
  ∃ i, i + 5 < 9 ∧ (∀ j, i ≤ j ∧ j ≤ i + 5 → s j = true)

def total_sequences : ℕ := 2^9

def favorable_sequences : ℕ := 10

theorem probability_at_least_6_consecutive_heads :
  (favorable_sequences : ℚ) / total_sequences = 5 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_6_consecutive_heads_l1523_152369


namespace NUMINAMATH_CALUDE_eight_teams_twentyeight_games_unique_solution_eight_teams_l1523_152333

/-- The number of games played when each team in a conference plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that 8 teams results in 28 games played -/
theorem eight_teams_twentyeight_games :
  ∃ (n : ℕ), n > 0 ∧ games_played n = 28 ∧ n = 8 := by
  sorry

/-- The theorem proving that 8 is the only positive integer satisfying the conditions -/
theorem unique_solution_eight_teams :
  ∀ (n : ℕ), n > 0 → games_played n = 28 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_teams_twentyeight_games_unique_solution_eight_teams_l1523_152333


namespace NUMINAMATH_CALUDE_work_completion_time_l1523_152364

/-- 
Given:
- a and b complete a work in 9 days
- a and b together can do the work in 6 days

Prove: a alone can complete the work in 18 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a + b = 1 / 9) (h2 : a + b = 1 / 6) : a = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1523_152364


namespace NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_for_q_l1523_152389

theorem p_neither_necessary_nor_sufficient_for_q (a : ℝ) : 
  (∃ x, x < 0 ∧ x > x^2) ∧ 
  (∃ y, y < 0 ∧ ¬(y > y^2)) ∧ 
  (∃ z, z > z^2 ∧ ¬(z < 0)) := by
sorry

end NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_for_q_l1523_152389


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_l1523_152306

/-- The time taken to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 9 minutes later than normal, is 36 minutes. -/
theorem bus_stop_walking_time : ∀ T : ℝ, T > 0 → (5 / 4 = (T + 9) / T) → T = 36 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_l1523_152306


namespace NUMINAMATH_CALUDE_min_value_on_line_min_value_achieved_l1523_152397

/-- The minimum value of 2/a + 3/b for points (a, b) in the first quadrant on the line 2x + 3y = 1 -/
theorem min_value_on_line (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2*a + 3*b = 1) :
  2/a + 3/b ≥ 25 := by
  sorry

/-- The minimum value 25 is achieved for some point on the line -/
theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + 3*b = 1 ∧ |2/a + 3/b - 25| < ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_min_value_achieved_l1523_152397


namespace NUMINAMATH_CALUDE_expression_evaluation_l1523_152344

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹) = (a * b)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1523_152344


namespace NUMINAMATH_CALUDE_mushroom_consumption_l1523_152316

theorem mushroom_consumption (initial_amount leftover_amount : ℕ) 
  (h1 : initial_amount = 15)
  (h2 : leftover_amount = 7) :
  initial_amount - leftover_amount = 8 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_consumption_l1523_152316


namespace NUMINAMATH_CALUDE_triangle_properties_l1523_152350

/-- Given a, b, and c are side lengths of a triangle, prove the following properties --/
theorem triangle_properties (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) :
  (a + b - c > 0) ∧ 
  (a - b + c > 0) ∧ 
  (a - b - c < 0) ∧
  (|a + b - c| - |a - b + c| + |a - b - c| = -a + 3*b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1523_152350


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1523_152351

theorem cubic_equation_solution (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1523_152351


namespace NUMINAMATH_CALUDE_equation_system_solution_l1523_152313

/-- Given a system of equations with constants m and n, prove that the solution for a and b is (4, 1) -/
theorem equation_system_solution (m n : ℝ) : 
  (∃ x y : ℝ, -2*m*x + 5*y = 15 ∧ x + 7*n*y = 14 ∧ x = 5 ∧ y = 2) →
  (∃ a b : ℝ, -2*m*(a+b) + 5*(a-2*b) = 15 ∧ (a+b) + 7*n*(a-2*b) = 14 ∧ a = 4 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1523_152313


namespace NUMINAMATH_CALUDE_wire_circle_square_area_l1523_152304

/-- The area of a square formed by a wire that can also form a circle of radius 56 cm is 784π² cm² -/
theorem wire_circle_square_area :
  let r : ℝ := 56  -- radius of the circle in cm
  let circle_circumference : ℝ := 2 * Real.pi * r
  let square_side : ℝ := circle_circumference / 4
  let square_area : ℝ := square_side * square_side
  square_area = 784 * Real.pi ^ 2 := by
    sorry

end NUMINAMATH_CALUDE_wire_circle_square_area_l1523_152304


namespace NUMINAMATH_CALUDE_sam_eating_period_l1523_152338

def apples_per_sandwich : ℕ := 4
def sandwiches_per_day : ℕ := 10
def total_apples : ℕ := 280

theorem sam_eating_period :
  (total_apples / (apples_per_sandwich * sandwiches_per_day) : ℕ) = 7 :=
sorry

end NUMINAMATH_CALUDE_sam_eating_period_l1523_152338


namespace NUMINAMATH_CALUDE_triangle_perimeter_for_radius_3_l1523_152394

/-- A configuration of three circles and a triangle -/
structure CircleTriangleConfig where
  /-- The radius of each circle -/
  radius : ℝ
  /-- The circles are externally tangent to each other -/
  circles_tangent : Prop
  /-- Each side of the triangle is tangent to two of the circles -/
  triangle_tangent : Prop

/-- The perimeter of the triangle in the given configuration -/
def triangle_perimeter (config : CircleTriangleConfig) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the triangle for the given configuration -/
theorem triangle_perimeter_for_radius_3 :
  ∀ (config : CircleTriangleConfig),
    config.radius = 3 →
    config.circles_tangent →
    config.triangle_tangent →
    triangle_perimeter config = 18 + 18 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_for_radius_3_l1523_152394


namespace NUMINAMATH_CALUDE_relay_race_fifth_runner_l1523_152365

def relay_race (t1 t2 t3 t4 t5 : ℝ) : Prop :=
  t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t4 > 0 ∧ t5 > 0 ∧
  (t1/2 + t2 + t3 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.95 ∧
  (t1 + t2/2 + t3 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.90 ∧
  (t1 + t2 + t3/2 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.88 ∧
  (t1 + t2 + t3 + t4/2 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.85

theorem relay_race_fifth_runner (t1 t2 t3 t4 t5 : ℝ) :
  relay_race t1 t2 t3 t4 t5 →
  (t1 + t2 + t3 + t4 + t5/2) / (t1 + t2 + t3 + t4 + t5) = 0.92 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_fifth_runner_l1523_152365


namespace NUMINAMATH_CALUDE_circle_geometric_mean_arc_l1523_152362

theorem circle_geometric_mean_arc (C : ℝ) (α : ℝ) (h : C > 0) :
  let L₁ := (α / 360) * C
  let L₂ := ((360 - α) / 360) * C
  L₁ = Real.sqrt (C * L₂) →
  α = 180 * (Real.sqrt 5 - 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_geometric_mean_arc_l1523_152362


namespace NUMINAMATH_CALUDE_triangle_side_length_l1523_152310

/-- Checks if three lengths can form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In a triangle with sides 5, 8, and a, the only valid value for a is 9 --/
theorem triangle_side_length : ∃! a : ℝ, is_valid_triangle 5 8 a ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1523_152310


namespace NUMINAMATH_CALUDE_distribute_10_balls_3_boxes_l1523_152382

/-- The number of ways to distribute n identical balls into k boxes, where each box i must contain at least i balls. -/
def distributeWithMinimum (n : ℕ) (k : ℕ) : ℕ :=
  let remainingBalls := n - (k * (k + 1) / 2)
  Nat.choose (remainingBalls + k - 1) (k - 1)

/-- Theorem stating that there are 15 ways to distribute 10 identical balls into 3 boxes with the given conditions. -/
theorem distribute_10_balls_3_boxes : distributeWithMinimum 10 3 = 15 := by
  sorry

#eval distributeWithMinimum 10 3

end NUMINAMATH_CALUDE_distribute_10_balls_3_boxes_l1523_152382


namespace NUMINAMATH_CALUDE_cookies_distribution_l1523_152300

/-- Represents the number of cookies the oldest son gets after school -/
def oldest_son_cookies : ℕ := 4

/-- Represents the number of cookies the youngest son gets after school -/
def youngest_son_cookies : ℕ := 2

/-- Represents the total number of cookies in a box -/
def cookies_in_box : ℕ := 54

/-- Represents the number of days the box lasts -/
def days_box_lasts : ℕ := 9

theorem cookies_distribution :
  oldest_son_cookies * days_box_lasts + youngest_son_cookies * days_box_lasts = cookies_in_box :=
by sorry

end NUMINAMATH_CALUDE_cookies_distribution_l1523_152300


namespace NUMINAMATH_CALUDE_no_real_roots_equation_implies_value_l1523_152328

theorem no_real_roots_equation_implies_value (a b : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (x / (x - 1) + (x - 1) / x ≠ (a + b * x) / (x^2 - x))) →
  8 * a + 4 * b - 5 = 11 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_equation_implies_value_l1523_152328


namespace NUMINAMATH_CALUDE_assignment_schemes_l1523_152302

/-- Represents the number of teachers and schools -/
def n : ℕ := 4

/-- The total number of assignment schemes -/
def total_schemes : ℕ := n^n

/-- The number of schemes where exactly one school is not assigned any teachers -/
def one_school_empty : ℕ := n * (n - 1)^(n - 1)

/-- The number of schemes where a certain school is assigned 2 teachers -/
def two_teachers_one_school : ℕ := Nat.choose n 2 * (n - 1)^(n - 2)

/-- The number of schemes where exactly two schools are not assigned any teachers -/
def two_schools_empty : ℕ := Nat.choose n 2 * (Nat.choose n 2 / 2 + n) * 2

theorem assignment_schemes :
  total_schemes = 256 ∧
  one_school_empty = 144 ∧
  two_teachers_one_school = 54 ∧
  two_schools_empty = 84 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_l1523_152302


namespace NUMINAMATH_CALUDE_aartis_work_completion_time_l1523_152303

/-- If Aarti can complete three times a piece of work in 15 days, 
    then she can complete one piece of work in 5 days. -/
theorem aartis_work_completion_time :
  ∀ (work_completion_time : ℝ),
  (3 * work_completion_time = 15) →
  work_completion_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_aartis_work_completion_time_l1523_152303


namespace NUMINAMATH_CALUDE_odd_even_subsets_equal_l1523_152332

theorem odd_even_subsets_equal (n : ℕ) :
  let S := Fin (2 * n + 1)
  (Finset.filter (fun X : Finset S => X.card % 2 = 1) (Finset.powerset (Finset.univ))).card =
  (Finset.filter (fun X : Finset S => X.card % 2 = 0) (Finset.powerset (Finset.univ))).card :=
by sorry

end NUMINAMATH_CALUDE_odd_even_subsets_equal_l1523_152332


namespace NUMINAMATH_CALUDE_system_of_equations_range_l1523_152348

theorem system_of_equations_range (x y m : ℝ) : 
  x + 2*y = 1 - m →
  2*x + y = 3 →
  x + y > 0 →
  m < 4 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_range_l1523_152348


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1523_152347

/-- Given a quadratic function f(x) = ax^2 + bx + c with a ≠ 0,
    if f(r) = f(s) = k for two distinct points r and s,
    then f(r + s) = c -/
theorem quadratic_function_property
  (a b c r s k : ℝ)
  (h_a : a ≠ 0)
  (h_distinct : r ≠ s)
  (h_fr : a * r^2 + b * r + c = k)
  (h_fs : a * s^2 + b * s + c = k) :
  a * (r + s)^2 + b * (r + s) + c = c :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1523_152347
