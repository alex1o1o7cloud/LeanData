import Mathlib

namespace NUMINAMATH_CALUDE_average_sale_proof_l2464_246414

def sales : List ℕ := [6335, 6927, 6855, 7230, 6562]
def required_sale : ℕ := 5091
def num_months : ℕ := 6

theorem average_sale_proof :
  (sales.sum + required_sale) / num_months = 6500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_proof_l2464_246414


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2464_246492

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(725 * m ≡ 1275 * m [ZMOD 35])) ∧ 
  (725 * n ≡ 1275 * n [ZMOD 35]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2464_246492


namespace NUMINAMATH_CALUDE_circle_center_l2464_246480

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0

-- Define what it means for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), 
    circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_center : 
  ∃ (a : ℝ), is_circle a ∧ 
  ∀ (h k : ℝ), (∀ (x y : ℝ), circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = 25) → 
  h = -2 ∧ k = -4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2464_246480


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2464_246406

/-- An isosceles triangle with side lengths m-2, 2m+1, and 8 has a perimeter of 17.5 -/
theorem isosceles_triangle_perimeter : ∀ m : ℝ,
  let a := m - 2
  let b := 2 * m + 1
  let c := 8
  (a = c ∨ b = c) → -- isosceles condition
  (a + b > c ∧ b + c > a ∧ c + a > b) → -- triangle inequality
  a + b + c = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2464_246406


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2464_246450

theorem max_value_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 24 ∧
  ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2464_246450


namespace NUMINAMATH_CALUDE_pool_volume_l2464_246404

/-- The volume of a circular pool with linearly varying depth -/
theorem pool_volume (diameter : ℝ) (min_depth max_depth : ℝ) :
  diameter = 20 →
  min_depth = 3 →
  max_depth = 6 →
  let radius := diameter / 2
  let avg_depth := (min_depth + max_depth) / 2
  let volume := π * radius^2 * avg_depth
  volume = 450 * π := by
  sorry

end NUMINAMATH_CALUDE_pool_volume_l2464_246404


namespace NUMINAMATH_CALUDE_diamond_19_98_l2464_246415

/-- Definition of the diamond operation -/
noncomputable def diamond (x y : ℝ) : ℝ := sorry

/-- Axioms for the diamond operation -/
axiom diamond_positive (x y : ℝ) (hx : x > 0) (hy : y > 0) : diamond x y > 0

axiom diamond_mul_left (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  diamond (x * y) y = x * (diamond y y)

axiom diamond_one_left (x : ℝ) (hx : x > 0) : 
  diamond (diamond x 1) x = diamond x 1

axiom diamond_one_one : diamond 1 1 = 1

/-- Theorem to be proved -/
theorem diamond_19_98 : diamond 19 98 = 19 := by sorry

end NUMINAMATH_CALUDE_diamond_19_98_l2464_246415


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l2464_246473

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

theorem heptagon_diagonals :
  num_diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l2464_246473


namespace NUMINAMATH_CALUDE_fourteenth_root_unity_l2464_246432

theorem fourteenth_root_unity : ∃ n : ℤ, 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * Real.pi / 14)) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_unity_l2464_246432


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2464_246405

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x + y = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 2006 * x - 2007
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x + y = -1003) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2464_246405


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l2464_246412

def binomial (n k : ℕ) : ℕ := sorry

theorem binomial_equation_solution :
  ∀ x : ℕ, binomial 15 (2*x+1) = binomial 15 (x+2) → x = 1 ∨ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l2464_246412


namespace NUMINAMATH_CALUDE_clock_angles_l2464_246433

/-- Represents the angle between the hour and minute hands on a clock face -/
def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

/-- Definition of a straight angle -/
def isStraightAngle (angle : ℝ) : Prop :=
  angle = 180

/-- Definition of a right angle -/
def isRightAngle (angle : ℝ) : Prop :=
  angle = 90

/-- Definition of an obtuse angle -/
def isObtuseAngle (angle : ℝ) : Prop :=
  90 < angle ∧ angle < 180

theorem clock_angles :
  (isStraightAngle (clockAngle 6 0)) ∧
  (isRightAngle (clockAngle 9 0)) ∧
  (isObtuseAngle (clockAngle 4 0)) :=
by sorry

end NUMINAMATH_CALUDE_clock_angles_l2464_246433


namespace NUMINAMATH_CALUDE_optimal_tent_purchase_l2464_246471

/-- Represents the cost and quantity of tents --/
structure TentPurchase where
  costA : ℕ  -- Cost of tent A in yuan
  costB : ℕ  -- Cost of tent B in yuan
  quantA : ℕ  -- Quantity of tent A
  quantB : ℕ  -- Quantity of tent B

/-- The given conditions for tent purchases --/
def tent_conditions (p : TentPurchase) : Prop :=
  p.costA * 2 + p.costB * 4 = 5200 ∧
  p.costA * 3 + p.costB * 1 = 2800 ∧
  p.quantA + p.quantB = 20 ∧
  p.quantA ≤ p.quantB / 3

/-- The total cost of a tent purchase --/
def total_cost (p : TentPurchase) : ℕ :=
  p.costA * p.quantA + p.costB * p.quantB

/-- The theorem to be proved --/
theorem optimal_tent_purchase (p : TentPurchase) :
  tent_conditions p →
  (∀ q : TentPurchase, tent_conditions q → total_cost p ≤ total_cost q) →
  p.costA = 600 ∧ p.costB = 1000 ∧ p.quantA = 5 ∧ p.quantB = 15 ∧ total_cost p = 18000 :=
sorry

end NUMINAMATH_CALUDE_optimal_tent_purchase_l2464_246471


namespace NUMINAMATH_CALUDE_book_student_difference_l2464_246464

/-- Proves that in 5 classrooms, where each classroom has 18 students and each student has 3 books,
    the difference between the total number of books and the total number of students is 180. -/
theorem book_student_difference :
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 18
  let books_per_student : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_books : ℕ := total_students * books_per_student
  total_books - total_students = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_book_student_difference_l2464_246464


namespace NUMINAMATH_CALUDE_quadratic_factorization_conditions_l2464_246413

theorem quadratic_factorization_conditions (b : ℤ) : 
  ¬ ∀ (m n p q : ℤ), 
    (15 : ℤ) * x^2 + b * x + 75 = (m * x + n) * (p * x + q) → 
    ∃ (r s : ℤ), (15 : ℤ) * x^2 + b * x + 75 = (m * x + n) * (p * x + q) * (r * x + s) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_conditions_l2464_246413


namespace NUMINAMATH_CALUDE_red_balls_count_l2464_246438

theorem red_balls_count (white_balls : ℕ) (ratio_white : ℕ) (ratio_red : ℕ) : 
  white_balls = 16 → ratio_white = 4 → ratio_red = 3 → 
  (white_balls * ratio_red) / ratio_white = 12 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l2464_246438


namespace NUMINAMATH_CALUDE_expression_evaluation_l2464_246490

theorem expression_evaluation (a b c : ℝ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 20) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2464_246490


namespace NUMINAMATH_CALUDE_expression_evaluation_l2464_246486

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (hsum : x + y ≠ 0) (hsum_sq : x^2 + y^2 ≠ 0) :
  (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2464_246486


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l2464_246427

theorem fgh_supermarket_difference (total : ℕ) (us : ℕ) (h1 : total = 70) (h2 : us = 42) (h3 : us < total) :
  us - (total - us) = 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l2464_246427


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l2464_246449

theorem friend_lunch_cost (total : ℕ) (difference : ℕ) (friend_cost : ℕ) : 
  total = 19 →
  difference = 3 →
  friend_cost = total / 2 + difference →
  friend_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l2464_246449


namespace NUMINAMATH_CALUDE_baker_cakes_l2464_246440

theorem baker_cakes (initial : ℕ) (sold : ℕ) (bought : ℕ) :
  initial ≥ sold →
  initial - sold + bought = initial + bought - sold :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_l2464_246440


namespace NUMINAMATH_CALUDE_least_k_for_prime_divisor_inequality_l2464_246452

/-- The number of distinct prime divisors of a positive integer -/
def w (n : ℕ+) : ℕ := sorry

/-- The statement that 5 is the least positive integer k such that 
    2^(w(n)) ≤ k * n^(1/4) for all positive integers n -/
theorem least_k_for_prime_divisor_inequality : 
  ∃ k : ℕ+, (∀ n : ℕ+, (2 : ℝ)^(w n) ≤ (k : ℝ) * n^(1/4)) ∧
  (∀ m : ℕ+, m < k → ∃ n : ℕ+, (2 : ℝ)^(w n) > (m : ℝ) * n^(1/4)) ∧
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_k_for_prime_divisor_inequality_l2464_246452


namespace NUMINAMATH_CALUDE_calvin_mistake_l2464_246456

theorem calvin_mistake (a : ℝ) : 37 + 31 * a = 37 * 31 + a ↔ a = 37 :=
  sorry

end NUMINAMATH_CALUDE_calvin_mistake_l2464_246456


namespace NUMINAMATH_CALUDE_trip_duration_l2464_246444

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the angle between hour and minute hands at a given time -/
def angleBetweenHands (t : TimeOfDay) : ℝ :=
  sorry

/-- Finds the time when clock hands are at a specific angle apart within a given hour -/
def findTimeAtAngle (hour : ℕ) (angle : ℝ) : TimeOfDay :=
  sorry

/-- Calculates the duration between two times -/
def durationBetween (t1 t2 : TimeOfDay) : ℕ × ℕ :=
  sorry

theorem trip_duration : 
  let startTime := findTimeAtAngle 7 90
  let endTime := findTimeAtAngle 15 270
  durationBetween startTime endTime = (8, 29) := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_l2464_246444


namespace NUMINAMATH_CALUDE_trajectory_equation_l2464_246436

/-- Given a fixed point A(1,2) and a moving point P(x,y), if the projection of vector OP on vector OA is -√5,
    then the equation x + 2y + 5 = 0 represents the trajectory of point P. -/
theorem trajectory_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (x, y)
  let OA : ℝ × ℝ := A
  let OP : ℝ × ℝ := P
  (OP.1 * OA.1 + OP.2 * OA.2) / Real.sqrt (OA.1^2 + OA.2^2) = -Real.sqrt 5 →
  x + 2*y + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2464_246436


namespace NUMINAMATH_CALUDE_min_distance_log_circle_l2464_246425

theorem min_distance_log_circle (e : ℝ) (h : e > 0) :
  let f := fun x : ℝ => Real.log x
  let circle := fun (x y : ℝ) => (x - (e + 1/e))^2 + y^2 = 1/4
  ∃ (min_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), f x₁ = y₁ → circle x₂ y₂ →
      min_dist ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
    min_dist = (2 * Real.sqrt (e^2 + 1) - e) / (2 * e) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_log_circle_l2464_246425


namespace NUMINAMATH_CALUDE_age_difference_daughter_daughterInLaw_l2464_246442

/-- Represents the ages of family members 5 years ago -/
structure FamilyAges5YearsAgo where
  member1 : ℕ
  member2 : ℕ
  member3 : ℕ
  daughter : ℕ

/-- Represents the current ages of family members -/
structure CurrentFamilyAges where
  member1 : ℕ
  member2 : ℕ
  member3 : ℕ
  daughterInLaw : ℕ

/-- The main theorem stating the difference in ages between daughter and daughter-in-law -/
theorem age_difference_daughter_daughterInLaw 
  (ages5YearsAgo : FamilyAges5YearsAgo)
  (currentAges : CurrentFamilyAges)
  (h1 : ages5YearsAgo.member1 + ages5YearsAgo.member2 + ages5YearsAgo.member3 + ages5YearsAgo.daughter = 114)
  (h2 : currentAges.member1 + currentAges.member2 + currentAges.member3 + currentAges.daughterInLaw = 85)
  (h3 : currentAges.member1 = ages5YearsAgo.member1 + 5)
  (h4 : currentAges.member2 = ages5YearsAgo.member2 + 5)
  (h5 : currentAges.member3 = ages5YearsAgo.member3 + 5) :
  ages5YearsAgo.daughter - currentAges.daughterInLaw = 29 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_daughter_daughterInLaw_l2464_246442


namespace NUMINAMATH_CALUDE_added_to_broken_ratio_l2464_246472

/-- Represents the number of turtle statues on Grandma Molly's lawn over four years -/
def TurtleStatues : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => 4 * 4  -- Quadrupled in the second year
| 2 => 4 * 4 + 12 - 3  -- Added 12, broke 3 in the third year
| 3 => 31  -- Total at the end of the fourth year
| _ => 31  -- Assume it stays constant after the fourth year

/-- The number of statues broken in the third year -/
def BrokenStatues : ℕ := 3

/-- Theorem stating the ratio of added statues in the fourth year to broken statues in the third year -/
theorem added_to_broken_ratio :
  (TurtleStatues 3 - TurtleStatues 2) / BrokenStatues = 2 := by
  sorry


end NUMINAMATH_CALUDE_added_to_broken_ratio_l2464_246472


namespace NUMINAMATH_CALUDE_correct_regression_equation_l2464_246417

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if a linear regression equation passes through a given point -/
def passes_through (eq : LinearRegression) (x y : ℝ) : Prop :=
  eq.slope * x + eq.intercept = y

/-- Represents the properties of the given data -/
structure DataProperties where
  x_mean : ℝ
  y_mean : ℝ
  positively_correlated : Prop

theorem correct_regression_equation 
  (data : DataProperties)
  (h_x_mean : data.x_mean = 2.4)
  (h_y_mean : data.y_mean = 3.2)
  (h_corr : data.positively_correlated) :
  ∃ (eq : LinearRegression), 
    eq.slope = 0.5 ∧ 
    eq.intercept = 2 ∧ 
    passes_through eq data.x_mean data.y_mean :=
sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l2464_246417


namespace NUMINAMATH_CALUDE_complex_roots_condition_l2464_246498

theorem complex_roots_condition (p : ℝ) :
  (∀ x : ℝ, x^2 + p*x + 1 ≠ 0) →
  p < 2 ∧
  ¬(p < 2 → ∀ x : ℝ, x^2 + p*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_condition_l2464_246498


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2464_246482

/-- An isosceles triangle with congruent sides of 5 cm and perimeter of 17 cm has a base of 7 cm. -/
theorem isosceles_triangle_base_length : ∀ (base : ℝ),
  base > 0 →
  5 + 5 + base = 17 →
  base = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2464_246482


namespace NUMINAMATH_CALUDE_sets_equal_iff_m_eq_neg_two_sqrt_two_l2464_246468

def A (m : ℝ) : Set ℝ := {x | x^2 + m*x + 2 ≥ 0 ∧ x ≥ 0}

def B (m : ℝ) : Set ℝ := {y | ∃ x ∈ A m, y = Real.sqrt (x^2 + m*x + 2)}

theorem sets_equal_iff_m_eq_neg_two_sqrt_two (m : ℝ) :
  A m = B m ↔ m = -2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sets_equal_iff_m_eq_neg_two_sqrt_two_l2464_246468


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2464_246408

theorem simplify_and_evaluate (m : ℝ) (h : m = 4 * Real.sqrt 3) :
  (1 - m / (m - 3)) / ((m^2 - 3*m) / (m^2 - 6*m + 9)) = -(Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2464_246408


namespace NUMINAMATH_CALUDE_quadratic_max_sum_roots_l2464_246459

theorem quadratic_max_sum_roots (m : ℝ) :
  let f := fun x : ℝ => 2 * x^2 - 5 * x + m
  let Δ := 25 - 8 * m  -- discriminant
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0) →  -- real roots exist
  (∀ k : ℝ, (∃ y₁ y₂ : ℝ, f y₁ = 0 ∧ f y₂ = 0) → y₁ + y₂ ≤ 5/2) ∧  -- 5/2 is max sum
  (m = 25/8 → ∃ z₁ z₂ : ℝ, f z₁ = 0 ∧ f z₂ = 0 ∧ z₁ + z₂ = 5/2)  -- max sum occurs at m = 25/8
  :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_sum_roots_l2464_246459


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2464_246409

/-- Given a line segment with midpoint (3, 4) and one endpoint (-2, -5), 
    the other endpoint is (8, 13) -/
theorem other_endpoint_of_line_segment 
  (midpoint : ℝ × ℝ) 
  (endpoint1 : ℝ × ℝ) 
  (endpoint2 : ℝ × ℝ) : 
  midpoint = (3, 4) → 
  endpoint1 = (-2, -5) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) → 
  endpoint2 = (8, 13) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2464_246409


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2464_246401

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_prod1 : a 1 * a 2 * a 3 = 5)
  (h_prod2 : a 4 * a 8 * a 9 = 10) :
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2464_246401


namespace NUMINAMATH_CALUDE_mikes_painting_area_l2464_246489

/-- The area Mike needs to paint on the wall -/
def area_to_paint (wall_height wall_length window_height window_length painting_side : ℝ) : ℝ :=
  wall_height * wall_length - (window_height * window_length + painting_side * painting_side)

/-- Theorem stating the area Mike needs to paint -/
theorem mikes_painting_area :
  area_to_paint 10 15 3 5 2 = 131 := by
  sorry

end NUMINAMATH_CALUDE_mikes_painting_area_l2464_246489


namespace NUMINAMATH_CALUDE_increasing_f_implies_t_ge_5_l2464_246475

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

/-- The function f(x) defined as the dot product of (x^2, x+1) and (1-x, t) -/
def f (t : ℝ) (x : ℝ) : ℝ := dot_product (x^2, x+1) (1-x, t)

/-- A function is increasing on an interval if for any two points in the interval, 
    the function value at the larger point is greater than at the smaller point -/
def is_increasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → g x < g y

theorem increasing_f_implies_t_ge_5 :
  ∀ t : ℝ, is_increasing (f t) (-1) 1 → t ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_t_ge_5_l2464_246475


namespace NUMINAMATH_CALUDE_correct_distribution_l2464_246447

/-- Represents the jellybean distribution problem --/
structure JellybeanDistribution where
  total_jellybeans : ℕ
  num_nephews : ℕ
  num_nieces : ℕ
  nephew_ratio : ℕ
  niece_ratio : ℕ

/-- Calculates the maximum number of jellybeans each nephew and niece can receive --/
def max_distribution (jd : JellybeanDistribution) : ℕ × ℕ :=
  let total_parts := jd.num_nephews * jd.nephew_ratio + jd.num_nieces * jd.niece_ratio
  let jellybeans_per_part := jd.total_jellybeans / total_parts
  (jellybeans_per_part * jd.nephew_ratio, jellybeans_per_part * jd.niece_ratio)

/-- Theorem stating the correct distribution for the given problem --/
theorem correct_distribution (jd : JellybeanDistribution) 
  (h1 : jd.total_jellybeans = 537)
  (h2 : jd.num_nephews = 4)
  (h3 : jd.num_nieces = 3)
  (h4 : jd.nephew_ratio = 2)
  (h5 : jd.niece_ratio = 1) :
  max_distribution jd = (96, 48) ∧ 
  96 * jd.num_nephews + 48 * jd.num_nieces ≤ jd.total_jellybeans :=
by
  sorry

#eval max_distribution {
  total_jellybeans := 537,
  num_nephews := 4,
  num_nieces := 3,
  nephew_ratio := 2,
  niece_ratio := 1
}

end NUMINAMATH_CALUDE_correct_distribution_l2464_246447


namespace NUMINAMATH_CALUDE_subtraction_with_division_l2464_246422

theorem subtraction_with_division : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l2464_246422


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2464_246454

theorem min_distance_to_line (x y : ℝ) :
  8 * x + 15 * y = 120 →
  x ≥ 0 →
  ∃ (min : ℝ), min = 120 / 17 ∧ ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 → x' ≥ 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2464_246454


namespace NUMINAMATH_CALUDE_remainder_of_s_1012_l2464_246466

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (x^1012 - 1) / (x - 1)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^3 + x^2 + x + 1

-- Define s(x) as the polynomial remainder
noncomputable def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_of_s_1012 : |s 1012| % 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_s_1012_l2464_246466


namespace NUMINAMATH_CALUDE_cannot_form_triangle_5_6_11_l2464_246460

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem cannot_form_triangle_5_6_11 :
  ¬ can_form_triangle 5 6 11 := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_5_6_11_l2464_246460


namespace NUMINAMATH_CALUDE_toll_booth_traffic_l2464_246495

theorem toll_booth_traffic (total : ℕ) (mon : ℕ) (tues : ℕ) (wed : ℕ) (thur : ℕ) :
  total = 450 →
  mon = 50 →
  tues = mon →
  wed = 2 * mon →
  thur = wed →
  ∃ (remaining : ℕ), 
    remaining * 3 = total - (mon + tues + wed + thur) ∧
    remaining = 50 :=
by sorry

end NUMINAMATH_CALUDE_toll_booth_traffic_l2464_246495


namespace NUMINAMATH_CALUDE_robbie_win_probability_l2464_246421

/-- A special six-sided die where rolling number x is x times as likely as rolling a 1 -/
structure SpecialDie :=
  (prob_one : ℝ)
  (sum_to_one : prob_one * (1 + 2 + 3 + 4 + 5 + 6) = 1)

/-- The game where two players roll the special die three times each -/
def Game (d : SpecialDie) :=
  { score : ℕ × ℕ // score.1 ≤ 18 ∧ score.2 ≤ 18 }

/-- The probability of rolling a specific number on the special die -/
def prob_roll (d : SpecialDie) (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 6 then n * d.prob_one else 0

/-- The probability of Robbie winning given the current game state -/
def prob_robbie_win (d : SpecialDie) (g : Game d) : ℝ :=
  sorry

theorem robbie_win_probability (d : SpecialDie) (g : Game d) 
  (h1 : g.val.1 = 8) (h2 : g.val.2 = 10) : 
  prob_robbie_win d g = 55 / 441 :=
sorry

end NUMINAMATH_CALUDE_robbie_win_probability_l2464_246421


namespace NUMINAMATH_CALUDE_third_number_in_multiplication_l2464_246426

theorem third_number_in_multiplication (p n : ℕ) : 
  (p = 125 * 243 * n / 405) → 
  (1000 ≤ p) → 
  (p < 10000) → 
  (∀ m : ℕ, m < n → 125 * 243 * m / 405 < 1000) →
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_third_number_in_multiplication_l2464_246426


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2464_246497

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > (1/2 : ℝ) → 2*x^2 + x - 1 > 0) ∧
  ¬(2*x^2 + x - 1 > 0 → x > (1/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2464_246497


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l2464_246403

theorem subtraction_of_negatives : -5 - (-2) = -3 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l2464_246403


namespace NUMINAMATH_CALUDE_mean_temperature_is_zero_l2464_246419

def temperatures : List ℤ := [-3, -1, -6, 0, 4, 6]

theorem mean_temperature_is_zero : 
  (temperatures.sum : ℚ) / temperatures.length = 0 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_zero_l2464_246419


namespace NUMINAMATH_CALUDE_locus_equals_homothety_image_l2464_246461

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- Represents a rotational homothety transformation -/
structure RotationalHomothety where
  center : Point
  angle : ℝ
  factor : ℝ

/-- The locus of points Y for a given semicircle and constant k -/
def locusOfY (s : Semicircle) (k : ℝ) : Set Point :=
  sorry

/-- The image of a semicircle under rotational homothety -/
def imageUnderHomothety (s : Semicircle) (h : RotationalHomothety) : Set Point :=
  sorry

/-- Main theorem: The locus of Y is the image of the semicircle under rotational homothety -/
theorem locus_equals_homothety_image (s : Semicircle) (k : ℝ) (h0 : k > 0) :
  locusOfY s k = imageUnderHomothety s ⟨s.center, Real.arctan k, Real.sqrt (k^2 + 1)⟩ :=
sorry

end NUMINAMATH_CALUDE_locus_equals_homothety_image_l2464_246461


namespace NUMINAMATH_CALUDE_solve_for_a_l2464_246441

theorem solve_for_a (a b : ℚ) (h1 : b/a = 4) (h2 : b = 20 - 3*a) : a = 20/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2464_246441


namespace NUMINAMATH_CALUDE_price_per_square_foot_l2464_246462

def house_area : ℝ := 2400
def barn_area : ℝ := 1000
def total_property_value : ℝ := 333200

theorem price_per_square_foot :
  total_property_value / (house_area + barn_area) = 98 := by
sorry

end NUMINAMATH_CALUDE_price_per_square_foot_l2464_246462


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l2464_246458

def number_of_friends : ℕ := 7

/-- The number of ways to choose 2 people from n friends to sit next to Cara in a circular arrangement -/
def circular_seating_arrangements (n : ℕ) : ℕ := Nat.choose n 2

theorem cara_seating_arrangements :
  circular_seating_arrangements number_of_friends = 21 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l2464_246458


namespace NUMINAMATH_CALUDE_probability_more_ones_than_sixes_l2464_246478

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the outcome of rolling five dice -/
def FiveDiceRoll := Vector DieOutcome 5

/-- The total number of possible outcomes when rolling five fair six-sided dice -/
def totalOutcomes : Nat := 7776

/-- The number of outcomes where there are more 1's than 6's -/
def favorableOutcomes : Nat := 2676

/-- The probability of rolling more 1's than 6's when rolling five fair six-sided dice -/
def probabilityMoreOnesThanSixes : Rat := favorableOutcomes / totalOutcomes

theorem probability_more_ones_than_sixes :
  probabilityMoreOnesThanSixes = 2676 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_probability_more_ones_than_sixes_l2464_246478


namespace NUMINAMATH_CALUDE_slope_of_line_l2464_246481

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l2464_246481


namespace NUMINAMATH_CALUDE_triangle_max_area_l2464_246411

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    prove that the maximum area is 9√7 when a = 6 and √7 * b * cos(A) = 3 * a * sin(B) -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 6 →
  Real.sqrt 7 * b * Real.cos A = 3 * a * Real.sin B →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ S ≤ 9 * Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2464_246411


namespace NUMINAMATH_CALUDE_even_mono_increasing_negative_l2464_246463

-- Define an even function that is monotonically increasing on [0, +∞)
def EvenMonoIncreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x ≤ f y)

-- Theorem statement
theorem even_mono_increasing_negative (f : ℝ → ℝ) (a b : ℝ) 
  (hf : EvenMonoIncreasing f) (hab : a < b) (hneg : b < 0) : 
  f a > f b := by
  sorry

end NUMINAMATH_CALUDE_even_mono_increasing_negative_l2464_246463


namespace NUMINAMATH_CALUDE_basketball_points_l2464_246439

theorem basketball_points (T : ℕ) : 
  T + (T + 6) + (2 * T + 4) = 26 → T = 4 := by sorry

end NUMINAMATH_CALUDE_basketball_points_l2464_246439


namespace NUMINAMATH_CALUDE_divisible_by_120_l2464_246429

theorem divisible_by_120 (m : ℕ) : ∃ k : ℤ, (m ^ 5 : ℤ) - 5 * (m ^ 3) + 4 * m = 120 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l2464_246429


namespace NUMINAMATH_CALUDE_polygon_sides_l2464_246479

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : sum_angles = 900 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2464_246479


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_false_l2464_246455

theorem baron_munchausen_claim_false : 
  ¬ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ∃ m : ℕ, 0 ≤ m ∧ m ≤ 99 ∧ ∃ k : ℕ, (n * 100 + m) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_false_l2464_246455


namespace NUMINAMATH_CALUDE_side_angle_relation_l2464_246453

theorem side_angle_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  (a > b ↔ Real.sin A > Real.sin B) :=
by sorry

end NUMINAMATH_CALUDE_side_angle_relation_l2464_246453


namespace NUMINAMATH_CALUDE_range_of_a_l2464_246410

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 - x - a - 2
def g (a x : ℝ) : ℝ := x^2 - (a+1)*x - 2

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : f a x₁ = 0)
  (h₂ : f a x₂ = 0)
  (h₃ : g a x₃ = 0)
  (h₄ : g a x₄ = 0)
  (h₅ : x₃ < x₁ ∧ x₁ < x₄ ∧ x₄ < x₂) :
  -2 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2464_246410


namespace NUMINAMATH_CALUDE_smallest_k_and_exponent_l2464_246476

theorem smallest_k_and_exponent (k : ℕ) (h : k = 7) :
  64^k > 4^20 ∧ 64^k ≤ 4^21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_and_exponent_l2464_246476


namespace NUMINAMATH_CALUDE_crayons_per_friend_l2464_246467

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (crayons_per_friend : ℕ) : 
  total_crayons = 210 → num_friends = 30 → crayons_per_friend = total_crayons / num_friends →
  crayons_per_friend = 7 := by
sorry

end NUMINAMATH_CALUDE_crayons_per_friend_l2464_246467


namespace NUMINAMATH_CALUDE_coat_shirt_ratio_l2464_246474

theorem coat_shirt_ratio (pants shirt coat : ℕ) : 
  pants + shirt = 100 →
  pants + coat = 244 →
  coat = 180 →
  ∃ (k : ℕ), coat = k * shirt →
  coat / shirt = 5 := by
sorry

end NUMINAMATH_CALUDE_coat_shirt_ratio_l2464_246474


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_12_l2464_246470

theorem circle_area_with_diameter_12 (π : Real) (diameter : Real) (area : Real) :
  diameter = 12 →
  area = π * (diameter / 2)^2 →
  area = π * 36 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_12_l2464_246470


namespace NUMINAMATH_CALUDE_distance_home_to_school_l2464_246424

theorem distance_home_to_school :
  ∀ (d : ℝ) (t : ℝ),
    d = 6 * (t + 7/60) ∧
    d = 12 * (t - 8/60) →
    d = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l2464_246424


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2464_246485

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2464_246485


namespace NUMINAMATH_CALUDE_rectangle_similarity_l2464_246400

theorem rectangle_similarity (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  let r := (y, x)
  let r' := (y - x, x)
  let r'' := if y - x < x then ((y - x), (2 * x - y)) else (x, (y - 2 * x))
  ¬ (r'.1 / r'.2 = r.1 / r.2) →
  (r''.1 / r''.2 = r.1 / r.2) →
  y / x = 1 + Real.sqrt 2 ∨ y / x = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_similarity_l2464_246400


namespace NUMINAMATH_CALUDE_equation_solution_l2464_246496

theorem equation_solution (x : ℝ) : 
  x ≠ 1 →
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1)) ↔ (x = 6 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2464_246496


namespace NUMINAMATH_CALUDE_f_composition_value_l2464_246483

def f (x : ℚ) : ℚ := x⁻¹ + x⁻¹ / (1 + x⁻¹)

theorem f_composition_value : f (f (-3)) = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2464_246483


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2464_246465

structure GeometricSpace where
  Line : Type
  Plane : Type
  perpendicular : Line → Line → Prop
  parallel : Line → Plane → Prop
  perpendicular_plane : Line → Plane → Prop

variable (S : GeometricSpace)

def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem perpendicular_necessary_not_sufficient
  (l m : S.Line) (α : S.Plane)
  (h1 : l ≠ m)
  (h2 : S.perpendicular_plane m α) :
  necessary_not_sufficient (S.perpendicular l m) (S.parallel l α) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2464_246465


namespace NUMINAMATH_CALUDE_area_between_circles_l2464_246423

-- Define the circles
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the centers of the circles
def center_X : ℝ × ℝ := (0, 0)
def center_Y : ℝ × ℝ := (2, 0)
def center_Z : ℝ × ℝ := (0, 2)

-- Define the circles
def X := Circle center_X 1
def Y := Circle center_Y 1
def Z := Circle center_Z 1

-- Define the area function
def area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_between_circles :
  (∀ p ∈ X ∩ Y, p = (1, 0)) →  -- X and Y are tangent
  (∃ p, p ∈ X ∩ Z ∧ p ≠ center_X) →  -- Z is tangent to X
  (∀ p, p ∉ Y ∩ Z) →  -- Z does not intersect Y
  area (Z \ X) = π / 2 := by sorry

end NUMINAMATH_CALUDE_area_between_circles_l2464_246423


namespace NUMINAMATH_CALUDE_equation_solutions_l2464_246434

theorem equation_solutions (p : ℕ) (h_prime : Nat.Prime p) :
  ∃! (solutions : Finset (ℕ × ℕ)), 
    solutions.card = 3 ∧
    ∀ (x y : ℕ), (x, y) ∈ solutions ↔ 
      (x > 0 ∧ y > 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / p) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2464_246434


namespace NUMINAMATH_CALUDE_annular_area_l2464_246491

/-- The area of an annular region formed by two concentric circles -/
theorem annular_area (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 10) :
  π * r₂^2 - π * r₁^2 = 84 * π := by
  sorry

end NUMINAMATH_CALUDE_annular_area_l2464_246491


namespace NUMINAMATH_CALUDE_sara_balloons_l2464_246428

/-- The number of red balloons Sara has left after giving some away -/
def balloons_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sara is left with 7 red balloons -/
theorem sara_balloons : balloons_left 31 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sara_balloons_l2464_246428


namespace NUMINAMATH_CALUDE_soybean_oil_production_l2464_246443

/-- Represents the conversion rates and prices for soybeans, tofu, and soybean oil -/
structure SoybeanProduction where
  soybean_to_tofu : ℝ        -- kg of tofu per kg of soybeans
  soybean_to_oil : ℝ         -- kg of soybeans needed for 1 kg of oil
  tofu_price : ℝ             -- yuan per kg of tofu
  oil_price : ℝ              -- yuan per kg of oil

/-- Represents the batch of soybeans and its processing -/
structure SoybeanBatch where
  total_soybeans : ℝ         -- total kg of soybeans in the batch
  tofu_soybeans : ℝ          -- kg of soybeans used for tofu
  oil_soybeans : ℝ           -- kg of soybeans used for oil
  total_revenue : ℝ          -- total revenue in yuan

/-- Theorem stating that given the conditions, 360 kg of soybeans were used for oil production -/
theorem soybean_oil_production (prod : SoybeanProduction) (batch : SoybeanBatch) :
  prod.soybean_to_tofu = 3 ∧
  prod.soybean_to_oil = 6 ∧
  prod.tofu_price = 3 ∧
  prod.oil_price = 15 ∧
  batch.total_soybeans = 460 ∧
  batch.total_revenue = 1800 ∧
  batch.tofu_soybeans + batch.oil_soybeans = batch.total_soybeans ∧
  batch.total_revenue = (batch.tofu_soybeans * prod.soybean_to_tofu * prod.tofu_price) +
                        (batch.oil_soybeans / prod.soybean_to_oil * prod.oil_price) →
  batch.oil_soybeans = 360 := by
  sorry

end NUMINAMATH_CALUDE_soybean_oil_production_l2464_246443


namespace NUMINAMATH_CALUDE_cone_radii_sum_l2464_246477

/-- Given a circle with radius 5 divided into three sectors with area ratios 1:2:3,
    when these sectors are used as lateral surfaces of three cones,
    the sum of the base radii of these cones equals 5. -/
theorem cone_radii_sum (r₁ r₂ r₃ : ℝ) : r₁ + r₂ + r₃ = 5 :=
  sorry

end NUMINAMATH_CALUDE_cone_radii_sum_l2464_246477


namespace NUMINAMATH_CALUDE_min_value_theorem_l2464_246431

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 1 → 1 / (2 * x) + 2 / y ≥ 1 / (2 * a) + 2 / b) ∧
  1 / (2 * a) + 2 / b = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2464_246431


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2464_246430

def divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem divisibility_equivalence :
  (∀ n : ℕ, divisible n 6 → divisible n 3) ↔
  (∀ n : ℕ, ¬(divisible n 3) → ¬(divisible n 6)) ∧
  (∀ n : ℕ, ¬(divisible n 6) ∨ divisible n 3) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2464_246430


namespace NUMINAMATH_CALUDE_number_of_spiders_l2464_246488

def total_legs : ℕ := 136
def num_ants : ℕ := 12
def spider_legs : ℕ := 8
def ant_legs : ℕ := 6

theorem number_of_spiders :
  ∃ (num_spiders : ℕ), 
    num_spiders * spider_legs + num_ants * ant_legs = total_legs ∧ 
    num_spiders = 8 :=
by sorry

end NUMINAMATH_CALUDE_number_of_spiders_l2464_246488


namespace NUMINAMATH_CALUDE_vegetable_growth_rate_equation_l2464_246494

theorem vegetable_growth_rate_equation 
  (initial_production final_production : ℝ) 
  (growth_years : ℕ) 
  (x : ℝ) 
  (h1 : initial_production = 800)
  (h2 : final_production = 968)
  (h3 : growth_years = 2)
  (h4 : final_production = initial_production * (1 + x) ^ growth_years) :
  800 * (1 + x)^2 = 968 := by
sorry

end NUMINAMATH_CALUDE_vegetable_growth_rate_equation_l2464_246494


namespace NUMINAMATH_CALUDE_fish_filets_count_l2464_246418

/-- The number of fish filets Ben and his family will have -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let thrown_back := 3
  let filets_per_fish := 2
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_caught - thrown_back
  fish_kept * filets_per_fish

theorem fish_filets_count : fish_filets = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_filets_count_l2464_246418


namespace NUMINAMATH_CALUDE_system_coefficients_proof_l2464_246445

theorem system_coefficients_proof : ∃! (a b c : ℝ),
  (∀ x y : ℝ, a * (x - 1) + 2 * y ≠ 1 ∨ b * (x - 1) + c * y ≠ 3) ∧
  (a * (-1/4) + 2 * (5/8) = 1) ∧
  (b * (1/4) + c * (5/8) = 3) ∧
  a = 1 ∧ b = 2 ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_coefficients_proof_l2464_246445


namespace NUMINAMATH_CALUDE_inequality_proof_l2464_246416

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) : 
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2464_246416


namespace NUMINAMATH_CALUDE_total_ages_is_56_l2464_246437

/-- Given Craig's age and the age difference with his mother, calculate the total of their ages -/
def total_ages (craig_age : ℕ) (age_difference : ℕ) : ℕ :=
  craig_age + (craig_age + age_difference)

/-- Theorem: The total of Craig and his mother's ages is 56 years -/
theorem total_ages_is_56 : total_ages 16 24 = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_ages_is_56_l2464_246437


namespace NUMINAMATH_CALUDE_jessica_remaining_money_l2464_246457

/-- Given Jessica's initial amount and spending, calculate the remaining amount -/
theorem jessica_remaining_money (initial : ℚ) (spent : ℚ) (remaining : ℚ) : 
  initial = 11.73 ∧ spent = 10.22 ∧ remaining = initial - spent → remaining = 1.51 := by
  sorry

end NUMINAMATH_CALUDE_jessica_remaining_money_l2464_246457


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2464_246448

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2464_246448


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2464_246446

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (x - 2)^6 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2464_246446


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2464_246407

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, mx - n > 0 ↔ x < 1/3) → 
  (∀ x, (m + n) * x < n - m ↔ x > -1/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2464_246407


namespace NUMINAMATH_CALUDE_anhui_imports_exports_2012_l2464_246435

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem anhui_imports_exports_2012 :
  toScientificNotation (39.33 * 10^9) = ScientificNotation.mk 3.933 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_anhui_imports_exports_2012_l2464_246435


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2464_246484

theorem sum_of_distinct_prime_factors : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → p ∣ (7^7 - 7^4) ↔ p ∈ s) ∧
  (s.sum id = 24) := by
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2464_246484


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2464_246469

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 4 + a 5 = 3) 
  (h_a8 : a 8 = 8) : 
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2464_246469


namespace NUMINAMATH_CALUDE_sqrt_equation_l2464_246402

theorem sqrt_equation (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l2464_246402


namespace NUMINAMATH_CALUDE_girls_from_clay_middle_school_l2464_246493

theorem girls_from_clay_middle_school
  (total_students : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (jonas_students : ℕ)
  (clay_students : ℕ)
  (hart_students : ℕ)
  (jonas_boys : ℕ)
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : jonas_students = 50)
  (h5 : clay_students = 70)
  (h6 : hart_students = 30)
  (h7 : jonas_boys = 25)
  (h8 : total_students = total_boys + total_girls)
  (h9 : total_students = jonas_students + clay_students + hart_students)
  : ∃ clay_girls : ℕ, clay_girls = 30 ∧ clay_girls ≤ clay_students :=
by sorry

end NUMINAMATH_CALUDE_girls_from_clay_middle_school_l2464_246493


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l2464_246420

theorem fraction_multiplication_addition : (1 / 3 : ℚ) * (3 / 4 : ℚ) * (1 / 5 : ℚ) + (1 / 6 : ℚ) = 13 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l2464_246420


namespace NUMINAMATH_CALUDE_power_of_square_l2464_246499

theorem power_of_square (b : ℝ) : (b^2)^3 = b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l2464_246499


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l2464_246487

-- Define the inequality function
def f (k x : ℝ) : ℝ := k * x^2 - 2 * abs (x - 1) + 3 * k

-- Define the property of having an empty solution set
def has_empty_solution_set (k : ℝ) : Prop :=
  ∀ x, f k x ≥ 0

-- State the theorem
theorem empty_solution_set_iff_k_ge_one :
  ∀ k, has_empty_solution_set k ↔ k ≥ 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l2464_246487


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_almost_squares_l2464_246451

/-- A natural number is almost a square if it can be represented as a product of two numbers
    that differ by no more than one percent of the larger of them. -/
def AlmostSquare (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * b ∧ (a : ℝ) ≥ (b : ℝ) ∧ (a : ℝ) ≤ (b : ℝ) * 1.01

/-- There exist infinitely many natural numbers m such that 4m^4 - 1, 4m^4, 4m^4 + 1, and 4m^4 + 2
    are all almost squares. -/
theorem infinitely_many_consecutive_almost_squares :
  ∀ N : ℕ, ∃ m : ℕ, m > N ∧
    AlmostSquare (4 * m^4 - 1) ∧
    AlmostSquare (4 * m^4) ∧
    AlmostSquare (4 * m^4 + 1) ∧
    AlmostSquare (4 * m^4 + 2) := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_consecutive_almost_squares_l2464_246451
