import Mathlib

namespace NUMINAMATH_CALUDE_binomial_max_probability_l3608_360808

/-- The number of trials in the binomial distribution -/
def n : ℕ := 10

/-- The probability of success in each trial -/
def p : ℝ := 0.8

/-- The probability mass function of the binomial distribution -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The value of k that maximizes the binomial PMF -/
def kMax : ℕ := 8

theorem binomial_max_probability :
  ∀ k : ℕ, k ≠ kMax → binomialPMF k ≤ binomialPMF kMax :=
sorry

end NUMINAMATH_CALUDE_binomial_max_probability_l3608_360808


namespace NUMINAMATH_CALUDE_karthiks_weight_upper_bound_l3608_360853

-- Define the variables
def lower_bound : ℝ := 55
def upper_bound : ℝ := 58
def average_weight : ℝ := 56.5

-- Define the theorem
theorem karthiks_weight_upper_bound (X : ℝ) 
  (h1 : X > 50)  -- Karthik's brother's lower bound
  (h2 : X ≤ 62)  -- Karthik's upper bound
  (h3 : X ≤ 58)  -- Karthik's father's upper bound
  (h4 : (lower_bound + X) / 2 = average_weight)  -- Average condition
  : X = upper_bound := by
  sorry

end NUMINAMATH_CALUDE_karthiks_weight_upper_bound_l3608_360853


namespace NUMINAMATH_CALUDE_movie_theater_ticket_sales_l3608_360883

/-- Theorem: Movie Theater Ticket Sales
Given the prices and quantities of different types of movie tickets,
prove that the number of evening tickets sold is 300. -/
theorem movie_theater_ticket_sales
  (matinee_price : ℕ) (evening_price : ℕ) (threeD_price : ℕ)
  (matinee_quantity : ℕ) (threeD_quantity : ℕ)
  (total_revenue : ℕ) :
  matinee_price = 5 →
  evening_price = 12 →
  threeD_price = 20 →
  matinee_quantity = 200 →
  threeD_quantity = 100 →
  total_revenue = 6600 →
  ∃ evening_quantity : ℕ,
    evening_quantity = 300 ∧
    total_revenue = matinee_price * matinee_quantity +
                    evening_price * evening_quantity +
                    threeD_price * threeD_quantity :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_ticket_sales_l3608_360883


namespace NUMINAMATH_CALUDE_stationery_box_cost_l3608_360865

/-- The cost of a single stationery box in yuan -/
def unit_price : ℕ := 23

/-- The number of stationery boxes to be purchased -/
def quantity : ℕ := 3

/-- The total cost of purchasing the stationery boxes -/
def total_cost : ℕ := unit_price * quantity

theorem stationery_box_cost : total_cost = 69 := by
  sorry

end NUMINAMATH_CALUDE_stationery_box_cost_l3608_360865


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l3608_360820

theorem cubic_root_equation_solution :
  ∃ x : ℝ, (30 * x + (30 * x + 15) ^ (1/3)) ^ (1/3) = 15 ∧ x = 112 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l3608_360820


namespace NUMINAMATH_CALUDE_work_completion_time_l3608_360834

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℚ
  hours : ℚ

/-- Represents a worker's capacity to complete work -/
structure Worker where
  completion_time : ℚ

/-- Represents a work period with multiple workers -/
structure WorkPeriod where
  duration : ℚ
  workers : List Worker

/-- Calculates the fraction of work completed in a day by a worker -/
def Worker.daily_work (w : Worker) : ℚ :=
  1 / w.completion_time

/-- Calculates the total work completed in a period -/
def WorkPeriod.work_completed (wp : WorkPeriod) : ℚ :=
  wp.duration * (wp.workers.map Worker.daily_work).sum

/-- Converts days to a WorkTime structure -/
def days_to_work_time (d : ℚ) : WorkTime :=
  ⟨d.floor, (d - d.floor) * 24⟩

theorem work_completion_time 
  (worker_a worker_b worker_c : Worker)
  (period1 period2 period3 : WorkPeriod)
  (h1 : worker_a.completion_time = 15)
  (h2 : worker_b.completion_time = 10)
  (h3 : worker_c.completion_time = 12)
  (h4 : period1 = ⟨2, [worker_a, worker_b, worker_c]⟩)
  (h5 : period2 = ⟨3, [worker_a, worker_c]⟩)
  (h6 : period3 = ⟨(1 - period1.work_completed - period2.work_completed) / worker_a.daily_work, [worker_a]⟩) :
  days_to_work_time (period1.duration + period2.duration + period3.duration) = ⟨5, 18⟩ :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3608_360834


namespace NUMINAMATH_CALUDE_paper_folding_volumes_l3608_360826

/-- Given a square paper with side length 1, prove the volume of a cone and max volume of a rectangular prism --/
theorem paper_folding_volumes (ε : ℝ) (hε : ε = 0.0001) :
  ∃ (V_cone V_prism : ℝ),
    (abs (V_cone - (π / 6)) < ε) ∧
    (abs (V_prism - (1 / (3 * Real.sqrt 3))) < ε) ∧
    (∀ (a b c : ℝ), 2 * (a * b + b * c + c * a) = 1 → a * b * c ≤ V_prism) := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_volumes_l3608_360826


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l3608_360864

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l3608_360864


namespace NUMINAMATH_CALUDE_power_of_81_four_thirds_l3608_360899

theorem power_of_81_four_thirds :
  (81 : ℝ) ^ (4/3) = 243 * (3 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_power_of_81_four_thirds_l3608_360899


namespace NUMINAMATH_CALUDE_log2_derivative_l3608_360875

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log2_derivative_l3608_360875


namespace NUMINAMATH_CALUDE_train_encounters_l3608_360823

/-- Represents the number of hours in the journey -/
def journey_duration : ℕ := 5

/-- Represents the number of trains already on the route when the journey begins -/
def initial_trains : ℕ := 4

/-- Calculates the number of trains encountered during the journey -/
def trains_encountered (duration : ℕ) (initial : ℕ) : ℕ :=
  initial + duration

theorem train_encounters :
  trains_encountered journey_duration initial_trains = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_encounters_l3608_360823


namespace NUMINAMATH_CALUDE_marble_probability_l3608_360871

/-- The probability of drawing a red, blue, or green marble from a bag -/
theorem marble_probability (red blue green yellow : ℕ) : 
  red = 5 → blue = 4 → green = 3 → yellow = 6 →
  (red + blue + green : ℚ) / (red + blue + green + yellow) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3608_360871


namespace NUMINAMATH_CALUDE_vessel_width_proof_l3608_360852

/-- The width of a rectangular vessel's base when a cube is immersed in it -/
theorem vessel_width_proof (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) 
  (h_cube_edge : cube_edge = 16)
  (h_vessel_length : vessel_length = 20)
  (h_water_rise : water_rise = 13.653333333333334) : 
  (cube_edge ^ 3) / (vessel_length * water_rise) = 15 := by
  sorry

end NUMINAMATH_CALUDE_vessel_width_proof_l3608_360852


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3608_360825

def Strip (n : ℤ) := {p : ℝ × ℝ | n ≤ p.1 ∧ p.1 < n + 1}

def ColoredStrip (n : ℤ) := Strip n → Bool

structure ColoredPlane :=
  (coloring : ℤ → Bool)

def hasMonochromaticRectangle (cp : ColoredPlane) (a b : ℕ) : Prop :=
  ∃ (x y : ℝ), 
    cp.coloring ⌊x⌋ = cp.coloring ⌊x + a⌋ ∧
    cp.coloring ⌊x⌋ = cp.coloring ⌊y⌋ ∧
    cp.coloring ⌊x⌋ = cp.coloring ⌊y + b⌋

theorem monochromatic_rectangle_exists (a b : ℕ) (h : a ≠ b) :
  ∀ cp : ColoredPlane, hasMonochromaticRectangle cp a b :=
sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3608_360825


namespace NUMINAMATH_CALUDE_min_cups_to_fill_container_l3608_360850

def container_capacity : ℝ := 640
def cup_capacity : ℝ := 120

theorem min_cups_to_fill_container : 
  ∃ n : ℕ, (n : ℝ) * cup_capacity ≥ container_capacity ∧ 
  ∀ m : ℕ, (m : ℝ) * cup_capacity ≥ container_capacity → n ≤ m ∧ 
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_min_cups_to_fill_container_l3608_360850


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l3608_360804

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (a b c d e f : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (d_pos : d > 0)
  (e_pos : e > 0)
  (f_pos : f > 0)

-- Define what it means for a quadrilateral to be cyclic
def is_cyclic (q : Quadrilateral) : Prop :=
  q.e^2 + q.f^2 = q.b^2 + q.d^2 + 2*q.a*q.c

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) :
  q.e^2 + q.f^2 ≤ q.b^2 + q.d^2 + 2*q.a*q.c ∧
  (q.e^2 + q.f^2 = q.b^2 + q.d^2 + 2*q.a*q.c ↔ is_cyclic q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l3608_360804


namespace NUMINAMATH_CALUDE_period_2_gym_class_size_l3608_360817

theorem period_2_gym_class_size :
  ∀ (period_2_size : ℕ),
  (2 * period_2_size - 5 = 11) →
  period_2_size = 8 := by
sorry

end NUMINAMATH_CALUDE_period_2_gym_class_size_l3608_360817


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_18_l3608_360845

/-- The area of a triangle with vertices at (1, 2), (7, 6), and (1, 8) is 18 square units. -/
theorem triangle_area : ℝ :=
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (7, 6)
  let C : ℝ × ℝ := (1, 8)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  18

/-- The theorem statement. -/
theorem triangle_area_is_18 : triangle_area = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_18_l3608_360845


namespace NUMINAMATH_CALUDE_right_angle_in_clerts_l3608_360830

/-- In a system where a full circle is measured as 500 units, a right angle is 125 units. -/
theorem right_angle_in_clerts (full_circle : ℕ) (right_angle : ℕ) 
  (h1 : full_circle = 500) 
  (h2 : right_angle = full_circle / 4) : 
  right_angle = 125 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_in_clerts_l3608_360830


namespace NUMINAMATH_CALUDE_simplify_expression_range_of_values_find_values_l3608_360866

-- Question 1
theorem simplify_expression (a : ℝ) (h : 3 ≤ a ∧ a ≤ 7) :
  Real.sqrt ((3 - a)^2) + Real.sqrt ((a - 7)^2) = 4 :=
sorry

-- Question 2
theorem range_of_values (a : ℝ) :
  Real.sqrt ((a - 1)^2) + Real.sqrt ((a - 6)^2) = 5 ↔ 1 ≤ a ∧ a ≤ 6 :=
sorry

-- Question 3
theorem find_values (a : ℝ) :
  Real.sqrt ((a + 1)^2) + Real.sqrt ((a - 3)^2) = 6 ↔ a = -2 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_range_of_values_find_values_l3608_360866


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3608_360890

/-- 
Given an isosceles triangle where one of the angles opposite an equal side is 40°,
prove that the largest angle measures 100°.
-/
theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = β →            -- The triangle is isosceles (two angles are equal)
  α = 40 →           -- One of the angles opposite an equal side is 40°
  max α (max β γ) = 100 := by  -- The largest angle measures 100°
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3608_360890


namespace NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l3608_360896

def atomic_weight_Al : ℝ := 27
def atomic_weight_O : ℝ := 16
def atomic_weight_H : ℝ := 1

def num_Al : ℕ := 1
def num_O : ℕ := 3

def molecular_weight : ℝ := 78

theorem hydrogen_atoms_in_compound :
  ∃ (num_H : ℕ), 
    (num_Al : ℝ) * atomic_weight_Al + 
    (num_O : ℝ) * atomic_weight_O + 
    (num_H : ℝ) * atomic_weight_H = molecular_weight ∧
    num_H = 3 := by sorry

end NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l3608_360896


namespace NUMINAMATH_CALUDE_product_inequality_l3608_360810

theorem product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (2 + x) * (2 + y) * (2 + z) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3608_360810


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l3608_360815

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l3608_360815


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3608_360812

/-- Given a principal P put at simple interest for 3 years, if increasing the interest rate by 2% 
    results in Rs. 360 more interest, then P = 6000. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 2) * 3) / 100 = (P * R * 3) / 100 + 360 → P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3608_360812


namespace NUMINAMATH_CALUDE_solve_equation_l3608_360882

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3608_360882


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l3608_360833

def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * x + 1

def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x = 0

theorem quadratic_real_roots_range (m : ℝ) :
  has_real_roots m ↔ m ≤ 2 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l3608_360833


namespace NUMINAMATH_CALUDE_rational_cube_root_sum_implies_rational_inverse_sum_l3608_360844

theorem rational_cube_root_sum_implies_rational_inverse_sum 
  (p q r : ℚ) 
  (h : ∃ (x : ℚ), x = (p^2*q)^(1/3) + (q^2*r)^(1/3) + (r^2*p)^(1/3)) : 
  ∃ (y : ℚ), y = 1/(p^2*q)^(1/3) + 1/(q^2*r)^(1/3) + 1/(r^2*p)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_root_sum_implies_rational_inverse_sum_l3608_360844


namespace NUMINAMATH_CALUDE_weight_replacement_l3608_360876

theorem weight_replacement (W : ℝ) (original_weight replaced_weight : ℝ) : 
  W / 10 + 2.5 = (W - replaced_weight + 75) / 10 → replaced_weight = 50 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l3608_360876


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3608_360840

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (vote_difference_percentage : ℝ)
  (candidate_b_votes : ℕ)
  (h1 : total_votes = 6720)
  (h2 : vote_difference_percentage = 0.15)
  (h3 : candidate_b_votes = 2184) :
  (total_votes - (2 * candidate_b_votes + vote_difference_percentage * total_votes)) / total_votes = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3608_360840


namespace NUMINAMATH_CALUDE_ten_people_seating_arrangement_l3608_360892

/-- The number of ways to seat n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange 3 people in a block where one person is fixed between the other two -/
def fixedBlockArrangements : ℕ := 2

theorem ten_people_seating_arrangement :
  roundTableArrangements 9 * fixedBlockArrangements = 80640 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_seating_arrangement_l3608_360892


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3608_360889

theorem fraction_equivalence : (16 : ℝ) / (8 * 17) = 1.6 / (0.8 * 17) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3608_360889


namespace NUMINAMATH_CALUDE_simplify_fraction_l3608_360811

theorem simplify_fraction (a : ℝ) (h : a = 2) : (15 * a^4) / (75 * a^3) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3608_360811


namespace NUMINAMATH_CALUDE_permutation_cover_iff_m_gt_half_n_l3608_360821

/-- A permutation of the set {1, ..., n} -/
def Permutation (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- Two permutations have common points if they agree on at least one element -/
def have_common_points {n : ℕ} (f g : Permutation n) : Prop :=
  ∃ k : Fin n, f.val k = g.val k

/-- The main theorem: m permutations cover all permutations iff m > n/2 -/
theorem permutation_cover_iff_m_gt_half_n (n m : ℕ) :
  (∃ (fs : Fin m → Permutation n), ∀ f : Permutation n, ∃ i : Fin m, have_common_points f (fs i)) ↔
  m > n / 2 := by sorry

end NUMINAMATH_CALUDE_permutation_cover_iff_m_gt_half_n_l3608_360821


namespace NUMINAMATH_CALUDE_elisa_math_books_l3608_360891

theorem elisa_math_books :
  ∀ (total math lit : ℕ),
  total < 100 →
  total = 24 + math + lit →
  (math + 1) * 9 = total + 1 →
  lit * 4 = total + 1 →
  math = 7 := by
sorry

end NUMINAMATH_CALUDE_elisa_math_books_l3608_360891


namespace NUMINAMATH_CALUDE_composition_of_convex_increasing_and_convex_is_convex_l3608_360858

def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x ≤ y → f x ≤ f y

theorem composition_of_convex_increasing_and_convex_is_convex
  (f g : ℝ → ℝ) (hf : IsConvex f) (hg : IsConvex g) (hf_inc : IsIncreasing f) :
  IsConvex (f ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_convex_increasing_and_convex_is_convex_l3608_360858


namespace NUMINAMATH_CALUDE_x_value_proof_l3608_360860

theorem x_value_proof (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 7)
  (eq2 : x^2 + 3 * y = 17) : 
  x = 3.5 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3608_360860


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3608_360880

-- Problem 1
theorem problem_one : (-1/3)⁻¹ + (Real.pi - 3.14)^0 = -2 := by sorry

-- Problem 2
theorem problem_two : ∀ x : ℝ, (2*x - 3)^2 - 2*x*(2*x - 6) = 9 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3608_360880


namespace NUMINAMATH_CALUDE_exam_question_count_exam_question_count_proof_l3608_360813

theorem exam_question_count (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) 
  (total_marks : ℕ) (correct_answers : ℕ) (total_questions : ℕ) : Prop :=
  (marks_per_correct = 4) →
  (marks_per_incorrect = 1) →
  (total_marks = 120) →
  (correct_answers = 40) →
  (marks_per_correct * correct_answers - marks_per_incorrect * (total_questions - correct_answers) = total_marks) →
  total_questions = 80

-- Proof
theorem exam_question_count_proof : 
  exam_question_count 4 1 120 40 80 := by sorry

end NUMINAMATH_CALUDE_exam_question_count_exam_question_count_proof_l3608_360813


namespace NUMINAMATH_CALUDE_total_students_is_63_l3608_360898

/-- The number of tables in the classroom -/
def num_tables : ℕ := 6

/-- The number of students currently sitting at each table -/
def students_per_table : ℕ := 3

/-- The number of girls who went to the bathroom -/
def girls_in_bathroom : ℕ := 4

/-- The number of students in new group 1 -/
def new_group1 : ℕ := 4

/-- The number of students in new group 2 -/
def new_group2 : ℕ := 5

/-- The number of students in new group 3 -/
def new_group3 : ℕ := 6

/-- The number of foreign exchange students from Germany -/
def german_students : ℕ := 2

/-- The number of foreign exchange students from France -/
def french_students : ℕ := 4

/-- The number of foreign exchange students from Norway -/
def norwegian_students : ℕ := 3

/-- The number of foreign exchange students from Italy -/
def italian_students : ℕ := 1

/-- The total number of students supposed to be in the class -/
def total_students : ℕ := 
  num_tables * students_per_table + 
  girls_in_bathroom + 
  4 * girls_in_bathroom + 
  new_group1 + new_group2 + new_group3 + 
  german_students + french_students + norwegian_students + italian_students

theorem total_students_is_63 : total_students = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_63_l3608_360898


namespace NUMINAMATH_CALUDE_exists_unresolved_conjecture_l3608_360805

/-- A structure representing a mathematical conjecture -/
structure Conjecture where
  statement : Prop
  is_proven : Prop
  is_disproven : Prop

/-- A predicate that determines if a conjecture is unresolved -/
def is_unresolved (c : Conjecture) : Prop :=
  ¬c.is_proven ∧ ¬c.is_disproven

/-- There exists at least one unresolved conjecture in mathematics -/
theorem exists_unresolved_conjecture : ∃ c : Conjecture, is_unresolved c := by
  sorry

#check exists_unresolved_conjecture

end NUMINAMATH_CALUDE_exists_unresolved_conjecture_l3608_360805


namespace NUMINAMATH_CALUDE_part_one_part_two_l3608_360869

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
def B := {x : ℝ | x^2 - 5*x + 6 = 0}
def C := {x : ℝ | x^2 + 2*x - 8 = 0}

-- Theorem for part (1)
theorem part_one (a : ℝ) : A a ∩ B = A a ∪ B → a = 5 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (∅ ⊂ A a ∩ B) ∧ (A a ∩ C = ∅) → a = -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3608_360869


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3608_360803

/-- The lateral surface area of a cylinder, given the diagonal length and intersection angle of its rectangular lateral surface. -/
theorem cylinder_lateral_surface_area 
  (d : ℝ) 
  (α : ℝ) 
  (h_d_pos : d > 0) 
  (h_α_pos : α > 0) 
  (h_α_lt_pi : α < π) : 
  ∃ (S : ℝ), S = (1/2) * d^2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3608_360803


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l3608_360855

theorem similar_triangles_perimeter (h_small h_large : ℝ) (p_small p_large : ℝ) :
  h_small / h_large = 3 / 5 →
  p_small = 12 →
  p_small / p_large = h_small / h_large →
  p_large = 20 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l3608_360855


namespace NUMINAMATH_CALUDE_minimize_z_l3608_360870

-- Define the function z
def z (x a b c d : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c*(x - a) + d*(x - b)

-- Theorem statement
theorem minimize_z (a b c d : ℝ) :
  ∃ x : ℝ, ∀ y : ℝ, z x a b c d ≤ z y a b c d ∧ x = (2*(a+b) - (c+d)) / 4 :=
sorry

end NUMINAMATH_CALUDE_minimize_z_l3608_360870


namespace NUMINAMATH_CALUDE_sequence_term_correct_l3608_360816

def sequence_sum (n : ℕ) : ℕ := 2^n + 3

def sequence_term (n : ℕ) : ℕ :=
  match n with
  | 1 => 5
  | _ => 2^(n-1)

theorem sequence_term_correct :
  ∀ n : ℕ, n ≥ 1 → sequence_term n = 
    if n = 1 
    then sequence_sum 1
    else sequence_sum n - sequence_sum (n-1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_correct_l3608_360816


namespace NUMINAMATH_CALUDE_derivative_reciprocal_l3608_360802

theorem derivative_reciprocal (x : ℝ) (hx : x ≠ 0) :
  deriv (fun x => 1 / x) x = -(1 / x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_reciprocal_l3608_360802


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l3608_360857

theorem largest_gcd_of_sum_1008 :
  ∃ (max_gcd : ℕ), ∀ (a b : ℕ), 
    a > 0 → b > 0 → a + b = 1008 →
    gcd a b ≤ max_gcd ∧
    ∃ (a' b' : ℕ), a' > 0 ∧ b' > 0 ∧ a' + b' = 1008 ∧ gcd a' b' = max_gcd :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l3608_360857


namespace NUMINAMATH_CALUDE_basketball_score_total_l3608_360874

theorem basketball_score_total (tim joe ken : ℕ) : 
  tim = joe + 20 →
  tim = ken / 2 →
  tim = 30 →
  tim + joe + ken = 100 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_total_l3608_360874


namespace NUMINAMATH_CALUDE_f_extremum_and_monotonicity_l3608_360819

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem f_extremum_and_monotonicity :
  (∀ x > 0, f (-1/2) x ≤ f (-1/2) 1) ∧ f (-1/2) 1 = 0 ∧
  (∀ a : ℝ, (∀ x > 0, ∀ y > 0, x < y → f a x < f a y) ↔ a ≥ 1 / (2 * Real.exp 2)) :=
sorry

end NUMINAMATH_CALUDE_f_extremum_and_monotonicity_l3608_360819


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3608_360879

theorem rectangle_max_area (l w : ℝ) : 
  l + w = 30 →  -- Perimeter condition (half of 60)
  l - w = 10 →  -- Difference between length and width
  l * w ≤ 200   -- Maximum area
  := by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3608_360879


namespace NUMINAMATH_CALUDE_cows_per_herd_l3608_360863

theorem cows_per_herd (total_cows : ℕ) (num_herds : ℕ) (h1 : total_cows = 320) (h2 : num_herds = 8) :
  total_cows / num_herds = 40 := by
  sorry

end NUMINAMATH_CALUDE_cows_per_herd_l3608_360863


namespace NUMINAMATH_CALUDE_y_min_max_sum_l3608_360884

theorem y_min_max_sum (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ y', (∃ x' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11) → m ≤ y' ∧ y' ≤ M) ∧
  m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_y_min_max_sum_l3608_360884


namespace NUMINAMATH_CALUDE_smartphone_cost_l3608_360886

theorem smartphone_cost (selling_price : ℝ) (loss_percentage : ℝ) (initial_cost : ℝ) : 
  selling_price = 255 ∧ 
  loss_percentage = 15 ∧ 
  selling_price = initial_cost * (1 - loss_percentage / 100) →
  initial_cost = 300 :=
by sorry

end NUMINAMATH_CALUDE_smartphone_cost_l3608_360886


namespace NUMINAMATH_CALUDE_angleBisectorRatioNotDeterminesShape_twoAnglesAndSideDeterminesShape_angleBisectorRatiosDetermineShape_sideLengthRatiosDetermineShape_threeAnglesDetermineShape_l3608_360851

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of an angle bisector to its corresponding opposite side --/
def angleBisectorToOppositeSideRatio (t : Triangle) : ℝ := sorry

/-- Determines if two triangles have the same shape (are similar) --/
def sameShape (t1 t2 : Triangle) : Prop := sorry

/-- The theorem stating that the ratio of an angle bisector to its corresponding opposite side
    does not uniquely determine the shape of a triangle --/
theorem angleBisectorRatioNotDeterminesShape :
  ∃ t1 t2 : Triangle, 
    angleBisectorToOppositeSideRatio t1 = angleBisectorToOppositeSideRatio t2 ∧
    ¬ sameShape t1 t2 := by sorry

/-- The theorem stating that the ratio of two angles and the included side
    uniquely determines the shape of a triangle --/
theorem twoAnglesAndSideDeterminesShape (α β : ℝ) (s : ℝ) :
  ∀ t1 t2 : Triangle,
    (α = sorry) ∧ (β = sorry) ∧ (s = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that the ratios of the three angle bisectors
    uniquely determine the shape of a triangle --/
theorem angleBisectorRatiosDetermineShape (r1 r2 r3 : ℝ) :
  ∀ t1 t2 : Triangle,
    (r1 = sorry) ∧ (r2 = sorry) ∧ (r3 = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that the ratios of the three side lengths
    uniquely determine the shape of a triangle --/
theorem sideLengthRatiosDetermineShape (r1 r2 r3 : ℝ) :
  ∀ t1 t2 : Triangle,
    (r1 = sorry) ∧ (r2 = sorry) ∧ (r3 = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that three angles
    uniquely determine the shape of a triangle --/
theorem threeAnglesDetermineShape (α β γ : ℝ) :
  ∀ t1 t2 : Triangle,
    (α = sorry) ∧ (β = sorry) ∧ (γ = sorry) →
    sameShape t1 t2 := by sorry

end NUMINAMATH_CALUDE_angleBisectorRatioNotDeterminesShape_twoAnglesAndSideDeterminesShape_angleBisectorRatiosDetermineShape_sideLengthRatiosDetermineShape_threeAnglesDetermineShape_l3608_360851


namespace NUMINAMATH_CALUDE_root_quadratic_implies_value_l3608_360839

theorem root_quadratic_implies_value (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → 2*m^2 - 4*m = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_implies_value_l3608_360839


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3608_360895

theorem quadratic_inequality_solution_range (q : ℝ) : 
  (q > 0) → 
  (∃ x : ℝ, x^2 - 8*x + q < 0) ↔ 
  (q > 0 ∧ q < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3608_360895


namespace NUMINAMATH_CALUDE_puss_in_boots_pikes_l3608_360894

theorem puss_in_boots_pikes (x : ℚ) : x = 4 + (1/2) * x → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_puss_in_boots_pikes_l3608_360894


namespace NUMINAMATH_CALUDE_total_flowers_and_sticks_l3608_360827

theorem total_flowers_and_sticks (num_pots : ℕ) (flowers_per_pot : ℕ) (sticks_per_pot : ℕ) 
  (h1 : num_pots = 466) 
  (h2 : flowers_per_pot = 53) 
  (h3 : sticks_per_pot = 181) : 
  num_pots * flowers_per_pot + num_pots * sticks_per_pot = 109044 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_and_sticks_l3608_360827


namespace NUMINAMATH_CALUDE_P_not_subset_Q_l3608_360846

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | |x| > 0}

-- Statement to prove
theorem P_not_subset_Q : ¬(P ⊆ Q) := by
  sorry

end NUMINAMATH_CALUDE_P_not_subset_Q_l3608_360846


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l3608_360856

/-- Workshop salary problem -/
theorem workshop_salary_problem 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_others : ℕ) 
  (h1 : total_workers = 14)
  (h2 : num_technicians = 7)
  (h3 : avg_salary_technicians = 10000)
  (h4 : avg_salary_others = 6000) :
  (num_technicians * avg_salary_technicians + 
   (total_workers - num_technicians) * avg_salary_others) / total_workers = 8000 := by
  sorry


end NUMINAMATH_CALUDE_workshop_salary_problem_l3608_360856


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l3608_360801

/-- The number of heartbeats during a race --/
def heartbeats_during_race (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proof that the athlete's heart beats 19200 times during the race --/
theorem athlete_heartbeats :
  heartbeats_during_race 160 20 6 = 19200 := by
  sorry

#eval heartbeats_during_race 160 20 6

end NUMINAMATH_CALUDE_athlete_heartbeats_l3608_360801


namespace NUMINAMATH_CALUDE_jay_change_is_twenty_l3608_360818

-- Define the prices of items and the payment amount
def book_price : ℕ := 25
def pen_price : ℕ := 4
def ruler_price : ℕ := 1
def payment : ℕ := 50

-- Define the change received
def change : ℕ := payment - (book_price + pen_price + ruler_price)

-- Theorem statement
theorem jay_change_is_twenty : change = 20 := by
  sorry

end NUMINAMATH_CALUDE_jay_change_is_twenty_l3608_360818


namespace NUMINAMATH_CALUDE_player_A_win_probability_l3608_360893

/-- The probability of winning a single game for either player -/
def win_prob : ℚ := 1/2

/-- The number of games player A needs to win to become the final winner -/
def games_needed_A : ℕ := 2

/-- The number of games player B needs to win to become the final winner -/
def games_needed_B : ℕ := 3

/-- The probability of player A becoming the final winner -/
def prob_A_wins : ℚ := 11/16

theorem player_A_win_probability :
  prob_A_wins = 11/16 := by sorry

end NUMINAMATH_CALUDE_player_A_win_probability_l3608_360893


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l3608_360807

/-- The expected value of winnings for a biased coin flip -/
theorem biased_coin_expected_value :
  let p_heads : ℚ := 1/4  -- Probability of heads
  let p_tails : ℚ := 3/4  -- Probability of tails
  let win_heads : ℚ := 4  -- Amount won for heads
  let lose_tails : ℚ := 3 -- Amount lost for tails
  p_heads * win_heads - p_tails * lose_tails = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l3608_360807


namespace NUMINAMATH_CALUDE_log_expression_equals_zero_l3608_360837

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_equals_zero (x : ℝ) (h : x > 10) :
  (log x) ^ (log (log (log x))) - (log (log x)) ^ (log (log x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_zero_l3608_360837


namespace NUMINAMATH_CALUDE_adrianna_gum_l3608_360897

/-- Calculates the remaining pieces of gum after sharing with friends -/
def remaining_gum (initial : ℕ) (additional : ℕ) (friends : ℕ) : ℕ :=
  initial + additional - friends

/-- Proves that Adrianna has 2 pieces of gum left -/
theorem adrianna_gum : remaining_gum 10 3 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_l3608_360897


namespace NUMINAMATH_CALUDE_problem_solution_l3608_360848

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^2 + 12
def g (x : ℝ) : ℝ := x^2 - 6

-- State the theorem
theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 12) : a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3608_360848


namespace NUMINAMATH_CALUDE_smallest_angle_60_implies_n_3_or_4_l3608_360859

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space determined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The angle between two lines in 3D space -/
def angle (l1 l2 : Line3D) : ℝ := sorry

/-- A configuration of n points in 3D space -/
def Configuration (n : ℕ) := Fin n → Point3D

/-- The smallest angle formed by any pair of lines in a configuration -/
def smallestAngle (config : Configuration n) : ℝ := sorry

theorem smallest_angle_60_implies_n_3_or_4 (n : ℕ) (h1 : n > 2) 
  (config : Configuration n) (h2 : smallestAngle config = 60) :
  n = 3 ∨ n = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_60_implies_n_3_or_4_l3608_360859


namespace NUMINAMATH_CALUDE_rational_numbers_closed_l3608_360829

-- Define the set of rational numbers
def RationalNumbers : Set ℚ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

-- State the theorem
theorem rational_numbers_closed :
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a + b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a - b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a * b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → b ≠ 0 → (a / b) ∈ RationalNumbers) :=
by sorry

end NUMINAMATH_CALUDE_rational_numbers_closed_l3608_360829


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_inequality_l3608_360831

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d
def S (a₁ d : ℚ) (n : ℕ) : ℚ := n * a₁ + n * (n - 1 : ℚ) / 2 * d

-- State the theorem
theorem arithmetic_sequence_sum_inequality 
  (p q : ℕ) (a₁ d : ℚ) (hp : p ≠ q) (hSp : S a₁ d p = p / q) (hSq : S a₁ d q = q / p) :
  S a₁ d (p + q) > 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_inequality_l3608_360831


namespace NUMINAMATH_CALUDE_smallest_number_l3608_360841

def numbers : Set ℤ := {0, -2, -1, 3}

theorem smallest_number (n : ℤ) (hn : n ∈ numbers) : -2 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3608_360841


namespace NUMINAMATH_CALUDE_scientific_notation_86000000_l3608_360835

theorem scientific_notation_86000000 : 
  86000000 = 8.6 * (10 : ℝ)^7 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_86000000_l3608_360835


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3608_360873

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = 5) : z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3608_360873


namespace NUMINAMATH_CALUDE_find_a_l3608_360838

theorem find_a (a b : ℚ) (h1 : a / 3 = b / 2) (h2 : a + b = 10) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3608_360838


namespace NUMINAMATH_CALUDE_production_rates_and_minimum_machines_l3608_360836

/-- Represents the production rate of machine A in kg per hour -/
def machine_a_rate : ℝ := 60

/-- Represents the production rate of machine B in kg per hour -/
def machine_b_rate : ℝ := 50

/-- The difference in production rate between machine A and B -/
def rate_difference : ℝ := 10

/-- The total number of machines used -/
def total_machines : ℕ := 18

/-- The minimum required production in kg per hour -/
def min_production : ℝ := 1000

theorem production_rates_and_minimum_machines :
  (machine_a_rate = machine_b_rate + rate_difference) ∧
  (600 / machine_a_rate = 500 / machine_b_rate) ∧
  (∃ (m : ℕ), m ≤ total_machines ∧ 
    machine_a_rate * m + machine_b_rate * (total_machines - m) ≥ min_production ∧
    ∀ (n : ℕ), n < m → 
      machine_a_rate * n + machine_b_rate * (total_machines - n) < min_production) :=
by sorry

end NUMINAMATH_CALUDE_production_rates_and_minimum_machines_l3608_360836


namespace NUMINAMATH_CALUDE_cost_calculation_l3608_360867

theorem cost_calculation (pencil_cost pen_cost eraser_cost : ℝ) 
  (eq1 : 8 * pencil_cost + 2 * pen_cost + eraser_cost = 4.60)
  (eq2 : 2 * pencil_cost + 5 * pen_cost + eraser_cost = 3.90)
  (eq3 : pencil_cost + pen_cost + 3 * eraser_cost = 2.75) :
  4 * pencil_cost + 3 * pen_cost + 2 * eraser_cost = 7.4135 := by
sorry

end NUMINAMATH_CALUDE_cost_calculation_l3608_360867


namespace NUMINAMATH_CALUDE_remainder_theorem_application_l3608_360868

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^6 + E * x^4 + F * x^2 + 6
  (q 2 = 16) → (q (-2) = 16) := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_application_l3608_360868


namespace NUMINAMATH_CALUDE_calculate_expression_l3608_360800

theorem calculate_expression : (-5) / ((1 / 4) - (1 / 3)) * 12 = 720 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3608_360800


namespace NUMINAMATH_CALUDE_square_perimeter_l3608_360824

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 200 → 
  side^2 = area → 
  perimeter = 4 * side → 
  perimeter = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3608_360824


namespace NUMINAMATH_CALUDE_expression_equivalence_l3608_360881

theorem expression_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3608_360881


namespace NUMINAMATH_CALUDE_interior_triangle_area_l3608_360862

/-- Given three squares with areas 36, 64, and 100, where the largest square is diagonal to the other two squares, the area of the interior triangle is 24. -/
theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (h_diagonal : c = max a b) : (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l3608_360862


namespace NUMINAMATH_CALUDE_bucket_water_problem_l3608_360854

/-- Given two equations representing the weight of a bucket with water,
    prove that the original amount of water is 3 kg and the bucket weighs 4 kg. -/
theorem bucket_water_problem (x y : ℝ) 
  (eq1 : 4 * x + y = 16)
  (eq2 : 6 * x + y = 22) :
  x = 3 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_bucket_water_problem_l3608_360854


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3608_360806

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 8) ^ x) + Real.sqrt ((3 - Real.sqrt 8) ^ x) = 6) ↔ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3608_360806


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3608_360847

/-- The equation of a hyperbola with given properties -/
theorem hyperbola_equation (m n : ℝ) (h : m < 0) :
  (∀ x y : ℝ, x^2 / m + y^2 / n = 1) →  -- Given hyperbola equation
  (n = 1) →                            -- Derived from eccentricity = 2 and a = 1
  (m = -3) →                           -- Derived from b^2 = 3
  (∀ x y : ℝ, y^2 - x^2 / 3 = 1) :=    -- Equation to prove
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3608_360847


namespace NUMINAMATH_CALUDE_relationship_abc_l3608_360814

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 4 then 4 / x + 1 else Real.log x / Real.log 2

theorem relationship_abc (a b c : ℝ) :
  (0 < a ∧ a < 4) →
  (b ≥ 4) →
  (f a = c) →
  (f b = c) →
  (deriv f b < 0) →
  b > a ∧ a > c :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l3608_360814


namespace NUMINAMATH_CALUDE_pet_store_cages_l3608_360885

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : puppies_per_cage = 9) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3608_360885


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l3608_360828

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Checks if two line segments are parallel -/
def isParallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem trapezoid_triangle_area
  (ABCD : Trapezoid)
  (E : Point)
  (h1 : isPerpendicular ABCD.A ABCD.D ABCD.D ABCD.C)
  (h2 : ABCD.A.x - ABCD.D.x = 5)
  (h3 : ABCD.A.y - ABCD.B.y = 5)
  (h4 : ABCD.D.x - ABCD.C.x = 10)
  (h5 : isOnSegment E ABCD.D ABCD.C)
  (h6 : E.x - ABCD.D.x = 4)
  (h7 : isParallel ABCD.B E ABCD.A ABCD.D)
  : triangleArea ABCD.A ABCD.D E = 10 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l3608_360828


namespace NUMINAMATH_CALUDE_contractor_fine_proof_l3608_360822

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine_per_day (total_days : ℕ) (pay_per_day : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let total_earned := pay_per_day * worked_days
  (total_earned - total_payment) / absent_days

/-- Proves that the fine per day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_proof :
  calculate_fine_per_day 30 25 425 10 = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_contractor_fine_proof_l3608_360822


namespace NUMINAMATH_CALUDE_matilda_jellybeans_l3608_360877

/-- Given that:
    1. Matilda has half as many jellybeans as Matt
    2. Matt has ten times as many jellybeans as Steve
    3. Steve has 84 jellybeans
    Prove that Matilda has 420 jellybeans. -/
theorem matilda_jellybeans (steve_jellybeans : ℕ) (matt_jellybeans : ℕ) (matilda_jellybeans : ℕ)
  (h1 : steve_jellybeans = 84)
  (h2 : matt_jellybeans = 10 * steve_jellybeans)
  (h3 : matilda_jellybeans = matt_jellybeans / 2) :
  matilda_jellybeans = 420 := by
  sorry

end NUMINAMATH_CALUDE_matilda_jellybeans_l3608_360877


namespace NUMINAMATH_CALUDE_impossible_30_gon_numbering_l3608_360842

theorem impossible_30_gon_numbering : ¬ ∃ (f : Fin 30 → Nat),
  (∀ i, f i ∈ Finset.range 30) ∧
  (∀ i, f i ≠ 0) ∧
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i : Fin 30, ∃ k : Nat, (f i + f ((i + 1) % 30) : Nat) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_impossible_30_gon_numbering_l3608_360842


namespace NUMINAMATH_CALUDE_library_visitors_average_l3608_360809

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 510) (h2 : other_day_visitors = 240) 
  (h3 : days_in_month = 30) : 
  (5 * sunday_visitors + 25 * other_day_visitors) / days_in_month = 285 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l3608_360809


namespace NUMINAMATH_CALUDE_xy_sum_values_l3608_360887

theorem xy_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_lt : x < 25) (hy_lt : y < 25) 
  (h_eq : x + y + x * y = 119) : 
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end NUMINAMATH_CALUDE_xy_sum_values_l3608_360887


namespace NUMINAMATH_CALUDE_circle_radius_l3608_360849

theorem circle_radius (x y : ℝ) (h : x + y = 150 * Real.pi) : 
  ∃ (r : ℝ), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = Real.sqrt 151 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3608_360849


namespace NUMINAMATH_CALUDE_monday_polygons_tuesday_segments_wednesday_polygons_l3608_360888

/-- Represents the types of polygons Miky can draw -/
inductive Polygon
| Square
| Pentagon
| Hexagon

/-- Number of sides for each polygon type -/
def sides (p : Polygon) : Nat :=
  match p with
  | .Square => 4
  | .Pentagon => 5
  | .Hexagon => 6

/-- Number of diagonals for each polygon type -/
def diagonals (p : Polygon) : Nat :=
  match p with
  | .Square => 2
  | .Pentagon => 5
  | .Hexagon => 9

/-- Total number of line segments (sides + diagonals) for each polygon type -/
def totalSegments (p : Polygon) : Nat :=
  sides p + diagonals p

theorem monday_polygons :
  ∃ p : Polygon, sides p = diagonals p ∧ p = Polygon.Pentagon :=
sorry

theorem tuesday_segments (n : Nat) (h : n * sides Polygon.Hexagon = 18) :
  n * diagonals Polygon.Hexagon = 27 :=
sorry

theorem wednesday_polygons (n : Nat) (h : n * totalSegments Polygon.Pentagon = 70) :
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_monday_polygons_tuesday_segments_wednesday_polygons_l3608_360888


namespace NUMINAMATH_CALUDE_integer_with_specific_cube_root_l3608_360861

theorem integer_with_specific_cube_root : ∃ n : ℕ+,
  (↑n : ℝ) > 0 ∧
  ∃ k : ℕ, n = 24 * k ∧
  (9 : ℝ) < (↑n : ℝ) ^ (1/3) ∧ (↑n : ℝ) ^ (1/3) < 9.1 :=
by
  use 744
  sorry

end NUMINAMATH_CALUDE_integer_with_specific_cube_root_l3608_360861


namespace NUMINAMATH_CALUDE_apple_cost_l3608_360872

/-- The cost of apples under specific pricing conditions -/
theorem apple_cost (l q : ℚ) : 
  (30 * l + 3 * q = 333) →  -- Price for 33 kg
  (30 * l + 6 * q = 366) →  -- Price for 36 kg
  (15 * l = 150)            -- Price for 15 kg
:= by sorry

end NUMINAMATH_CALUDE_apple_cost_l3608_360872


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3608_360878

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 12) : 
  a^3 + 1/a^3 = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3608_360878


namespace NUMINAMATH_CALUDE_coffee_package_size_l3608_360843

theorem coffee_package_size (total_coffee : ℝ) (known_size : ℝ) (extra_known : ℕ) (unknown_count : ℕ) :
  total_coffee = 85 ∧ 
  known_size = 5 ∧ 
  extra_known = 2 ∧ 
  unknown_count = 5 → 
  ∃ (unknown_size : ℝ), 
    unknown_size * unknown_count + known_size * (unknown_count + extra_known) = total_coffee ∧ 
    unknown_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_coffee_package_size_l3608_360843


namespace NUMINAMATH_CALUDE_triangle_focus_property_l3608_360832

/-- Given a triangle ABC with vertices corresponding to complex numbers z₁, z₂, and z₃,
    and a point F corresponding to complex number z, prove that:
    (z - z₁)(z - z₂) + (z - z₂)(z - z₃) + (z - z₃)(z - z₁) = 0 -/
theorem triangle_focus_property (z z₁ z₂ z₃ : ℂ) : 
  (z - z₁) * (z - z₂) + (z - z₂) * (z - z₃) + (z - z₃) * (z - z₁) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_focus_property_l3608_360832
