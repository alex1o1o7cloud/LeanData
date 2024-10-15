import Mathlib

namespace NUMINAMATH_CALUDE_annual_savings_20_over_30_l1539_153926

/-- Represents the internet plans and their costs -/
structure InternetPlan where
  speed : ℕ  -- Speed in Mbps
  monthlyCost : ℕ  -- Monthly cost in dollars

/-- Calculates the annual cost of an internet plan -/
def annualCost (plan : InternetPlan) : ℕ :=
  plan.monthlyCost * 12

/-- Represents Marites' internet plans -/
def marites : {currentPlan : InternetPlan // currentPlan.speed = 10 ∧ currentPlan.monthlyCost = 20} :=
  ⟨⟨10, 20⟩, by simp⟩

/-- The 30 Mbps plan -/
def plan30 : InternetPlan :=
  ⟨30, 2 * marites.val.monthlyCost⟩

/-- The 20 Mbps plan -/
def plan20 : InternetPlan :=
  ⟨20, marites.val.monthlyCost + 10⟩

/-- Theorem: Annual savings when choosing 20 Mbps over 30 Mbps is $120 -/
theorem annual_savings_20_over_30 :
  annualCost plan30 - annualCost plan20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_annual_savings_20_over_30_l1539_153926


namespace NUMINAMATH_CALUDE_find_x_l1539_153930

theorem find_x (x y z : ℝ) 
  (h1 : x - y = 10) 
  (h2 : x * z = 2 * y) 
  (h3 : x + y = 14) : 
  x = 12 := by sorry

end NUMINAMATH_CALUDE_find_x_l1539_153930


namespace NUMINAMATH_CALUDE_remaining_pages_l1539_153958

-- Define the total number of pages
def total_pages : ℕ := 120

-- Define the percentage used for the science project
def science_project_percentage : ℚ := 25 / 100

-- Define the number of pages used for math homework
def math_homework_pages : ℕ := 10

-- Theorem statement
theorem remaining_pages :
  total_pages - (total_pages * science_project_percentage).floor - math_homework_pages = 80 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pages_l1539_153958


namespace NUMINAMATH_CALUDE_imaginary_cube_l1539_153979

theorem imaginary_cube (i : ℂ) : i^2 = -1 → 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_cube_l1539_153979


namespace NUMINAMATH_CALUDE_inequality_proof_l1539_153988

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  3 + a / b + b / c + c / a ≥ a + b + c + 1 / a + 1 / b + 1 / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1539_153988


namespace NUMINAMATH_CALUDE_extreme_points_sum_l1539_153986

/-- Given that x = 2 and x = -4 are extreme points of f(x) = x³ + px² + qx, prove that p + q = -21 -/
theorem extreme_points_sum (p q : ℝ) : 
  (∀ x : ℝ, x = 2 ∨ x = -4 → (3*x^2 + 2*p*x + q = 0)) → 
  p + q = -21 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_sum_l1539_153986


namespace NUMINAMATH_CALUDE_parabola_vertex_l1539_153993

/-- The vertex of the parabola defined by y² + 10y + 3x + 9 = 0 is (16/3, -5) -/
theorem parabola_vertex :
  let f : ℝ → ℝ → ℝ := λ x y ↦ y^2 + 10*y + 3*x + 9
  ∃! (x₀ y₀ : ℝ), (∀ x y, f x y = 0 → y ≥ y₀) ∧ f x₀ y₀ = 0 ∧ x₀ = 16/3 ∧ y₀ = -5 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1539_153993


namespace NUMINAMATH_CALUDE_circle_k_range_l1539_153947

/-- The equation of a circle in general form --/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

/-- The condition for the equation to represent a circle --/
def is_circle (k : ℝ) : Prop :=
  ∃ (h c r : ℝ), ∀ (x y : ℝ), circle_equation x y k ↔ (x - h)^2 + (y - c)^2 = r^2 ∧ r > 0

/-- The theorem stating the range of k for which the equation represents a circle --/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k > 4 ∨ k < -1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l1539_153947


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1539_153987

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 3*p - 2 = 0) → 
  (q^3 - 3*q - 2 = 0) → 
  (r^3 - 3*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1539_153987


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1539_153929

/-- Proves that the cost price of an article is 78.944 given the specified conditions --/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  marked_price = 98.68 →
  discount_rate = 0.05 →
  profit_rate = 0.25 →
  ∃ (cost_price : ℝ),
    (1 - discount_rate) * marked_price = cost_price * (1 + profit_rate) ∧
    cost_price = 78.944 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1539_153929


namespace NUMINAMATH_CALUDE_central_region_perimeter_l1539_153938

/-- The perimeter of the central region formed by four identical circles in a square formation --/
theorem central_region_perimeter (c : ℝ) (h : c = 48) : 
  let r := c / (2 * Real.pi)
  4 * (Real.pi * r / 2) = c :=
by sorry

end NUMINAMATH_CALUDE_central_region_perimeter_l1539_153938


namespace NUMINAMATH_CALUDE_point_opposite_sides_range_l1539_153920

/-- Determines if two points are on opposite sides of a line -/
def oppositeSides (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) < 0

theorem point_opposite_sides_range (a : ℝ) :
  oppositeSides 3 1 (-4) 6 3 (-2) a ↔ -7 < a ∧ a < 24 := by
  sorry

end NUMINAMATH_CALUDE_point_opposite_sides_range_l1539_153920


namespace NUMINAMATH_CALUDE_sum_representation_exists_l1539_153931

/-- Regular 15-gon inscribed in a circle -/
structure RegularPolygon :=
  (n : ℕ)
  (radius : ℝ)
  (is_regular : n = 15)
  (is_inscribed : radius = 15)

/-- Sum of lengths of all sides and diagonals -/
def sum_lengths (p : RegularPolygon) : ℝ := sorry

/-- Representation of the sum in the required form -/
structure SumRepresentation :=
  (a b c d : ℕ)
  (sum : ℝ)
  (eq : sum = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5)

/-- Theorem stating the existence of the representation -/
theorem sum_representation_exists (p : RegularPolygon) :
  ∃ (rep : SumRepresentation), sum_lengths p = rep.sum :=
sorry

end NUMINAMATH_CALUDE_sum_representation_exists_l1539_153931


namespace NUMINAMATH_CALUDE_range_of_a_l1539_153908

def set_A : Set ℝ := {x : ℝ | (3 * x) / (x + 1) ≤ 2}

def set_B (a : ℝ) : Set ℝ := {x : ℝ | a - 2 < x ∧ x < 2 * a + 1}

theorem range_of_a (a : ℝ) :
  set_A = set_B a → a ∈ Set.Ioo (1/2 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1539_153908


namespace NUMINAMATH_CALUDE_inequality_proof_l1539_153945

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1539_153945


namespace NUMINAMATH_CALUDE_factory_workers_count_l1539_153912

/-- The total number of workers in the factory -/
def total_workers : ℕ := 900

/-- The number of workers in Workshop B -/
def workshop_b_workers : ℕ := 300

/-- The total sample size -/
def total_sample : ℕ := 45

/-- The number of people sampled from Workshop A -/
def sample_a : ℕ := 20

/-- The number of people sampled from Workshop C -/
def sample_c : ℕ := 10

/-- The number of people sampled from Workshop B -/
def sample_b : ℕ := total_sample - sample_a - sample_c

theorem factory_workers_count :
  (sample_b : ℚ) / workshop_b_workers = (total_sample : ℚ) / total_workers :=
by sorry

end NUMINAMATH_CALUDE_factory_workers_count_l1539_153912


namespace NUMINAMATH_CALUDE_f_is_2x_plus_7_range_f_is_5_to_13_l1539_153990

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being a first-degree function
def is_first_degree (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- Define the given condition for f
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17

-- Theorem for the first part
theorem f_is_2x_plus_7 (h1 : is_first_degree f) (h2 : satisfies_condition f) :
  ∀ x, f x = 2 * x + 7 :=
sorry

-- Define the range of f for x ∈ (-1, 3]
def range_f : Set ℝ := { y | ∃ x ∈ Set.Ioc (-1) 3, f x = y }

-- Theorem for the second part
theorem range_f_is_5_to_13 (h1 : is_first_degree f) (h2 : satisfies_condition f) :
  range_f = Set.Ioc 5 13 :=
sorry

end NUMINAMATH_CALUDE_f_is_2x_plus_7_range_f_is_5_to_13_l1539_153990


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1539_153944

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0,
    and |f(x)| ≥ 2 for all real x, prove that the coordinates of
    the focus of the parabolic curve are (0, 1/(4a) + 2). -/
theorem parabola_focus_coordinates
  (a b : ℝ) (ha : a ≠ 0)
  (hf : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1 / (4 * a) + 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1539_153944


namespace NUMINAMATH_CALUDE_min_value_theorem_l1539_153975

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c = 3) :
  ∃ (min_value : ℝ), min_value = 9 ∧ 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
  x^2 + y^2 + (x + y)^2 + z^2 ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1539_153975


namespace NUMINAMATH_CALUDE_vector_perpendicular_value_l1539_153953

-- Define the vectors
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the perpendicularity condition
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem statement
theorem vector_perpendicular_value (x : ℝ) :
  perpendicular a (a.1 - (b x).1, a.2 - (b x).2) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_value_l1539_153953


namespace NUMINAMATH_CALUDE_restaurant_spend_l1539_153940

/-- The total amount spent by a group at a restaurant -/
def total_spent (n : ℕ) (individual_spends : Fin n → ℚ) : ℚ :=
  (Finset.univ.sum fun i => individual_spends i)

/-- The average expenditure of a group -/
def average_spend (n : ℕ) (individual_spends : Fin n → ℚ) : ℚ :=
  (total_spent n individual_spends) / n

theorem restaurant_spend :
  ∀ (individual_spends : Fin 8 → ℚ),
  (∀ i : Fin 7, individual_spends i = 10) →
  (individual_spends 7 = average_spend 8 individual_spends + 7) →
  total_spent 8 individual_spends = 88 := by
sorry

end NUMINAMATH_CALUDE_restaurant_spend_l1539_153940


namespace NUMINAMATH_CALUDE_tax_free_amount_correct_l1539_153954

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ := 600

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate applied to the portion exceeding the tax-free amount -/
def tax_rate : ℝ := 0.08

/-- The amount of tax paid -/
def tax_paid : ℝ := 89.6

/-- Theorem stating that the tax-free amount satisfies the given conditions -/
theorem tax_free_amount_correct : 
  tax_rate * (total_value - tax_free_amount) = tax_paid := by sorry

end NUMINAMATH_CALUDE_tax_free_amount_correct_l1539_153954


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1539_153961

theorem smaller_number_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x - y = 1650) (h4 : 0.075 * x = 0.125 * y) : y = 2475 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1539_153961


namespace NUMINAMATH_CALUDE_sqrt_difference_simplification_l1539_153963

theorem sqrt_difference_simplification (x : ℝ) (h : -1 < x ∧ x < 0) :
  Real.sqrt (x^2) - Real.sqrt ((x+1)^2) = -2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_simplification_l1539_153963


namespace NUMINAMATH_CALUDE_max_fraction_sum_l1539_153935

theorem max_fraction_sum (n : ℕ) (hn : n ≥ 2) :
  ∃ (a b c d : ℕ),
    a / b + c / d < 1 ∧
    a + c ≤ n ∧
    ∀ (a' b' c' d' : ℕ),
      a' / b' + c' / d' < 1 →
      a' + c' ≤ n →
      a' / b' + c' / d' ≤ a / (a + (a * c + 1)) + c / (c + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l1539_153935


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l1539_153971

theorem factorial_fraction_simplification (N : ℕ) (h : N > 2) :
  (Nat.factorial (N - 2) * (N - 1)^2) / Nat.factorial (N + 2) =
  (N - 1) / (N * (N + 1) * (N + 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l1539_153971


namespace NUMINAMATH_CALUDE_four_lines_equal_angles_l1539_153928

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A rectangular box in 3D space -/
structure RectangularBox where
  corner : Point3D
  width : ℝ
  length : ℝ
  height : ℝ

/-- The angle between a line and an edge of the box -/
def angleWithEdge (l : Line3D) (b : RectangularBox) (edge : Fin 12) : ℝ :=
  sorry

/-- A line forms equal angles with all edges of the box -/
def formsEqualAngles (l : Line3D) (b : RectangularBox) : Prop :=
  ∀ (e1 e2 : Fin 12), angleWithEdge l b e1 = angleWithEdge l b e2

/-- The main theorem -/
theorem four_lines_equal_angles (P : Point3D) (b : RectangularBox) :
  ∃! (lines : Finset Line3D), lines.card = 4 ∧ 
    ∀ l ∈ lines, l.point = P ∧ formsEqualAngles l b :=
  sorry

end NUMINAMATH_CALUDE_four_lines_equal_angles_l1539_153928


namespace NUMINAMATH_CALUDE_f_always_positive_sum_reciprocals_geq_nine_l1539_153914

-- Problem 1
def f (x : ℝ) : ℝ := x^6 - x^3 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

-- Problem 2
theorem sum_reciprocals_geq_nine {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c = 1) : 1/a + 1/b + 1/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_sum_reciprocals_geq_nine_l1539_153914


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l1539_153998

/-- Given a principal amount where the simple interest for 2 years at 5% per annum is 52,
    prove that the compound interest at 5% per annum for 2 years is 53.30 -/
theorem compound_interest_calculation (P : ℝ) : 
  (P * 5 * 2) / 100 = 52 →
  P * ((1 + 5/100)^2 - 1) = 53.30 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l1539_153998


namespace NUMINAMATH_CALUDE_quiz_min_correct_answers_l1539_153977

theorem quiz_min_correct_answers 
  (total_questions : ℕ) 
  (points_correct : ℕ) 
  (points_incorrect : ℕ) 
  (target_score : ℕ) 
  (min_correct : ℕ) :
  total_questions = 20 →
  points_correct = 10 →
  points_incorrect = 4 →
  target_score = 88 →
  min_correct = 12 →
  (∀ x : ℕ, x ≥ min_correct ↔ 
    points_correct * x - points_incorrect * (total_questions - x) ≥ target_score) :=
by sorry

end NUMINAMATH_CALUDE_quiz_min_correct_answers_l1539_153977


namespace NUMINAMATH_CALUDE_freshman_psychology_liberal_arts_percentage_l1539_153927

/-- Represents the student categories -/
inductive StudentCategory
| Freshman
| Sophomore
| Junior
| Senior

/-- Represents the schools -/
inductive School
| LiberalArts
| Science
| Business

/-- Represents the distribution of students across categories and schools -/
structure StudentDistribution where
  totalStudents : ℕ
  categoryPercentage : StudentCategory → ℚ
  schoolPercentage : StudentCategory → School → ℚ
  psychologyMajorPercentage : ℚ

/-- The given student distribution -/
def givenDistribution : StudentDistribution :=
  { totalStudents := 1000,  -- Arbitrary total, doesn't affect the percentage
    categoryPercentage := fun c => match c with
      | StudentCategory.Freshman => 2/5
      | StudentCategory.Sophomore => 3/10
      | StudentCategory.Junior => 1/5
      | StudentCategory.Senior => 1/10,
    schoolPercentage := fun c s => match c, s with
      | StudentCategory.Freshman, School.LiberalArts => 3/5
      | StudentCategory.Freshman, School.Science => 3/10
      | StudentCategory.Freshman, School.Business => 1/10
      | _, _ => 0,  -- Other percentages are not needed for this problem
    psychologyMajorPercentage := 1/2 }

theorem freshman_psychology_liberal_arts_percentage 
  (d : StudentDistribution) 
  (h1 : d.categoryPercentage StudentCategory.Freshman = 2/5)
  (h2 : d.schoolPercentage StudentCategory.Freshman School.LiberalArts = 3/5)
  (h3 : d.psychologyMajorPercentage = 1/2) :
  d.categoryPercentage StudentCategory.Freshman * 
  d.schoolPercentage StudentCategory.Freshman School.LiberalArts * 
  d.psychologyMajorPercentage = 12/100 := by
  sorry

end NUMINAMATH_CALUDE_freshman_psychology_liberal_arts_percentage_l1539_153927


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1539_153913

/-- The number of players in the team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 5

/-- The number of quadruplets that must be in the starting lineup -/
def quadruplets_in_lineup : ℕ := 3

/-- The number of ways to choose the starting lineup -/
def ways_to_choose_lineup : ℕ := Nat.choose num_quadruplets quadruplets_in_lineup * 
  Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)

theorem starting_lineup_combinations : ways_to_choose_lineup = 264 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1539_153913


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l1539_153910

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 7 * a (n + 1) - a n - 2

theorem a_is_perfect_square : ∀ n : ℕ, ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l1539_153910


namespace NUMINAMATH_CALUDE_smallest_cut_for_non_triangle_l1539_153915

theorem smallest_cut_for_non_triangle (a b c : ℝ) (ha : a = 10) (hb : b = 24) (hc : c = 26) :
  let f := fun x => (a - x) + (b - x) ≤ (c - x)
  ∃ x₀ : ℝ, x₀ = 8 ∧ (∀ x, 0 ≤ x ∧ x < a → (f x → x ≥ x₀) ∧ (x < x₀ → ¬f x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_for_non_triangle_l1539_153915


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1539_153996

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1539_153996


namespace NUMINAMATH_CALUDE_abs_inequality_exponential_inequality_l1539_153903

-- Problem 1
theorem abs_inequality (x : ℝ) :
  |x - 1| > 2 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 3 :=
sorry

-- Problem 2
theorem exponential_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) :
  a^(1 - x) < a^(x + 1) ↔ x ∈ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_exponential_inequality_l1539_153903


namespace NUMINAMATH_CALUDE_no_solution_exists_l1539_153917

theorem no_solution_exists : ¬∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧
  ∀ (n : ℕ), n > 0 → ((n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1539_153917


namespace NUMINAMATH_CALUDE_competition_score_l1539_153905

theorem competition_score (total_judges : Nat) (highest_score lowest_score avg_score : ℝ) :
  total_judges = 9 →
  highest_score = 86 →
  lowest_score = 45 →
  avg_score = 76 →
  (total_judges * avg_score - highest_score - lowest_score) / (total_judges - 2) = 79 := by
  sorry

end NUMINAMATH_CALUDE_competition_score_l1539_153905


namespace NUMINAMATH_CALUDE_aquarium_cost_is_63_l1539_153939

/-- The total cost of an aquarium after markdown and sales tax --/
def aquarium_total_cost (original_price : ℝ) (markdown_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let reduced_price := original_price * (1 - markdown_percent)
  let sales_tax := reduced_price * tax_percent
  reduced_price + sales_tax

/-- Theorem stating that the total cost of the aquarium is $63 --/
theorem aquarium_cost_is_63 :
  aquarium_total_cost 120 0.5 0.05 = 63 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_cost_is_63_l1539_153939


namespace NUMINAMATH_CALUDE_certain_number_proof_l1539_153933

/-- Given that g is the smallest positive integer such that n * g is a perfect square, 
    and g = 14, prove that n = 14 -/
theorem certain_number_proof (n : ℕ) (g : ℕ) (h1 : g = 14) 
  (h2 : ∃ m : ℕ, n * g = m^2)
  (h3 : ∀ k < g, ¬∃ m : ℕ, n * k = m^2) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1539_153933


namespace NUMINAMATH_CALUDE_circle_containing_three_points_l1539_153962

theorem circle_containing_three_points 
  (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 51) 
  (h2 : ∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) :
  ∃ (center : ℝ × ℝ), ∃ (contained_points : Finset (ℝ × ℝ)),
    contained_points ⊆ points ∧
    contained_points.card ≥ 3 ∧
    ∀ p ∈ contained_points, Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 1/7 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_containing_three_points_l1539_153962


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1539_153995

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 2
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1539_153995


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1539_153991

theorem opposite_of_negative_2023 : 
  (-(- 2023 : ℤ)) = (2023 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1539_153991


namespace NUMINAMATH_CALUDE_teacher_zhao_masks_l1539_153950

theorem teacher_zhao_masks (n : ℕ) : 
  (n / 2 * 5 + n / 2 * 7 + 25 = n / 3 * 10 + 2 * n / 3 * 7 - 35) →
  (n / 2 * 5 + n / 2 * 7 + 25 = 205) := by
  sorry

end NUMINAMATH_CALUDE_teacher_zhao_masks_l1539_153950


namespace NUMINAMATH_CALUDE_new_average_production_l1539_153942

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) :
  n = 9 →
  past_avg = 50 →
  today_prod = 100 →
  (n * past_avg + today_prod) / (n + 1) = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_average_production_l1539_153942


namespace NUMINAMATH_CALUDE_fraction_division_equality_l1539_153984

theorem fraction_division_equality : 
  (-1/42 : ℚ) / ((1/6 : ℚ) - (3/14 : ℚ) + (2/3 : ℚ) - (2/7 : ℚ)) = -1/14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l1539_153984


namespace NUMINAMATH_CALUDE_line_quadrants_l1539_153941

theorem line_quadrants (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c > 0) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (a * x₁ + b * y₁ - c = 0 ∧ x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (a * x₂ + b * y₂ - c = 0 ∧ x₂ < 0 ∧ y₂ < 0) ∧  -- Third quadrant
    (a * x₃ + b * y₃ - c = 0 ∧ x₃ > 0 ∧ y₃ < 0) :=  -- Fourth quadrant
by
  sorry

end NUMINAMATH_CALUDE_line_quadrants_l1539_153941


namespace NUMINAMATH_CALUDE_fraction_product_l1539_153952

theorem fraction_product : (2 : ℚ) / 9 * (4 : ℚ) / 5 = (8 : ℚ) / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1539_153952


namespace NUMINAMATH_CALUDE_mac_preference_l1539_153965

theorem mac_preference (total : ℕ) (no_pref : ℕ) (windows_pref : ℕ) 
  (h_total : total = 210)
  (h_no_pref : no_pref = 90)
  (h_windows_pref : windows_pref = 40)
  : ∃ (mac_pref : ℕ), 
    mac_pref = 60 ∧ 
    (total - no_pref = mac_pref + windows_pref + mac_pref / 3) :=
by sorry

end NUMINAMATH_CALUDE_mac_preference_l1539_153965


namespace NUMINAMATH_CALUDE_hour_hand_rotation_l1539_153948

/-- Represents the number of degrees in a complete rotation. -/
def complete_rotation : ℕ := 360

/-- Represents the number of hours in a day. -/
def hours_per_day : ℕ := 24

/-- Represents the number of complete rotations the hour hand makes. -/
def rotations : ℕ := 12

/-- Represents the number of days in which the rotations occur. -/
def days : ℕ := 6

/-- Calculates the number of degrees the hour hand rotates per hour. -/
def degrees_per_hour : ℚ :=
  (rotations * complete_rotation) / (days * hours_per_day)

theorem hour_hand_rotation :
  degrees_per_hour = 30 := by sorry

end NUMINAMATH_CALUDE_hour_hand_rotation_l1539_153948


namespace NUMINAMATH_CALUDE_rabbit_prob_reach_target_l1539_153956

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Defines the grid -/
def Grid := {p : Point | p.x ≤ 6 ∧ p.y ≤ 6}

/-- Defines the vertices of the grid -/
def Vertices : Set Point := {⟨0, 0⟩, ⟨0, 6⟩, ⟨6, 6⟩, ⟨6, 0⟩}

/-- Defines a valid jump on the grid -/
def ValidJump (p q : Point) : Prop :=
  (p.x = q.x ∧ (p.y + 1 = q.y ∨ q.y + 1 = p.y)) ∨
  (p.y = q.y ∧ (p.x + 1 = q.x ∨ q.x + 1 = p.x))

/-- Defines the probability of reaching (0,6) from a given point -/
noncomputable def ProbReachTarget (p : Point) : ℝ := sorry

/-- The main theorem to prove -/
theorem rabbit_prob_reach_target :
  ProbReachTarget ⟨1, 3⟩ = 1/4 := by sorry

end NUMINAMATH_CALUDE_rabbit_prob_reach_target_l1539_153956


namespace NUMINAMATH_CALUDE_xy_sum_inequality_l1539_153957

theorem xy_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y = 3) :
  (x + y ≥ 2) ∧ (x + y = 2 ↔ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_inequality_l1539_153957


namespace NUMINAMATH_CALUDE_larger_sphere_radius_l1539_153907

theorem larger_sphere_radius (r : ℝ) (n : ℕ) (h : r = 3 ∧ n = 9) :
  (n * (4 / 3 * Real.pi * r^3))^(1/3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_larger_sphere_radius_l1539_153907


namespace NUMINAMATH_CALUDE_ab_negative_sufficient_not_necessary_for_hyperbola_l1539_153906

/-- A conic section represented by the equation ax^2 + by^2 = c -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is a hyperbola -/
def is_hyperbola (conic : ConicSection) : Prop :=
  sorry -- Definition of hyperbola

/-- Theorem stating that ab < 0 is sufficient but not necessary for a hyperbola -/
theorem ab_negative_sufficient_not_necessary_for_hyperbola :
  ∀ (conic : ConicSection),
    (∀ (conic : ConicSection), conic.a * conic.b < 0 → is_hyperbola conic) ∧
    ¬(∀ (conic : ConicSection), is_hyperbola conic → conic.a * conic.b < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ab_negative_sufficient_not_necessary_for_hyperbola_l1539_153906


namespace NUMINAMATH_CALUDE_no_integer_solution_l1539_153968

theorem no_integer_solution : ¬∃ (x : ℝ), 
  (∃ (a b c : ℤ), (x - 1/x = a) ∧ (1/x - 1/(x^2 + 1) = b) ∧ (1/(x^2 + 1) - 2*x = c)) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1539_153968


namespace NUMINAMATH_CALUDE_pencil_count_l1539_153943

theorem pencil_count (pens pencils : ℕ) : 
  (5 * pencils = 6 * pens) → 
  (pencils = pens + 7) → 
  pencils = 42 := by sorry

end NUMINAMATH_CALUDE_pencil_count_l1539_153943


namespace NUMINAMATH_CALUDE_store_socks_problem_l1539_153966

theorem store_socks_problem (x y w z : ℕ) : 
  x + y + w + z = 15 →
  x + 2*y + 3*w + 4*z = 36 →
  x ≥ 1 →
  y ≥ 1 →
  w ≥ 1 →
  z ≥ 1 →
  x = 5 :=
by sorry

end NUMINAMATH_CALUDE_store_socks_problem_l1539_153966


namespace NUMINAMATH_CALUDE_homework_time_difference_prove_homework_time_difference_l1539_153992

/-- The difference in time taken to finish homework between Sarah and Samuel is 48 minutes -/
theorem homework_time_difference : ℝ → Prop :=
  fun difference =>
    let samuel_time : ℝ := 30  -- Samuel's time in minutes
    let sarah_time : ℝ := 1.3 * 60  -- Sarah's time converted to minutes
    difference = sarah_time - samuel_time ∧ difference = 48

/-- Proof of the homework time difference theorem -/
theorem prove_homework_time_difference : ∃ (difference : ℝ), homework_time_difference difference := by
  sorry

end NUMINAMATH_CALUDE_homework_time_difference_prove_homework_time_difference_l1539_153992


namespace NUMINAMATH_CALUDE_davids_biology_marks_l1539_153909

theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℕ)
  (h1 : english = 36)
  (h2 : mathematics = 35)
  (h3 : physics = 42)
  (h4 : chemistry = 57)
  (h5 : average = 45)
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) :
  biology = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l1539_153909


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1539_153959

theorem fraction_evaluation : (5 : ℝ) / (1 - 1/2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1539_153959


namespace NUMINAMATH_CALUDE_unique_angle_sin_cos_l1539_153923

theorem unique_angle_sin_cos :
  ∃! x : ℝ, 0 ≤ x ∧ x < π / 2 ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_sin_cos_l1539_153923


namespace NUMINAMATH_CALUDE_die_roll_probability_l1539_153978

def standard_die_roll := Fin 6

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_five (roll : standard_die_roll) : Prop := roll.val + 1 = 5

def probability_odd_roll : ℚ := 1/2

def probability_not_five : ℚ := 5/6

def num_rolls : ℕ := 8

theorem die_roll_probability :
  (probability_odd_roll ^ num_rolls) * (1 - probability_not_five ^ num_rolls) = 1288991/429981696 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1539_153978


namespace NUMINAMATH_CALUDE_value_of_a_l1539_153969

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (2 * x + 2) - 5

-- State the theorem
theorem value_of_a : ∃ a : ℝ, f (1/2 * a - 1) = 2 * a - 5 ∧ f a = 6 → a = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1539_153969


namespace NUMINAMATH_CALUDE_probability_ace_two_three_four_l1539_153955

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each rank (Ace, 2, 3, 4) in a standard deck -/
def cards_per_rank : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of four cards from a standard deck -/
def probability_four_card_sequence : ℚ :=
  (cards_per_rank : ℚ) / deck_size *
  (cards_per_rank : ℚ) / (deck_size - 1) *
  (cards_per_rank : ℚ) / (deck_size - 2) *
  (cards_per_rank : ℚ) / (deck_size - 3)

/-- The probability of drawing an Ace, 2, 3, and 4 in that order from a standard deck of 52 cards, without replacement, is equal to 16/405525 -/
theorem probability_ace_two_three_four : probability_four_card_sequence = 16 / 405525 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_two_three_four_l1539_153955


namespace NUMINAMATH_CALUDE_x_remaining_time_l1539_153982

-- Define the work rates and time worked
def x_rate : ℚ := 1 / 20
def y_rate : ℚ := 1 / 15
def y_time_worked : ℚ := 9

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem statement
theorem x_remaining_time (x_rate : ℚ) (y_rate : ℚ) (y_time_worked : ℚ) (total_work : ℚ) :
  x_rate = 1 / 20 →
  y_rate = 1 / 15 →
  y_time_worked = 9 →
  total_work = 1 →
  (total_work - y_rate * y_time_worked) / x_rate = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_x_remaining_time_l1539_153982


namespace NUMINAMATH_CALUDE_tan_alpha_two_l1539_153924

theorem tan_alpha_two (α : Real) (h : Real.tan α = 2) : 
  Real.tan (2 * α + π / 4) = 9 ∧ (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_l1539_153924


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1539_153901

theorem rectangle_area (square_area : Real) (rectangle_breadth : Real) : Real :=
  let square_side : Real := Real.sqrt square_area
  let circle_radius : Real := square_side
  let rectangle_length : Real := circle_radius / 4
  let rectangle_area : Real := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1225 10 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1539_153901


namespace NUMINAMATH_CALUDE_difference_of_percentages_l1539_153974

theorem difference_of_percentages : 
  (75 / 100 * 480) - (3 / 5 * (20 / 100 * 2500)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l1539_153974


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_when_intersection_equals_given_set_l1539_153949

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x : ℝ | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem m_value_when_intersection_equals_given_set :
  (∃ m : ℝ, A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4}) → 
  (∃ m : ℝ, A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4} ∧ m = 8) := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_when_intersection_equals_given_set_l1539_153949


namespace NUMINAMATH_CALUDE_mark_fruit_count_l1539_153983

/-- The number of apples Mark has chosen -/
def num_apples : ℕ := 3

/-- The number of bananas in the bunch Mark has selected -/
def num_bananas : ℕ := 4

/-- The number of oranges Mark needs to pick out -/
def num_oranges : ℕ := 5

/-- The total number of pieces of fruit Mark is looking to buy -/
def total_fruit : ℕ := num_apples + num_bananas + num_oranges

theorem mark_fruit_count : total_fruit = 12 := by
  sorry

end NUMINAMATH_CALUDE_mark_fruit_count_l1539_153983


namespace NUMINAMATH_CALUDE_total_outfits_l1539_153951

/-- Represents the number of shirts available. -/
def num_shirts : ℕ := 7

/-- Represents the number of ties available. -/
def num_ties : ℕ := 5

/-- Represents the number of pairs of pants available. -/
def num_pants : ℕ := 4

/-- Represents the number of shoe types available. -/
def num_shoe_types : ℕ := 2

/-- Calculates the number of outfit combinations with a tie. -/
def outfits_with_tie : ℕ := num_shirts * num_pants * num_ties

/-- Calculates the number of outfit combinations without a tie. -/
def outfits_without_tie : ℕ := num_shirts * num_pants

/-- Theorem stating the total number of different outfits. -/
theorem total_outfits : outfits_with_tie + outfits_without_tie = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_outfits_l1539_153951


namespace NUMINAMATH_CALUDE_app_difference_proof_l1539_153902

/-- Calculates the difference between added and deleted apps -/
def appDifference (initial final added : ℕ) : ℕ :=
  added - ((initial + added) - final)

theorem app_difference_proof (initial final added : ℕ) 
  (h1 : initial = 21)
  (h2 : final = 24)
  (h3 : added = 89) :
  appDifference initial final added = 3 := by
  sorry

#eval appDifference 21 24 89

end NUMINAMATH_CALUDE_app_difference_proof_l1539_153902


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1539_153964

theorem vector_equation_solution :
  let c₁ : ℚ := 5/6
  let c₂ : ℚ := -7/18
  let v₁ : Fin 2 → ℚ := ![1, 4]
  let v₂ : Fin 2 → ℚ := ![-3, 6]
  let result : Fin 2 → ℚ := ![2, 1]
  c₁ • v₁ + c₂ • v₂ = result :=
by
  sorry

#check vector_equation_solution

end NUMINAMATH_CALUDE_vector_equation_solution_l1539_153964


namespace NUMINAMATH_CALUDE_randys_pig_feed_per_week_l1539_153904

/-- Calculates the amount of pig feed needed per week given the daily feed per pig, number of pigs, and days in a week. -/
def pig_feed_per_week (feed_per_pig_per_day : ℕ) (num_pigs : ℕ) (days_in_week : ℕ) : ℕ :=
  feed_per_pig_per_day * num_pigs * days_in_week

/-- Proves that Randy's pigs will be fed 140 pounds of pig feed per week. -/
theorem randys_pig_feed_per_week :
  let feed_per_pig_per_day : ℕ := 10
  let num_pigs : ℕ := 2
  let days_in_week : ℕ := 7
  pig_feed_per_week feed_per_pig_per_day num_pigs days_in_week = 140 := by
  sorry

end NUMINAMATH_CALUDE_randys_pig_feed_per_week_l1539_153904


namespace NUMINAMATH_CALUDE_ellipse_and_point_G_l1539_153973

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given an ellipse C and points on it, prove the equation and find point G -/
theorem ellipse_and_point_G (C : Ellipse) 
  (h_triangle_area : (1/2) * C.a * (C.a^2 - C.b^2).sqrt * C.a = 4 * Real.sqrt 3)
  (B : Point) (h_B_on_C : B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1)
  (h_B_nonzero : B.x * B.y ≠ 0)
  (A : Point) (h_A : A = ⟨0, 2 * Real.sqrt 3⟩)
  (D E : Point) (h_D : D.y = 0) (h_E : E.y = 0)
  (h_collinear_ABD : (A.y - D.y) / (A.x - D.x) = (B.y - D.y) / (B.x - D.x))
  (h_collinear_ABE : (A.y - E.y) / (A.x - E.x) = (B.y + E.y) / (B.x - E.x))
  (G : Point) (h_G : G.x = 0)
  (h_angle_equal : (G.y / D.x)^2 = (G.y / E.x)^2) :
  (C.a = 4 ∧ C.b = 2 * Real.sqrt 3) ∧ 
  (G.y = 4 ∨ G.y = -4) := by sorry

end NUMINAMATH_CALUDE_ellipse_and_point_G_l1539_153973


namespace NUMINAMATH_CALUDE_problem_statement_l1539_153925

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^2 / y + y^2 / x + y = 95 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1539_153925


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1539_153985

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 1 = 0) → (x₂^2 + 5*x₂ - 1 = 0) → (x₁ + x₂ = -5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1539_153985


namespace NUMINAMATH_CALUDE_tree_planting_participants_l1539_153946

theorem tree_planting_participants : ∃ (x y : ℕ), 
  x * y = 2013 ∧ 
  (x - 5) * (y + 2) < 2013 ∧ 
  (x - 5) * (y + 3) > 2013 ∧ 
  x = 61 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_participants_l1539_153946


namespace NUMINAMATH_CALUDE_quadrilateral_formation_l1539_153932

/-- A function that checks if four line segments can form a quadrilateral --/
def can_form_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- The theorem stating which set of line segments can form a quadrilateral with length 5 --/
theorem quadrilateral_formation :
  ¬(can_form_quadrilateral 1 1 1 5) ∧
  ¬(can_form_quadrilateral 1 1 8 5) ∧
  ¬(can_form_quadrilateral 1 2 2 5) ∧
  can_form_quadrilateral 3 3 3 5 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_formation_l1539_153932


namespace NUMINAMATH_CALUDE_remainder_double_number_l1539_153970

theorem remainder_double_number (N : ℤ) : 
  N % 398 = 255 → (2 * N) % 398 = 112 := by
sorry

end NUMINAMATH_CALUDE_remainder_double_number_l1539_153970


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1539_153972

-- Define the polynomial
def P (k X : ℝ) : ℝ := X^4 + 2*X^3 + (2 + 2*k)*X^2 + (1 + 2*k)*X + 2*k

-- Define the theorem
theorem sum_of_squares_of_roots (k : ℝ) :
  (∃ r₁ r₂ : ℝ, P k r₁ = 0 ∧ P k r₂ = 0 ∧ r₁ * r₂ = -2013) →
  (∃ r₁ r₂ : ℝ, P k r₁ = 0 ∧ P k r₂ = 0 ∧ r₁^2 + r₂^2 = 4027) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1539_153972


namespace NUMINAMATH_CALUDE_bathroom_tile_side_length_l1539_153936

-- Define the dimensions of the bathroom
def bathroom_length : ℝ := 6
def bathroom_width : ℝ := 10

-- Define the number of tiles
def number_of_tiles : ℕ := 240

-- Define the side length of a tile
def tile_side_length : ℝ := 0.5

-- Theorem statement
theorem bathroom_tile_side_length :
  bathroom_length * bathroom_width = (number_of_tiles : ℝ) * tile_side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_bathroom_tile_side_length_l1539_153936


namespace NUMINAMATH_CALUDE_binomial_12_6_l1539_153981

theorem binomial_12_6 : Nat.choose 12 6 = 1848 := by sorry

end NUMINAMATH_CALUDE_binomial_12_6_l1539_153981


namespace NUMINAMATH_CALUDE_calculation_product_l1539_153921

theorem calculation_product (x : ℤ) (h : x - 9 - 12 = 24) : (x + 8 - 11) * 24 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_calculation_product_l1539_153921


namespace NUMINAMATH_CALUDE_combined_girls_avg_is_87_l1539_153976

/-- Represents a high school with average scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined average score for boys at both schools -/
def combined_boys_avg : ℝ := 73

/-- Theorem: The combined average score for girls at both schools is 87 -/
theorem combined_girls_avg_is_87 
  (lincoln : School)
  (grant : School)
  (h1 : lincoln.boys_avg = 68)
  (h2 : lincoln.girls_avg = 80)
  (h3 : lincoln.combined_avg = 72)
  (h4 : grant.boys_avg = 75)
  (h5 : grant.girls_avg = 88)
  (h6 : grant.combined_avg = 82)
  (h7 : combined_boys_avg = 73) :
  ∃ (combined_girls_avg : ℝ), combined_girls_avg = 87 := by
  sorry


end NUMINAMATH_CALUDE_combined_girls_avg_is_87_l1539_153976


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1539_153918

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1539_153918


namespace NUMINAMATH_CALUDE_expression_evaluation_l1539_153967

theorem expression_evaluation (x : ℝ) (h : x = 1) : 
  (x - 1)^2 + (x + 1)*(x - 1) - 2*x^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1539_153967


namespace NUMINAMATH_CALUDE_candy_mixture_price_l1539_153989

theorem candy_mixture_price (total_mixture : ℝ) (mixture_price : ℝ) 
  (first_candy_amount : ℝ) (second_candy_amount : ℝ) (first_candy_price : ℝ) :
  total_mixture = 30 ∧
  mixture_price = 3 ∧
  first_candy_amount = 20 ∧
  second_candy_amount = 10 ∧
  first_candy_price = 2.95 →
  ∃ (second_candy_price : ℝ),
    second_candy_price = 3.10 ∧
    total_mixture * mixture_price = 
      first_candy_amount * first_candy_price + second_candy_amount * second_candy_price :=
by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l1539_153989


namespace NUMINAMATH_CALUDE_katie_earnings_l1539_153999

/-- Calculates the total money earned from selling necklaces -/
def total_money_earned (bead_necklaces gem_necklaces price_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_necklaces) * price_per_necklace

/-- Proves that Katie earned 21 dollars from selling her necklaces -/
theorem katie_earnings : 
  let bead_necklaces : ℕ := 4
  let gem_necklaces : ℕ := 3
  let price_per_necklace : ℕ := 3
  total_money_earned bead_necklaces gem_necklaces price_per_necklace = 21 := by
sorry

end NUMINAMATH_CALUDE_katie_earnings_l1539_153999


namespace NUMINAMATH_CALUDE_amount_ratio_l1539_153922

theorem amount_ratio (total : ℚ) (r_amount : ℚ) 
  (h1 : total = 9000)
  (h2 : r_amount = 3600.0000000000005) :
  r_amount / (total - r_amount) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_amount_ratio_l1539_153922


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1539_153934

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 10th term of the specific arithmetic sequence -/
theorem tenth_term_of_sequence : 
  arithmeticSequenceTerm 3 6 10 = 57 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1539_153934


namespace NUMINAMATH_CALUDE_function_difference_bound_l1539_153916

/-- Given a function f(x) = x^2 - x + c and a real number a such that |x - a| < 1,
    prove that |f(x) - f(a)| < 2(|a| + 1) -/
theorem function_difference_bound (c a x : ℝ) (h : |x - a| < 1) :
  let f := fun (t : ℝ) => t^2 - t + c
  |f x - f a| < 2 * (|a| + 1) := by
sorry


end NUMINAMATH_CALUDE_function_difference_bound_l1539_153916


namespace NUMINAMATH_CALUDE_problem_solution_l1539_153960

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

theorem problem_solution :
  (∀ a : ℝ, a = 3 → M ∪ (Nᶜ a) = Set.univ) ∧
  (∀ a : ℝ, N a ⊆ M ↔ a ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1539_153960


namespace NUMINAMATH_CALUDE_horner_method_v3_l1539_153997

def horner_polynomial (x : ℝ) : ℝ := 1 + 5*x + 10*x^2 + 10*x^3 + 5*x^4 + x^5

def horner_v1 (x : ℝ) : ℝ := x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 10

def horner_v3 (x : ℝ) : ℝ := horner_v2 x * x + 10

theorem horner_method_v3 :
  horner_v3 (-2) = 2 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1539_153997


namespace NUMINAMATH_CALUDE_initial_boys_count_l1539_153911

theorem initial_boys_count (initial_total : ℕ) (initial_boys : ℕ) (final_boys : ℕ) : 
  initial_boys = initial_total / 2 →                   -- Initially, 50% are boys
  final_boys = initial_boys - 3 →                      -- 3 boys leave
  final_boys * 10 = 4 * initial_total →                -- After changes, 40% are boys
  initial_boys = 15 := by
sorry

end NUMINAMATH_CALUDE_initial_boys_count_l1539_153911


namespace NUMINAMATH_CALUDE_slope_of_line_with_30_degree_inclination_l1539_153937

theorem slope_of_line_with_30_degree_inclination :
  let angle_of_inclination : ℝ := 30 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_slope_of_line_with_30_degree_inclination_l1539_153937


namespace NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l1539_153900

/-- The shortest distance from a point on the curve y = ln x to the line 2x - y + 3 = 0 -/
theorem shortest_distance_ln_to_line : ∃ (d : ℝ), d = (4 + Real.log 2) / Real.sqrt 5 ∧
  ∀ (x y : ℝ), y = Real.log x →
    d ≤ (|2 * x - y + 3|) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l1539_153900


namespace NUMINAMATH_CALUDE_wage_payment_days_l1539_153919

theorem wage_payment_days (S : ℝ) (hX : S > 0) (hY : S > 0) : 
  (∃ (wX wY : ℝ), wX > 0 ∧ wY > 0 ∧ S = 36 * wX ∧ S = 45 * wY) →
  ∃ (d : ℝ), d = 20 ∧ S = d * (S / 36 + S / 45) :=
by sorry

end NUMINAMATH_CALUDE_wage_payment_days_l1539_153919


namespace NUMINAMATH_CALUDE_optimal_profit_profit_function_correct_l1539_153980

/-- Represents the daily profit function for a factory -/
def daily_profit (x : ℝ) : ℝ := -50 * x^2 + 400 * x + 9000

/-- Represents the optimal price reduction -/
def optimal_reduction : ℝ := 4

/-- Represents the maximum daily profit -/
def max_profit : ℝ := 9800

/-- Theorem stating the optimal price reduction and maximum profit -/
theorem optimal_profit :
  (∀ x : ℝ, daily_profit x ≤ daily_profit optimal_reduction) ∧
  daily_profit optimal_reduction = max_profit := by
  sorry

/-- Theorem stating the correctness of the daily profit function -/
theorem profit_function_correct
  (cost_per_kg : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (sales_increase_rate : ℝ)
  (h1 : cost_per_kg = 30)
  (h2 : initial_price = 48)
  (h3 : initial_sales = 500)
  (h4 : sales_increase_rate = 50) :
  ∀ x : ℝ, daily_profit x =
    (initial_price - x - cost_per_kg) * (initial_sales + sales_increase_rate * x) := by
  sorry

end NUMINAMATH_CALUDE_optimal_profit_profit_function_correct_l1539_153980


namespace NUMINAMATH_CALUDE_tree_planting_problem_l1539_153994

theorem tree_planting_problem (n t : ℕ) 
  (h1 : 4 * n = t + 11) 
  (h2 : 2 * n = t - 13) : 
  n = 12 ∧ t = 37 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l1539_153994
