import Mathlib

namespace NUMINAMATH_CALUDE_national_lipstick_day_attendance_l2417_241708

theorem national_lipstick_day_attendance (total_students : ℕ) : 
  (total_students : ℚ) / 2 = (total_students : ℚ) / 2 / 4 * 5 + 5 → total_students = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_national_lipstick_day_attendance_l2417_241708


namespace NUMINAMATH_CALUDE_triple_equation_solution_l2417_241777

theorem triple_equation_solution :
  ∀ a b c : ℝ,
    a + b + c = 14 ∧
    a^2 + b^2 + c^2 = 84 ∧
    a^3 + b^3 + c^3 = 584 →
    ((a = 4 ∧ b = 2 ∧ c = 8) ∨
     (a = 2 ∧ b = 4 ∧ c = 8) ∨
     (a = 8 ∧ b = 2 ∧ c = 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_equation_solution_l2417_241777


namespace NUMINAMATH_CALUDE_unique_all_ones_polynomial_l2417_241785

def is_all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ+, n = (10^k.val - 1) / 9

def polynomial_all_ones (P : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_all_ones n → is_all_ones (P n)

theorem unique_all_ones_polynomial :
  ∀ P : ℕ → ℕ, polynomial_all_ones P → P = id := by sorry

end NUMINAMATH_CALUDE_unique_all_ones_polynomial_l2417_241785


namespace NUMINAMATH_CALUDE_max_sum_composite_shape_l2417_241734

/-- Represents a composite shape formed by adding a pyramid to a pentagonal prism --/
structure CompositePrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_faces : Nat
  pyramid_edges : Nat
  pyramid_vertices : Nat

/-- The total number of faces in the composite shape --/
def total_faces (shape : CompositePrismPyramid) : Nat :=
  shape.prism_faces + shape.pyramid_faces - 1

/-- The total number of edges in the composite shape --/
def total_edges (shape : CompositePrismPyramid) : Nat :=
  shape.prism_edges + shape.pyramid_edges

/-- The total number of vertices in the composite shape --/
def total_vertices (shape : CompositePrismPyramid) : Nat :=
  shape.prism_vertices + shape.pyramid_vertices

/-- The sum of faces, edges, and vertices in the composite shape --/
def total_sum (shape : CompositePrismPyramid) : Nat :=
  total_faces shape + total_edges shape + total_vertices shape

/-- Theorem stating the maximum sum of faces, edges, and vertices --/
theorem max_sum_composite_shape :
  ∃ (shape : CompositePrismPyramid),
    shape.prism_faces = 7 ∧
    shape.prism_edges = 15 ∧
    shape.prism_vertices = 10 ∧
    shape.pyramid_faces = 5 ∧
    shape.pyramid_edges = 5 ∧
    shape.pyramid_vertices = 1 ∧
    total_sum shape = 42 ∧
    ∀ (other : CompositePrismPyramid), total_sum other ≤ 42 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_composite_shape_l2417_241734


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l2417_241731

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (4 * a) % 65 = 1 ∧ 
                (13 * b) % 65 = 1 ∧ 
                (3 * a + 12 * b) % 65 = 42 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l2417_241731


namespace NUMINAMATH_CALUDE_remainder_property_l2417_241717

theorem remainder_property (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : 0 ≤ v) (h5 : v < y) : 
  (x + 3 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_property_l2417_241717


namespace NUMINAMATH_CALUDE_negation_existence_sufficient_not_necessary_sufficient_necessary_relationship_quadratic_inequality_condition_l2417_241790

-- 1. Negation of existence statement
theorem negation_existence : 
  (¬ ∃ x : ℝ, x ≥ 1 ∧ x^2 > 1) ↔ (∀ x : ℝ, x ≥ 1 → x^2 ≤ 1) := by sorry

-- 2. Sufficient but not necessary condition
theorem sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 + 2*x - 3 = 0) ∧
  (∀ x : ℝ, x = 1 → x^2 + 2*x - 3 = 0) := by sorry

-- 3. Relationship between sufficient and necessary conditions
theorem sufficient_necessary_relationship (p q s : Prop) :
  ((p → q) ∧ (q → s)) → (p → s) := by sorry

-- 4. Conditions for quadratic inequality
theorem quadratic_inequality_condition (m : ℝ) :
  (¬ ∃ x : ℝ, m*x^2 + m*x + 1 < 0) → (0 ≤ m ∧ m ≤ 4) := by sorry

end NUMINAMATH_CALUDE_negation_existence_sufficient_not_necessary_sufficient_necessary_relationship_quadratic_inequality_condition_l2417_241790


namespace NUMINAMATH_CALUDE_phi_value_l2417_241761

theorem phi_value (φ : Real) (h1 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos φ - Real.sin φ)
  (h2 : 0 < φ ∧ φ < π / 2) : φ = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l2417_241761


namespace NUMINAMATH_CALUDE_gym_students_count_l2417_241728

theorem gym_students_count :
  ∀ (students_on_floor : ℕ) (total_students : ℕ),
    -- 4 students are on the bleachers
    total_students = students_on_floor + 4 →
    -- The ratio of students on the floor to total students is 11:13
    (students_on_floor : ℚ) / total_students = 11 / 13 →
    -- The total number of students is 26
    total_students = 26 := by
  sorry

end NUMINAMATH_CALUDE_gym_students_count_l2417_241728


namespace NUMINAMATH_CALUDE_task_completion_time_l2417_241725

/-- The time taken to complete a task when two people work together, with one person stopping early. -/
def completionTime (john_rate : ℚ) (jane_rate : ℚ) (early_stop : ℕ) : ℚ :=
  let combined_rate := john_rate + jane_rate
  let x := (1 - john_rate * early_stop) / combined_rate
  x + early_stop

theorem task_completion_time :
  let john_rate : ℚ := 1 / 20
  let jane_rate : ℚ := 1 / 12
  let early_stop : ℕ := 4
  completionTime john_rate jane_rate early_stop = 10 := by
  sorry

#eval completionTime (1 / 20 : ℚ) (1 / 12 : ℚ) 4

end NUMINAMATH_CALUDE_task_completion_time_l2417_241725


namespace NUMINAMATH_CALUDE_smallest_multiple_three_is_solution_three_is_smallest_l2417_241792

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 675 ∣ (450 * x) → x ≥ 3 :=
sorry

theorem three_is_solution : 675 ∣ (450 * 3) :=
sorry

theorem three_is_smallest : ∀ y : ℕ, y > 0 ∧ y < 3 → ¬(675 ∣ (450 * y)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_three_is_solution_three_is_smallest_l2417_241792


namespace NUMINAMATH_CALUDE_gerbil_revenue_calculation_l2417_241797

/-- Calculates the total revenue from gerbil sales given the initial stock, percentage sold, original price, and discount rate. -/
def gerbil_revenue (initial_stock : ℕ) (percent_sold : ℚ) (original_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let sold := ⌊initial_stock * percent_sold⌋
  let remaining := initial_stock - sold
  let discounted_price := original_price * (1 - discount_rate)
  sold * original_price + remaining * discounted_price

/-- Theorem stating that the total revenue from gerbil sales is $4696.80 given the specified conditions. -/
theorem gerbil_revenue_calculation :
  gerbil_revenue 450 (35/100) 12 (20/100) = 4696.80 := by
  sorry

end NUMINAMATH_CALUDE_gerbil_revenue_calculation_l2417_241797


namespace NUMINAMATH_CALUDE_madeline_score_is_28_l2417_241744

/-- Represents the score and mistakes in a Geometry exam -/
structure GeometryExam where
  totalQuestions : ℕ
  scorePerQuestion : ℕ
  madelineMistakes : ℕ
  leoMistakes : ℕ
  brentMistakes : ℕ
  brentScore : ℕ

/-- Calculates Madeline's score in the Geometry exam -/
def madelineScore (exam : GeometryExam) : ℕ :=
  exam.totalQuestions * exam.scorePerQuestion - exam.madelineMistakes * exam.scorePerQuestion

/-- Theorem: Given the conditions, Madeline's score in the Geometry exam is 28 -/
theorem madeline_score_is_28 (exam : GeometryExam)
  (h1 : exam.madelineMistakes = 2)
  (h2 : exam.leoMistakes = 2 * exam.madelineMistakes)
  (h3 : exam.brentMistakes = exam.leoMistakes + 1)
  (h4 : exam.brentScore = 25)
  (h5 : exam.totalQuestions = exam.brentScore + exam.brentMistakes)
  (h6 : exam.scorePerQuestion = 1) :
  madelineScore exam = 28 := by
  sorry


end NUMINAMATH_CALUDE_madeline_score_is_28_l2417_241744


namespace NUMINAMATH_CALUDE_g_of_3_equals_135_l2417_241724

/-- Given that g(x) = 3x^4 - 5x^3 + 2x^2 + x + 6, prove that g(3) = 135 -/
theorem g_of_3_equals_135 : 
  let g : ℝ → ℝ := λ x ↦ 3*x^4 - 5*x^3 + 2*x^2 + x + 6
  g 3 = 135 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_135_l2417_241724


namespace NUMINAMATH_CALUDE_average_side_lengths_of_squares_l2417_241730

theorem average_side_lengths_of_squares (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 36) (h₃ : a₃ = 64) (h₄ : a₄ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃ + Real.sqrt a₄) / 4 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_average_side_lengths_of_squares_l2417_241730


namespace NUMINAMATH_CALUDE_broken_line_path_length_l2417_241750

/-- Given a circle with diameter 12 units and points C, D each 3 units from the endpoints of the diameter,
    the length of the path CPD is 6√5 units for any point P on the circle forming a right angle CPD. -/
theorem broken_line_path_length (O : ℝ × ℝ) (A B C D P : ℝ × ℝ) : 
  let r : ℝ := 6 -- radius of the circle
  dist A B = 12 ∧ -- diameter is 12 units
  dist A C = 3 ∧ -- C is 3 units from A
  dist B D = 3 ∧ -- D is 3 units from B
  dist O P = r ∧ -- P is on the circle
  (C.1 - P.1) * (D.1 - P.1) + (C.2 - P.2) * (D.2 - P.2) = 0 -- angle CPD is right angle
  →
  dist C P + dist P D = 6 * Real.sqrt 5 := by
  sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x, y) (a, b) ↦ Real.sqrt ((x - a)^2 + (y - b)^2)

end NUMINAMATH_CALUDE_broken_line_path_length_l2417_241750


namespace NUMINAMATH_CALUDE_hexagon_angles_arithmetic_progression_l2417_241716

theorem hexagon_angles_arithmetic_progression :
  ∃ (a d : ℝ), 
    (∀ i : Fin 6, 0 ≤ a + i * d ∧ a + i * d ≤ 180) ∧ 
    (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 720) ∧
    (∃ i : Fin 6, a + i * d = 120) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angles_arithmetic_progression_l2417_241716


namespace NUMINAMATH_CALUDE_stating_arithmetic_sequence_length_is_twelve_l2417_241736

/-- 
The number of terms in an arithmetic sequence with 
first term 3, last term 69, and common difference 6
-/
def arithmetic_sequence_length : ℕ := 
  (69 - 3) / 6 + 1

/-- 
Theorem stating that the arithmetic sequence length is 12
-/
theorem arithmetic_sequence_length_is_twelve : 
  arithmetic_sequence_length = 12 := by sorry

end NUMINAMATH_CALUDE_stating_arithmetic_sequence_length_is_twelve_l2417_241736


namespace NUMINAMATH_CALUDE_folded_paper_height_approx_l2417_241719

/-- The height of a folded sheet of paper -/
def folded_paper_height (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

/-- Approximation of 2^10 -/
def approx_2_10 : ℝ := 1000

theorem folded_paper_height_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |folded_paper_height 0.1 20 - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_folded_paper_height_approx_l2417_241719


namespace NUMINAMATH_CALUDE_james_driving_distance_l2417_241727

/-- Calculates the total distance driven given multiple segments of a trip -/
def total_distance (speeds : List ℝ) (times : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) speeds times)

/-- James' driving problem -/
theorem james_driving_distance :
  let speeds : List ℝ := [30, 60, 75, 60]
  let times : List ℝ := [0.5, 0.75, 1.5, 2]
  total_distance speeds times = 292.5 := by
  sorry

end NUMINAMATH_CALUDE_james_driving_distance_l2417_241727


namespace NUMINAMATH_CALUDE_system_solution_l2417_241703

theorem system_solution : ∃ (X Y : ℝ), 
  (X^2 * Y^2 + X * Y^2 + X^2 * Y + X * Y + X + Y + 3 = 0) ∧ 
  (X^2 * Y + X * Y + 1 = 0) ∧ 
  (X = -2) ∧ (Y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2417_241703


namespace NUMINAMATH_CALUDE_weekday_hours_are_six_l2417_241733

/-- Represents the daily weekday operation hours of Jean's business -/
def weekday_hours : ℝ := sorry

/-- The number of weekdays the business operates -/
def weekdays : ℕ := 5

/-- The number of hours the business operates each day on weekends -/
def weekend_daily_hours : ℕ := 4

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The total weekly operation hours -/
def total_weekly_hours : ℕ := 38

/-- Theorem stating that the daily weekday operation hours are 6 -/
theorem weekday_hours_are_six : weekday_hours = 6 := by sorry

end NUMINAMATH_CALUDE_weekday_hours_are_six_l2417_241733


namespace NUMINAMATH_CALUDE_trig_sum_equals_sqrt_two_l2417_241740

theorem trig_sum_equals_sqrt_two : 
  Real.tan (60 * π / 180) + 2 * Real.sin (45 * π / 180) - 2 * Real.cos (30 * π / 180) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_sqrt_two_l2417_241740


namespace NUMINAMATH_CALUDE_p_and_q_implies_p_or_q_l2417_241786

theorem p_and_q_implies_p_or_q (p q : Prop) : (p ∧ q) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_implies_p_or_q_l2417_241786


namespace NUMINAMATH_CALUDE_min_sum_at_24_l2417_241707

/-- Arithmetic sequence with general term a_n = 2n - 49 -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

theorem min_sum_at_24 :
  ∀ k : ℕ, k ≠ 0 → S 24 ≤ S k :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_24_l2417_241707


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l2417_241758

/-- Given two functions f and g with identical symmetry axes, 
    prove that [0, π/8] is an increasing interval of f on [0, π] -/
theorem increasing_interval_of_f (ω : ℝ) (h_ω : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + π / 4)
  let g : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * x - π / 4)
  (∀ x y : ℝ, f x = f y ↔ g x = g y) →  -- Symmetry axes are identical
  (∀ x y : ℝ, x ∈ Set.Icc 0 (π / 8) → y ∈ Set.Icc 0 (π / 8) → x < y → f x < f y) ∧
  Set.Icc 0 (π / 8) ⊆ Set.Icc 0 π :=
by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l2417_241758


namespace NUMINAMATH_CALUDE_decimal_order_l2417_241737

theorem decimal_order : 0.6 < 0.67 ∧ 0.67 < 0.676 ∧ 0.676 < 0.677 := by
  sorry

end NUMINAMATH_CALUDE_decimal_order_l2417_241737


namespace NUMINAMATH_CALUDE_sunday_game_revenue_proof_l2417_241756

def sunday_game_revenue (total_revenue : ℚ) (revenue_difference : ℚ) : ℚ :=
  (total_revenue + revenue_difference) / 2

theorem sunday_game_revenue_proof (total_revenue revenue_difference : ℚ) 
  (h1 : total_revenue = 4994.50)
  (h2 : revenue_difference = 1330.50) :
  sunday_game_revenue total_revenue revenue_difference = 3162.50 := by
  sorry

end NUMINAMATH_CALUDE_sunday_game_revenue_proof_l2417_241756


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2417_241763

theorem trigonometric_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2417_241763


namespace NUMINAMATH_CALUDE_sum_of_E_3_and_4_l2417_241749

/-- Given a function E: ℝ → ℝ where E(3) = 5 and E(4) = 5, prove that E(3) + E(4) = 10 -/
theorem sum_of_E_3_and_4 (E : ℝ → ℝ) (h1 : E 3 = 5) (h2 : E 4 = 5) : E 3 + E 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_E_3_and_4_l2417_241749


namespace NUMINAMATH_CALUDE_ariel_birth_year_l2417_241772

/-- Calculates the birth year of a person given their fencing start year, years of fencing, and current age. -/
def birth_year (fencing_start_year : ℕ) (years_fencing : ℕ) (current_age : ℕ) : ℕ :=
  fencing_start_year - (current_age - years_fencing)

/-- Proves that Ariel's birth year is 1992 given the provided conditions. -/
theorem ariel_birth_year :
  let fencing_start_year : ℕ := 2006
  let years_fencing : ℕ := 16
  let current_age : ℕ := 30
  birth_year fencing_start_year years_fencing current_age = 1992 := by
  sorry

#eval birth_year 2006 16 30

end NUMINAMATH_CALUDE_ariel_birth_year_l2417_241772


namespace NUMINAMATH_CALUDE_infinitely_many_triplets_sum_of_squares_l2417_241735

theorem infinitely_many_triplets_sum_of_squares :
  ∃ f : ℕ → ℤ, ∀ k : ℕ,
    (∃ a b : ℤ, f k = a^2 + b^2) ∧
    (∃ c d : ℤ, f k + 1 = c^2 + d^2) ∧
    (∃ e g : ℤ, f k + 2 = e^2 + g^2) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_triplets_sum_of_squares_l2417_241735


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2417_241779

theorem cube_root_equation_solution (x : ℝ) :
  (7 * x * (x^2)^(1/2))^(1/3) = 5 → x = 5 * (35^(1/2)) / 7 ∨ x = -5 * (35^(1/2)) / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2417_241779


namespace NUMINAMATH_CALUDE_march_starts_on_friday_l2417_241773

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its properties -/
structure Month where
  days : Nat
  first_day : Weekday
  monday_count : Nat
  thursday_count : Nat

/-- The specific March we're considering -/
def march : Month :=
  { days := 31
  , first_day := Weekday.Friday  -- This is what we want to prove
  , monday_count := 5
  , thursday_count := 5 }

/-- Main theorem: If March has 31 days, 5 Mondays, and 5 Thursdays, then it starts on a Friday -/
theorem march_starts_on_friday :
  march.days = 31 ∧ march.monday_count = 5 ∧ march.thursday_count = 5 →
  march.first_day = Weekday.Friday :=
sorry

end NUMINAMATH_CALUDE_march_starts_on_friday_l2417_241773


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2417_241782

theorem discount_percentage_calculation (original_price : ℝ) 
  (john_tip_rate : ℝ) (jane_tip_rate : ℝ) (price_difference : ℝ) 
  (h1 : original_price = 24.00000000000002)
  (h2 : john_tip_rate = 0.15)
  (h3 : jane_tip_rate = 0.15)
  (h4 : price_difference = 0.36)
  (h5 : original_price * (1 + john_tip_rate) - 
        original_price * (1 - D) * (1 + jane_tip_rate) = price_difference) :
  D = price_difference / (original_price * (1 + john_tip_rate)) := by
sorry

#eval (0.36 / 27.600000000000024 : Float)

end NUMINAMATH_CALUDE_discount_percentage_calculation_l2417_241782


namespace NUMINAMATH_CALUDE_three_propositions_true_l2417_241705

-- Define the properties of functions
def IsConstant (f : ℝ → ℝ) : Prop := ∃ C : ℝ, ∀ x : ℝ, f x = C
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def HasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- Define the propositions
def Prop1 (f : ℝ → ℝ) : Prop := IsConstant f → (IsOdd f ∧ IsEven f)
def Prop2 (f : ℝ → ℝ) : Prop := IsOdd f → HasInverse f
def Prop3 (f : ℝ → ℝ) : Prop := IsOdd f → IsOdd (λ x => Real.sin (f x))
def Prop4 (f g : ℝ → ℝ) : Prop := IsOdd f → IsEven g → IsEven (g ∘ f)

-- The main theorem
theorem three_propositions_true :
  ∃ (f g : ℝ → ℝ),
    (Prop1 f ∧ ¬Prop2 f ∧ Prop3 f ∧ Prop4 f g) ∨
    (Prop1 f ∧ Prop2 f ∧ Prop3 f ∧ ¬Prop4 f g) ∨
    (Prop1 f ∧ Prop2 f ∧ ¬Prop3 f ∧ Prop4 f g) ∨
    (¬Prop1 f ∧ Prop2 f ∧ Prop3 f ∧ Prop4 f g) :=
  sorry

end NUMINAMATH_CALUDE_three_propositions_true_l2417_241705


namespace NUMINAMATH_CALUDE_symmetric_points_y_coordinate_l2417_241726

/-- Given two points P and Q in a 2D Cartesian coordinate system that are symmetric about the origin,
    prove that the y-coordinate of Q is -3. -/
theorem symmetric_points_y_coordinate
  (P Q : ℝ × ℝ)  -- P and Q are points in 2D real space
  (h_P : P = (-3, 5))  -- Coordinates of P
  (h_Q : Q.1 = 3 ∧ Q.2 = m - 2)  -- x-coordinate of Q is 3, y-coordinate is m-2
  (h_sym : P.1 = -Q.1 ∧ P.2 = -Q.2)  -- P and Q are symmetric about the origin
  : m = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_y_coordinate_l2417_241726


namespace NUMINAMATH_CALUDE_base5_polynomial_representation_l2417_241760

/-- Represents a polynomial in base 5 with coefficients less than 5 -/
def Base5Polynomial (coeffs : List Nat) : Prop :=
  coeffs.all (· < 5) ∧ coeffs.length > 0

/-- Converts a list of coefficients to a natural number in base 5 -/
def toBase5Number (coeffs : List Nat) : Nat :=
  coeffs.foldl (fun acc d => 5 * acc + d) 0

/-- Theorem: A polynomial with coefficients less than 5 can be uniquely represented as a base-5 number -/
theorem base5_polynomial_representation 
  (coeffs : List Nat) 
  (h : Base5Polynomial coeffs) :
  ∃! n : Nat, n = toBase5Number coeffs :=
sorry

end NUMINAMATH_CALUDE_base5_polynomial_representation_l2417_241760


namespace NUMINAMATH_CALUDE_exactly_three_two_digit_multiples_l2417_241798

theorem exactly_three_two_digit_multiples :
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, x > 0 ∧ (∃! (m : Finset ℕ), 
      (∀ y ∈ m, y ≥ 10 ∧ y ≤ 99 ∧ ∃ k : ℕ, y = k * x) ∧ 
      m.card = 3)) ∧ 
    s.card = 9 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_two_digit_multiples_l2417_241798


namespace NUMINAMATH_CALUDE_root_cubic_value_l2417_241764

theorem root_cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 - 7 = -6 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_value_l2417_241764


namespace NUMINAMATH_CALUDE_pqu_theorem_l2417_241704

/-- A structure representing the relationship between P, Q, and U -/
structure PQU where
  P : ℝ
  Q : ℝ
  U : ℝ
  k : ℝ
  h : P = k * Q / U

/-- The theorem statement -/
theorem pqu_theorem (x y : PQU) (h1 : x.P = 6) (h2 : x.U = 4) (h3 : x.Q = 8)
                    (h4 : y.P = 18) (h5 : y.U = 9) : y.Q = 54 := by
  sorry

end NUMINAMATH_CALUDE_pqu_theorem_l2417_241704


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l2417_241771

/-- Define the operation [a, b, c] as (a + b) / c, where c ≠ 0 -/
def bracket (a b c : ℚ) : ℚ :=
  if c ≠ 0 then (a + b) / c else 0

/-- The main theorem to prove -/
theorem nested_bracket_equals_two :
  bracket (bracket 50 50 100) (bracket 3 6 9) (bracket 20 30 50) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l2417_241771


namespace NUMINAMATH_CALUDE_intersection_distance_l2417_241780

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l2417_241780


namespace NUMINAMATH_CALUDE_shirts_washed_l2417_241774

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (not_washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 27 → not_washed = 16 →
  short_sleeve + long_sleeve - not_washed = 20 := by
  sorry

end NUMINAMATH_CALUDE_shirts_washed_l2417_241774


namespace NUMINAMATH_CALUDE_four_wheelers_count_l2417_241710

/-- Given a parking lot with only 2-wheelers and 4-wheelers, and a total of 58 wheels,
    prove that the number of 4-wheelers can be expressed in terms of the number of 2-wheelers. -/
theorem four_wheelers_count (x y : ℕ) (h1 : 2 * x + 4 * y = 58) :
  y = (29 - x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_four_wheelers_count_l2417_241710


namespace NUMINAMATH_CALUDE_compare_abc_l2417_241745

theorem compare_abc : ∃ (a b c : ℝ),
  a = 2 * Real.log 1.01 ∧
  b = Real.log 1.02 ∧
  c = Real.sqrt 1.04 - 1 ∧
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_compare_abc_l2417_241745


namespace NUMINAMATH_CALUDE_central_cell_value_l2417_241753

/-- Represents a 3x3 table of real numbers -/
def Table := Fin 3 → Fin 3 → ℝ

/-- The product of numbers in a row equals 10 -/
def row_product (t : Table) : Prop :=
  ∀ i : Fin 3, (t i 0) * (t i 1) * (t i 2) = 10

/-- The product of numbers in a column equals 10 -/
def col_product (t : Table) : Prop :=
  ∀ j : Fin 3, (t 0 j) * (t 1 j) * (t 2 j) = 10

/-- The product of numbers in any 2x2 square equals 3 -/
def square_product (t : Table) : Prop :=
  ∀ i j : Fin 2, (t i j) * (t i (j+1)) * (t (i+1) j) * (t (i+1) (j+1)) = 3

theorem central_cell_value (t : Table) 
  (h_row : row_product t) 
  (h_col : col_product t) 
  (h_square : square_product t) : 
  t 1 1 = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l2417_241753


namespace NUMINAMATH_CALUDE_flower_bunch_problem_l2417_241720

/-- The number of flowers in each bunch initially -/
def initial_flowers_per_bunch : ℕ := sorry

/-- The number of bunches initially -/
def initial_bunches : ℕ := 8

/-- The number of flowers per bunch in the alternative scenario -/
def alternative_flowers_per_bunch : ℕ := 12

/-- The number of bunches in the alternative scenario -/
def alternative_bunches : ℕ := 6

theorem flower_bunch_problem :
  initial_flowers_per_bunch * initial_bunches = alternative_flowers_per_bunch * alternative_bunches ∧
  initial_flowers_per_bunch = 9 := by sorry

end NUMINAMATH_CALUDE_flower_bunch_problem_l2417_241720


namespace NUMINAMATH_CALUDE_halloween_jelly_beans_l2417_241784

theorem halloween_jelly_beans 
  (initial_jelly_beans : ℕ)
  (total_children : ℕ)
  (jelly_beans_per_child : ℕ)
  (remaining_jelly_beans : ℕ)
  (h1 : initial_jelly_beans = 100)
  (h2 : total_children = 40)
  (h3 : jelly_beans_per_child = 2)
  (h4 : remaining_jelly_beans = 36)
  : (((initial_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child) / total_children) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_halloween_jelly_beans_l2417_241784


namespace NUMINAMATH_CALUDE_intersection_A_B_l2417_241738

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2417_241738


namespace NUMINAMATH_CALUDE_percentage_difference_l2417_241776

theorem percentage_difference : (40 * 0.8) - (25 * (2/5)) = 22 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2417_241776


namespace NUMINAMATH_CALUDE_book_sale_loss_l2417_241751

/-- Given that the cost price of 15 books equals the selling price of 20 books,
    prove that there is a 25% loss. -/
theorem book_sale_loss (C S : ℝ) (h : 15 * C = 20 * S) : 
  (C - S) / C = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_l2417_241751


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_total_oak_trees_after_planting_l2417_241754

/-- The number of oak trees in a park after planting new trees is equal to the sum of the initial number of trees and the number of newly planted trees. -/
theorem oak_trees_after_planting (initial_trees newly_planted_trees : ℕ) :
  initial_trees + newly_planted_trees = initial_trees + newly_planted_trees :=
by sorry

/-- The park initially has 5 oak trees. -/
def initial_oak_trees : ℕ := 5

/-- The number of oak trees to be planted is 4. -/
def oak_trees_to_plant : ℕ := 4

/-- The total number of oak trees after planting is 9. -/
theorem total_oak_trees_after_planting :
  initial_oak_trees + oak_trees_to_plant = 9 :=
by sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_total_oak_trees_after_planting_l2417_241754


namespace NUMINAMATH_CALUDE_hope_students_approximation_l2417_241769

/-- Rounds a natural number to the nearest thousand -/
def roundToNearestThousand (n : ℕ) : ℕ :=
  1000 * ((n + 500) / 1000)

/-- The number of students in Hope Primary School -/
def hopeStudents : ℕ := 1996

theorem hope_students_approximation :
  roundToNearestThousand hopeStudents = 2000 := by
  sorry

end NUMINAMATH_CALUDE_hope_students_approximation_l2417_241769


namespace NUMINAMATH_CALUDE_subtraction_value_l2417_241768

theorem subtraction_value (N x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 6) / 8 = 6) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_value_l2417_241768


namespace NUMINAMATH_CALUDE_canada_avg_sqft_approx_l2417_241748

/-- The population of Canada in the year 2000 -/
def canada_population : ℕ := 30690000

/-- The total area of Canada in square miles -/
def canada_area : ℕ := 3855103

/-- The number of square feet in one square mile -/
def sqft_per_sqmile : ℕ := 5280 * 5280

/-- The average number of square feet per person in Canada -/
def avg_sqft_per_person : ℚ :=
  (canada_area * sqft_per_sqmile) / canada_population

/-- Theorem stating that the average square feet per person in Canada 
    is approximately 3,000,000 -/
theorem canada_avg_sqft_approx :
  ∃ ε > 0, |avg_sqft_per_person - 3000000| < ε :=
sorry

end NUMINAMATH_CALUDE_canada_avg_sqft_approx_l2417_241748


namespace NUMINAMATH_CALUDE_periodic_sum_implies_periodic_components_l2417_241793

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem periodic_sum_implies_periodic_components
  (f g h : ℝ → ℝ) (T : ℝ)
  (h₁ : is_periodic (λ x => f x + g x) T)
  (h₂ : is_periodic (λ x => f x + h x) T)
  (h₃ : is_periodic (λ x => g x + h x) T) :
  is_periodic f T ∧ is_periodic g T ∧ is_periodic h T :=
sorry

end NUMINAMATH_CALUDE_periodic_sum_implies_periodic_components_l2417_241793


namespace NUMINAMATH_CALUDE_intersection_at_one_point_l2417_241766

theorem intersection_at_one_point (c : ℝ) : 
  (∃! x : ℝ, c * x^2 - 5 * x + 3 = 2 * x + 5) ↔ c = -49/8 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_one_point_l2417_241766


namespace NUMINAMATH_CALUDE_equation_solution_l2417_241723

theorem equation_solution :
  ∀ x : ℝ, x^6 - 19*x^3 = 216 ↔ x = 3 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2417_241723


namespace NUMINAMATH_CALUDE_original_price_calculation_l2417_241743

theorem original_price_calculation (current_price : ℝ) (reduction_percentage : ℝ) 
  (h1 : current_price = 56.10)
  (h2 : reduction_percentage = 0.15)
  : (current_price / (1 - reduction_percentage)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2417_241743


namespace NUMINAMATH_CALUDE_hospital_bill_breakdown_l2417_241701

theorem hospital_bill_breakdown (total_bill : ℝ) (medication_percentage : ℝ) 
  (overnight_percentage : ℝ) (food_cost : ℝ) (h1 : total_bill = 5000) 
  (h2 : medication_percentage = 0.5) (h3 : overnight_percentage = 0.25) 
  (h4 : food_cost = 175) : 
  total_bill * (1 - medication_percentage) * (1 - overnight_percentage) - food_cost = 1700 := by
  sorry

end NUMINAMATH_CALUDE_hospital_bill_breakdown_l2417_241701


namespace NUMINAMATH_CALUDE_flour_per_cake_l2417_241711

/-- The amount of flour needed for each cake given the initial conditions -/
theorem flour_per_cake 
  (traci_flour : ℕ) 
  (harris_flour : ℕ) 
  (traci_cakes : ℕ) 
  (harris_cakes : ℕ) 
  (h1 : traci_flour = 500)
  (h2 : harris_flour = 400)
  (h3 : traci_cakes = 9)
  (h4 : harris_cakes = 9) :
  (traci_flour + harris_flour) / (traci_cakes + harris_cakes) = 50 := by
  sorry

end NUMINAMATH_CALUDE_flour_per_cake_l2417_241711


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l2417_241746

/-- Given two lines in the Cartesian plane that intersect at point M,
    prove that the sum of squared distances from M to the fixed points P and Q is 10. -/
theorem intersection_distance_sum (a : ℝ) (M : ℝ × ℝ) :
  let P : ℝ × ℝ := (0, 1)
  let Q : ℝ × ℝ := (-3, 0)
  let l := {(x, y) : ℝ × ℝ | a * x + y - 1 = 0}
  let m := {(x, y) : ℝ × ℝ | x - a * y + 3 = 0}
  M ∈ l ∧ M ∈ m →
  (M.1 - P.1)^2 + (M.2 - P.2)^2 + (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l2417_241746


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l2417_241789

/-- Rachel's homework problem -/
theorem rachel_homework_difference (math_pages reading_pages : ℕ) : 
  math_pages = 8 →
  reading_pages = 14 →
  reading_pages > math_pages →
  reading_pages - math_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l2417_241789


namespace NUMINAMATH_CALUDE_inequality_theorem_l2417_241770

theorem inequality_theorem (a b : ℝ) (h1 : a^3 > b^3) (h2 : a * b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2417_241770


namespace NUMINAMATH_CALUDE_tourist_journey_days_l2417_241757

def tourist_journey (first_section second_section : ℝ) (speed_difference : ℝ) : Prop :=
  ∃ (x : ℝ),
    -- x is the number of days for the second section
    -- First section takes (x/2 + 1) days
    (x/2 + 1) * (second_section/x - speed_difference) = first_section ∧
    x * (second_section/x) = second_section ∧
    -- Total journey takes 4 days
    x + (x/2 + 1) = 4

theorem tourist_journey_days :
  tourist_journey 246 276 15 :=
sorry

end NUMINAMATH_CALUDE_tourist_journey_days_l2417_241757


namespace NUMINAMATH_CALUDE_triangle_area_l2417_241788

/-- The area of a triangle with base 12 and height 15 is 90 -/
theorem triangle_area (base height area : ℝ) : 
  base = 12 → height = 15 → area = (1/2) * base * height → area = 90 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2417_241788


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2417_241729

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, n = 986 ∧ 
  n % 17 = 0 ∧ 
  n ≥ 100 ∧ n < 1000 ∧
  ∀ m : ℕ, m % 17 = 0 → m ≥ 100 → m < 1000 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2417_241729


namespace NUMINAMATH_CALUDE_chloe_final_score_l2417_241778

def trivia_game (round1 round2 round3 round4 round5 round6 round7 : Int) : Int :=
  round1 + round2 + round3 + round4 + round5 + round6 + round7

theorem chloe_final_score :
  trivia_game 40 50 60 70 (-4) 80 (-6) = 290 := by
  sorry

end NUMINAMATH_CALUDE_chloe_final_score_l2417_241778


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2417_241775

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2417_241775


namespace NUMINAMATH_CALUDE_inequality_range_l2417_241700

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) → 
  a ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2417_241700


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2417_241767

theorem sum_of_cubes (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : (x + y)^2 = 2500) (h2 : x * y = 500) : x^3 + y^3 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2417_241767


namespace NUMINAMATH_CALUDE_chucks_team_score_final_score_proof_l2417_241702

theorem chucks_team_score (red_team_score : ℕ) (score_difference : ℕ) : ℕ :=
  red_team_score + score_difference

theorem final_score_proof (red_team_score : ℕ) (score_difference : ℕ) 
  (h1 : red_team_score = 76)
  (h2 : score_difference = 19) :
  chucks_team_score red_team_score score_difference = 95 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_score_final_score_proof_l2417_241702


namespace NUMINAMATH_CALUDE_symmetry_sum_l2417_241712

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_about_x_axis (a, 1) (2, b) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l2417_241712


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l2417_241742

/-- The equation has exactly one integral root -/
theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l2417_241742


namespace NUMINAMATH_CALUDE_power_equation_solution_l2417_241759

theorem power_equation_solution (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2417_241759


namespace NUMINAMATH_CALUDE_approx_cube_root_2370_l2417_241765

-- Define the approximation relation
def approx (x y : ℝ) := ∃ ε > 0, |x - y| < ε

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem approx_cube_root_2370 (h : approx (cubeRoot 2.37) 1.333) :
  approx (cubeRoot 2370) 13.33 := by
  sorry

end NUMINAMATH_CALUDE_approx_cube_root_2370_l2417_241765


namespace NUMINAMATH_CALUDE_trapezoid_height_l2417_241713

/-- Given S = (1/2)(a+b)h and a+b ≠ 0, prove that h = 2S / (a+b) -/
theorem trapezoid_height (a b S h : ℝ) (h_eq : S = (1/2) * (a + b) * h) (h_ne_zero : a + b ≠ 0) :
  h = 2 * S / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_height_l2417_241713


namespace NUMINAMATH_CALUDE_simplify_equation_l2417_241721

theorem simplify_equation : ∀ x : ℝ, 
  3 * x + 4.8 * x - 10 * x = 11 * (1 / 5) ↔ -2.2 * x = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_equation_l2417_241721


namespace NUMINAMATH_CALUDE_ellipse_intersection_product_range_l2417_241715

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the vector from origin to a point
def vector_from_origin (p : ℝ × ℝ) : ℝ × ℝ := p

-- Define the vector from M to a point
def vector_from_M (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - M.1, p.2 - M.2)

-- Statement of the theorem
theorem ellipse_intersection_product_range :
  ∀ P Q : ℝ × ℝ,
  C P.1 P.2 →
  C Q.1 Q.2 →
  (∃ k : ℝ, Q.2 - M.2 = k * (Q.1 - M.1) ∧ P.2 - M.2 = k * (P.1 - M.1)) →
  -20 ≤ (dot_product (vector_from_origin P) (vector_from_origin Q) +
         dot_product (vector_from_M P) (vector_from_M Q)) ∧
  (dot_product (vector_from_origin P) (vector_from_origin Q) +
   dot_product (vector_from_M P) (vector_from_M Q)) ≤ -52/3 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_product_range_l2417_241715


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l2417_241706

theorem cyclist_speed_problem (x y : ℝ) :
  y = x + 5 ∧  -- Y's speed is 5 mph faster than X's
  100 / y + 1/6 + 20 / y = 80 / x + 1/4 ∧  -- Time equality equation
  x > 0 ∧ y > 0  -- Positive speeds
  → x = 10 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l2417_241706


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l2417_241739

theorem integer_pair_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a ^ 3 = k * b ^ 2) ∧ 
  (∃ m : ℕ, b - 1 = m * (a - 1)) →
  b = 1 ∨ b = a := by
sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l2417_241739


namespace NUMINAMATH_CALUDE_total_students_calculation_l2417_241722

theorem total_students_calculation (short_ratio : Rat) (tall_count : Nat) (average_count : Nat) :
  short_ratio = 2/5 →
  tall_count = 90 →
  average_count = 150 →
  ∃ (total : Nat), total = (tall_count + average_count) / (1 - short_ratio) ∧ total = 400 :=
by sorry

end NUMINAMATH_CALUDE_total_students_calculation_l2417_241722


namespace NUMINAMATH_CALUDE_socks_purchase_problem_l2417_241795

theorem socks_purchase_problem :
  ∃ (a b c : ℕ), 
    a + b + c = 15 ∧
    2 * a + 3 * b + 5 * c = 40 ∧
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
    (a = 7 ∨ a = 9 ∨ a = 11) :=
by sorry

end NUMINAMATH_CALUDE_socks_purchase_problem_l2417_241795


namespace NUMINAMATH_CALUDE_millie_bracelets_l2417_241762

theorem millie_bracelets (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
  (h1 : lost = 2) 
  (h2 : remaining = 7) 
  (h3 : initial = remaining + lost) : initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_millie_bracelets_l2417_241762


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2417_241781

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) (k : ℝ) :
  isGeometricSequence a →
  a 5 * a 8 * a 11 = k →
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2417_241781


namespace NUMINAMATH_CALUDE_divergent_series_with_convergent_min_series_l2417_241791

theorem divergent_series_with_convergent_min_series :
  ∃ (x : ℕ → ℝ), 
    (∀ n, x n > 0) ∧ 
    (∀ n, x (n + 1) < x n) ∧ 
    (¬ Summable x) ∧
    (Summable (fun n => min (x (n + 1)) (1 / ((n + 1 : ℝ) * Real.log (n + 1))))) := by
  sorry

end NUMINAMATH_CALUDE_divergent_series_with_convergent_min_series_l2417_241791


namespace NUMINAMATH_CALUDE_dot_product_range_l2417_241714

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1)

-- Define the curve y = √(1-x^2)
def on_curve (P : ℝ × ℝ) : Prop :=
  P.2 = Real.sqrt (1 - P.1^2)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from B to P
def BP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - B.1, P.2 - B.2)

-- Define the vector from B to A
def BA : ℝ × ℝ :=
  (A.1 - B.1, A.2 - B.2)

-- The main theorem
theorem dot_product_range :
  ∀ P : ℝ × ℝ, on_curve P →
  0 ≤ dot_product (BP P) BA ∧ dot_product (BP P) BA ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l2417_241714


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l2417_241799

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x^4 + 1/x^4 = 2398) : 
  x^2 + 1/x^2 = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l2417_241799


namespace NUMINAMATH_CALUDE_unique_solution_for_C_equality_l2417_241747

-- Define C(k) as the sum of distinct prime divisors of k
def C (k : ℕ+) : ℕ := sorry

-- Theorem statement
theorem unique_solution_for_C_equality :
  ∀ n : ℕ+, C (2^n.val + 1) = C n ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_C_equality_l2417_241747


namespace NUMINAMATH_CALUDE_min_max_inequality_l2417_241796

theorem min_max_inequality {a b x₁ x₂ x₃ x₄ : ℝ} 
  (ha : 0 < a) (hab : a < b) 
  (hx₁ : a ≤ x₁ ∧ x₁ ≤ b) (hx₂ : a ≤ x₂ ∧ x₂ ≤ b) 
  (hx₃ : a ≤ x₃ ∧ x₃ ≤ b) (hx₄ : a ≤ x₄ ∧ x₄ ≤ b) :
  1 ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ b/a + a/b - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_max_inequality_l2417_241796


namespace NUMINAMATH_CALUDE_quadratic_expression_equals_64_l2417_241741

theorem quadratic_expression_equals_64 (x : ℝ) : 
  (2*x + 3)^2 + 2*(2*x + 3)*(5 - 2*x) + (5 - 2*x)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equals_64_l2417_241741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2417_241732

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1 + a 3 = 22)
  (h_sixth : a 6 = 7) :
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2417_241732


namespace NUMINAMATH_CALUDE_floor_of_negative_decimal_l2417_241752

theorem floor_of_negative_decimal (x : ℝ) : x = -3.7 → ⌊x⌋ = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_negative_decimal_l2417_241752


namespace NUMINAMATH_CALUDE_constant_sum_area_one_iff_identical_l2417_241794

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 1000
  col : Fin 1000
  value : ℝ

/-- Represents the entire 1000 × 1000 grid -/
def Grid := Cell → ℝ

/-- A rectangle within the grid -/
structure Rectangle where
  top_left : Cell
  bottom_right : Cell

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  (r.bottom_right.row - r.top_left.row + 1) * (r.bottom_right.col - r.top_left.col + 1)

/-- The sum of values in a rectangle -/
def sum_rectangle (g : Grid) (r : Rectangle) : ℝ := sorry

/-- Predicate: all rectangles of area s have the same sum -/
def constant_sum_for_area (g : Grid) (s : ℕ) : Prop :=
  ∀ r₁ r₂ : Rectangle, area r₁ = s → area r₂ = s → sum_rectangle g r₁ = sum_rectangle g r₂

/-- Predicate: all cells in the grid have the same value -/
def all_cells_identical (g : Grid) : Prop :=
  ∀ c₁ c₂ : Cell, g c₁ = g c₂

/-- Main theorem: constant sum for area 1 implies all cells are identical -/
theorem constant_sum_area_one_iff_identical :
  ∀ g : Grid, constant_sum_for_area g 1 ↔ all_cells_identical g :=
sorry

end NUMINAMATH_CALUDE_constant_sum_area_one_iff_identical_l2417_241794


namespace NUMINAMATH_CALUDE_intersection_A_B_l2417_241755

def A : Set ℕ := {0,1,2,3,4,5,6}

def B : Set ℕ := {x | ∃ n ∈ A, x = 2 * n}

theorem intersection_A_B : A ∩ B = {0,2,4,6} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2417_241755


namespace NUMINAMATH_CALUDE_complex_root_of_unity_product_l2417_241709

theorem complex_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_of_unity_product_l2417_241709


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2417_241783

/-- The perimeter of an isosceles triangle given specific conditions -/
theorem isosceles_triangle_perimeter : 
  ∀ (equilateral_perimeter isosceles_base : ℝ),
  equilateral_perimeter = 60 →
  isosceles_base = 15 →
  ∃ (isosceles_perimeter : ℝ),
  isosceles_perimeter = equilateral_perimeter / 3 + equilateral_perimeter / 3 + isosceles_base ∧
  isosceles_perimeter = 55 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2417_241783


namespace NUMINAMATH_CALUDE_integer_fraction_solutions_l2417_241787

theorem integer_fraction_solutions (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k) ↔
  (∃ n : ℕ+, (a = n ∧ b = 2 * n) ∨ 
             (a = 8 * n ^ 4 - n ∧ b = 2 * n) ∨ 
             (a = 2 * n ∧ b = 1)) :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_solutions_l2417_241787


namespace NUMINAMATH_CALUDE_problem_statement_l2417_241718

theorem problem_statement : 2 * ((7 + 5)^2 + (7^2 + 5^2)) = 436 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2417_241718
