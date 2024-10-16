import Mathlib

namespace NUMINAMATH_CALUDE_chinese_heritage_tv_event_is_random_l2277_227716

/-- Represents a TV event -/
structure TVEvent where
  program : String
  canOccur : Bool
  hasUncertainty : Bool

/-- Classifies an event as certain, impossible, or random -/
inductive EventClassification
  | Certain
  | Impossible
  | Random

/-- Determines if an event is random based on its properties -/
def isRandomEvent (e : TVEvent) : Bool :=
  e.canOccur ∧ e.hasUncertainty

/-- Classifies a TV event based on its properties -/
def classifyTVEvent (e : TVEvent) : EventClassification :=
  if isRandomEvent e then EventClassification.Random
  else if e.canOccur then EventClassification.Certain
  else EventClassification.Impossible

/-- The main theorem stating that turning on the TV and broadcasting
    "Chinese Intangible Cultural Heritage" is a random event -/
theorem chinese_heritage_tv_event_is_random :
  let e := TVEvent.mk "Chinese Intangible Cultural Heritage" true true
  classifyTVEvent e = EventClassification.Random := by
  sorry


end NUMINAMATH_CALUDE_chinese_heritage_tv_event_is_random_l2277_227716


namespace NUMINAMATH_CALUDE_f_inequality_implies_a_range_l2277_227776

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 + 4*x else Real.log (x + 1)

-- State the theorem
theorem f_inequality_implies_a_range :
  (∀ x, |f x| ≥ a * x) → a ∈ Set.Icc (-4) 0 :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_implies_a_range_l2277_227776


namespace NUMINAMATH_CALUDE_kermit_sleep_positions_l2277_227709

/-- Represents a position on the infinite square grid -/
structure Position :=
  (x : Int) (y : Int)

/-- The number of Joules Kermit starts with -/
def initial_energy : Nat := 100

/-- Calculates the number of unique positions Kermit can reach -/
def unique_positions (energy : Nat) : Nat :=
  (2 * energy + 1) * (2 * energy + 1)

/-- Theorem stating the number of unique positions Kermit can reach -/
theorem kermit_sleep_positions : 
  unique_positions initial_energy = 10201 := by
  sorry

end NUMINAMATH_CALUDE_kermit_sleep_positions_l2277_227709


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l2277_227761

def price_reduction_problem (original_price : ℝ) : Prop :=
  let reduced_price := 0.75 * original_price
  let original_amount := 1100 / original_price
  let new_amount := 1100 / reduced_price
  (new_amount - original_amount = 5) ∧ (reduced_price = 55)

theorem price_reduction_theorem :
  ∃ (original_price : ℝ), price_reduction_problem original_price :=
sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l2277_227761


namespace NUMINAMATH_CALUDE_three_digit_not_mult_4_or_6_eq_600_l2277_227700

/-- The number of three-digit numbers that are multiples of neither 4 nor 6 -/
def three_digit_not_mult_4_or_6 : ℕ :=
  let three_digit_count := 999 - 100 + 1
  let mult_4_count := (996 / 4) - (100 / 4) + 1
  let mult_6_count := (996 / 6) - (102 / 6) + 1
  let mult_12_count := (996 / 12) - (108 / 12) + 1
  three_digit_count - (mult_4_count + mult_6_count - mult_12_count)

theorem three_digit_not_mult_4_or_6_eq_600 :
  three_digit_not_mult_4_or_6 = 600 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_not_mult_4_or_6_eq_600_l2277_227700


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l2277_227784

-- Define the sets P and Q
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q : (P.compl) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l2277_227784


namespace NUMINAMATH_CALUDE_divide_plot_with_fences_l2277_227742

/-- Represents a rectangular plot with length and width -/
structure Plot where
  length : ℝ
  width : ℝ

/-- Represents a fence with a length -/
structure Fence where
  length : ℝ

/-- Represents a section of the plot -/
structure Section where
  area : ℝ

theorem divide_plot_with_fences (p : Plot) (f : Fence) :
  p.length = 80 →
  p.width = 50 →
  ∃ (sections : Finset Section),
    sections.card = 5 ∧
    (∀ s ∈ sections, s.area = (p.length * p.width) / 5) ∧
    f.length = 40 := by
  sorry

end NUMINAMATH_CALUDE_divide_plot_with_fences_l2277_227742


namespace NUMINAMATH_CALUDE_percent_less_than_l2277_227724

theorem percent_less_than (x y z : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : x = 0.78 * z) : 
  y = 0.6 * z := by
sorry

end NUMINAMATH_CALUDE_percent_less_than_l2277_227724


namespace NUMINAMATH_CALUDE_complex_equality_l2277_227780

/-- Given a complex number z = 1-ni, prove that m+ni = 2-i -/
theorem complex_equality (m n : ℝ) (z : ℂ) (h : z = 1 - n * Complex.I) :
  m + n * Complex.I = 2 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equality_l2277_227780


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l2277_227771

/-- A circle with center on y = b intersects y = (4/3)x^2 at least thrice, including the origin --/
def CircleIntersectsParabola (b : ℝ) : Prop :=
  ∃ (r : ℝ) (a : ℝ), (a^2 + b^2 = r^2) ∧ 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    ((x₁^2 + ((4/3)*x₁^2 - b)^2 = r^2) ∧
     (x₂^2 + ((4/3)*x₂^2 - b)^2 = r^2)))

/-- Two non-origin intersection points lie on y = (4/3)x + b --/
def IntersectionPointsOnLine (b : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    ((4/3)*x₁^2 = (4/3)*x₁ + b) ∧
    ((4/3)*x₂^2 = (4/3)*x₂ + b)

/-- The theorem to be proved --/
theorem circle_parabola_intersection (b : ℝ) :
  (CircleIntersectsParabola b ∧ IntersectionPointsOnLine b) ↔ b = 25/12 :=
sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l2277_227771


namespace NUMINAMATH_CALUDE_parabola_vertex_l2277_227779

/-- The vertex of the parabola y = 3(x+1)^2 + 4 is (-1, 4) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3 * (x + 1)^2 + 4 → (∃ h k : ℝ, h = -1 ∧ k = 4 ∧ ∀ x y : ℝ, y = 3 * (x - h)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2277_227779


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l2277_227703

def appetizer_cost : ℚ := 8
def entree_cost : ℚ := 20
def wine_cost : ℚ := 3
def dessert_cost : ℚ := 6
def discount_ratio : ℚ := (1/2)
def total_spent : ℚ := 38

def full_cost : ℚ := appetizer_cost + entree_cost + 2 * wine_cost + dessert_cost
def discounted_cost : ℚ := appetizer_cost + entree_cost * (1 - discount_ratio) + 2 * wine_cost + dessert_cost
def tip_amount : ℚ := total_spent - discounted_cost

theorem tip_percentage_is_twenty_percent :
  (tip_amount / full_cost) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l2277_227703


namespace NUMINAMATH_CALUDE_max_a_value_l2277_227725

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, a * x^2 + 2 * a * x + 3 * a ≤ 1) →
  a ≤ 1/6 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2277_227725


namespace NUMINAMATH_CALUDE_quotient_less_than_dividend_l2277_227734

theorem quotient_less_than_dividend : 
  let a := (5 : ℚ) / 7
  let b := (5 : ℚ) / 4
  a / b < a :=
by sorry

end NUMINAMATH_CALUDE_quotient_less_than_dividend_l2277_227734


namespace NUMINAMATH_CALUDE_digit_150_is_5_l2277_227714

-- Define the fraction
def fraction : ℚ := 5 / 37

-- Define the length of the repeating cycle
def cycle_length : ℕ := 3

-- Define the position we're interested in
def target_position : ℕ := 150

-- Define the function to get the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_150_is_5 : nth_digit target_position = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l2277_227714


namespace NUMINAMATH_CALUDE_yolanda_rate_l2277_227775

def total_distance : ℝ := 31
def bob_distance : ℝ := 20
def bob_rate : ℝ := 2

theorem yolanda_rate (total_distance : ℝ) (bob_distance : ℝ) (bob_rate : ℝ) :
  total_distance = 31 →
  bob_distance = 20 →
  bob_rate = 2 →
  ∃ yolanda_rate : ℝ,
    yolanda_rate = (total_distance - bob_distance) / (bob_distance / bob_rate) ∧
    yolanda_rate = 1.1 :=
by sorry

end NUMINAMATH_CALUDE_yolanda_rate_l2277_227775


namespace NUMINAMATH_CALUDE_cistern_filling_time_l2277_227702

theorem cistern_filling_time (p q : ℝ) (h1 : p > 0) (h2 : q > 0) : 
  p = 1 / 12 → q = 1 / 15 → 
  let combined_rate := p + q
  let filled_portion := 4 * combined_rate
  let remaining_portion := 1 - filled_portion
  remaining_portion / q = 6 := by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l2277_227702


namespace NUMINAMATH_CALUDE_midpoint_coordinates_sum_l2277_227799

/-- Given that N(5, -1) is the midpoint of segment CD and C has coordinates (11, 10),
    prove that the sum of the coordinates of D is -13. -/
theorem midpoint_coordinates_sum (N C D : ℝ × ℝ) : 
  N = (5, -1) →
  C = (11, 10) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = -13 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_sum_l2277_227799


namespace NUMINAMATH_CALUDE_cylinder_equation_l2277_227756

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) (c : ℝ) : Prop :=
  ∀ p : CylindricalPoint, p ∈ S ↔ p.r = c

theorem cylinder_equation (c : ℝ) (h : c > 0) :
  IsCylinder {p : CylindricalPoint | p.r = c} c :=
by sorry

end NUMINAMATH_CALUDE_cylinder_equation_l2277_227756


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2277_227796

/-- Two-dimensional vector -/
def Vector2D := ℝ × ℝ

/-- Parallel vectors are scalar multiples of each other -/
def is_parallel (v w : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : Vector2D := (1, -2)
  let b : Vector2D := (-2, x)
  is_parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2277_227796


namespace NUMINAMATH_CALUDE_star_three_six_eq_seven_l2277_227757

/-- The ☆ operation on rational numbers -/
def star (a : ℚ) (x y : ℚ) : ℚ := a^2 * x + a * y + 1

/-- Theorem: If 1 ☆ 2 = 3, then 3 ☆ 6 = 7 -/
theorem star_three_six_eq_seven (a : ℚ) (h : star a 1 2 = 3) : star a 3 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_star_three_six_eq_seven_l2277_227757


namespace NUMINAMATH_CALUDE_right_triangle_area_l2277_227744

theorem right_triangle_area (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 14) :
  (1/2) * a * b = (1/2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2277_227744


namespace NUMINAMATH_CALUDE_count_divisors_of_360_l2277_227748

theorem count_divisors_of_360 : Finset.card (Nat.divisors 360) = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_of_360_l2277_227748


namespace NUMINAMATH_CALUDE_tims_books_l2277_227789

theorem tims_books (mike_books : ℕ) (total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) :
  total_books - mike_books = 22 := by
sorry

end NUMINAMATH_CALUDE_tims_books_l2277_227789


namespace NUMINAMATH_CALUDE_gcf_and_lcm_of_numbers_l2277_227762

def numbers : List Nat := [42, 126, 105]

theorem gcf_and_lcm_of_numbers :
  (Nat.gcd (Nat.gcd 42 126) 105 = 21) ∧
  (Nat.lcm (Nat.lcm 42 126) 105 = 630) := by
  sorry

end NUMINAMATH_CALUDE_gcf_and_lcm_of_numbers_l2277_227762


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_l2277_227772

theorem sum_of_last_two_digits (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_l2277_227772


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l2277_227738

theorem complex_exponential_sum (γ δ : ℝ) : 
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -5/8 + 9/10 * Complex.I → 
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -5/8 - 9/10 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l2277_227738


namespace NUMINAMATH_CALUDE_groups_formed_equals_seven_l2277_227749

/-- Given a class with boys and girls, and a group size, calculate the number of groups formed. -/
def calculateGroups (boys : ℕ) (girls : ℕ) (groupSize : ℕ) : ℕ :=
  (boys + girls) / groupSize

/-- Theorem: Given 9 boys, 12 girls, and groups of 3 members, 7 groups are formed. -/
theorem groups_formed_equals_seven :
  calculateGroups 9 12 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_groups_formed_equals_seven_l2277_227749


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2277_227797

def cost_price : ℝ := 900
def selling_price : ℝ := 1080

theorem gain_percent_calculation :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2277_227797


namespace NUMINAMATH_CALUDE_student_average_age_l2277_227707

theorem student_average_age (n : ℕ) (teacher_age : ℕ) (avg_increase : ℚ) :
  n = 19 →
  teacher_age = 40 →
  avg_increase = 1 →
  ∃ (student_avg : ℚ),
    (n : ℚ) * student_avg + teacher_age = (n + 1 : ℚ) * (student_avg + avg_increase) ∧
    student_avg = 20 :=
by sorry

end NUMINAMATH_CALUDE_student_average_age_l2277_227707


namespace NUMINAMATH_CALUDE_two_special_numbers_exist_l2277_227792

theorem two_special_numbers_exist : ∃ (x y : ℕ), 
  x + y = 2013 ∧ 
  y = 5 * ((x / 100) + 1) ∧ 
  x > y :=
by sorry

end NUMINAMATH_CALUDE_two_special_numbers_exist_l2277_227792


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2277_227730

theorem cubic_sum_theorem (a b c : ℂ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 3) 
  (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2277_227730


namespace NUMINAMATH_CALUDE_xander_miles_more_l2277_227795

/-- The problem statement and conditions --/
theorem xander_miles_more (t s : ℝ) 
  (h1 : t > 0) 
  (h2 : s > 0) 
  (h3 : s * t + 100 = (s + 10) * (t + 1.5)) : 
  (s + 15) * (t + 3) - s * t = 215 := by
  sorry

end NUMINAMATH_CALUDE_xander_miles_more_l2277_227795


namespace NUMINAMATH_CALUDE_equal_probabilities_l2277_227785

/-- Represents a box containing colored balls -/
structure Box where
  red : ℕ
  green : ℕ

/-- The initial state of the boxes -/
def initial_state : Box × Box :=
  (⟨100, 0⟩, ⟨0, 100⟩)

/-- The state after transferring 8 red balls to the green box -/
def after_first_transfer (state : Box × Box) : Box × Box :=
  let (red_box, green_box) := state
  (⟨red_box.red - 8, red_box.green⟩, ⟨green_box.red + 8, green_box.green⟩)

/-- The final state after transferring 8 balls back to the red box -/
def final_state (state : Box × Box) : Box × Box :=
  let (red_box, green_box) := after_first_transfer state
  (⟨red_box.red + 8, red_box.green + 8⟩, ⟨green_box.red - 8, green_box.green - 8⟩)

/-- The probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  if color = "red" then
    box.red / (box.red + box.green)
  else
    box.green / (box.red + box.green)

theorem equal_probabilities :
  let (final_red_box, final_green_box) := final_state initial_state
  prob_draw final_red_box "green" = prob_draw final_green_box "red" := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_l2277_227785


namespace NUMINAMATH_CALUDE_polynomial_root_comparison_l2277_227759

theorem polynomial_root_comparison (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ ≤ a₂) (h2 : a₂ ≤ a₃) 
  (h3 : b₁ ≤ b₂) (h4 : b₂ ≤ b₃) 
  (h5 : a₁ + a₂ + a₃ = b₁ + b₂ + b₃) 
  (h6 : a₁*a₂ + a₂*a₃ + a₁*a₃ = b₁*b₂ + b₂*b₃ + b₁*b₃) 
  (h7 : a₁ ≤ b₁) : 
  a₃ ≤ b₃ := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_comparison_l2277_227759


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2277_227798

theorem negation_of_proposition (P : ℕ → Prop) :
  (∀ m : ℕ, 4^m ≥ 4*m) ↔ ¬(∃ m : ℕ, 4^m < 4*m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2277_227798


namespace NUMINAMATH_CALUDE_range_of_z_l2277_227754

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) : 
  ∃ z, z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l2277_227754


namespace NUMINAMATH_CALUDE_janine_reading_ratio_l2277_227778

/-- The number of books Janine read last month -/
def books_last_month : ℕ := 5

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := 150

/-- The number of books Janine read this month -/
def books_this_month : ℕ := (total_pages - books_last_month * pages_per_book) / pages_per_book

/-- The ratio of books read this month to last month -/
def book_ratio : ℚ := books_this_month / books_last_month

theorem janine_reading_ratio :
  book_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_janine_reading_ratio_l2277_227778


namespace NUMINAMATH_CALUDE_expression_evaluation_l2277_227751

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -2
  (3*x + 2*y) * (3*x - 2*y) - 5*x*(x - y) - (2*x - y)^2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2277_227751


namespace NUMINAMATH_CALUDE_james_yearly_pages_l2277_227758

/-- Calculates the number of pages James writes in a year -/
def pages_per_year (pages_per_letter : ℕ) (num_friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * num_friends * times_per_week * weeks_per_year

/-- Proves that James writes 624 pages in a year -/
theorem james_yearly_pages :
  pages_per_year 3 2 2 52 = 624 := by
  sorry

end NUMINAMATH_CALUDE_james_yearly_pages_l2277_227758


namespace NUMINAMATH_CALUDE_inverse_image_of_three_l2277_227708

-- Define the mapping f: A → B
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem inverse_image_of_three (h : f 1 = 3) : ∃ x, f x = 3 ∧ x = 1 := by
  sorry


end NUMINAMATH_CALUDE_inverse_image_of_three_l2277_227708


namespace NUMINAMATH_CALUDE_morgan_experiment_correct_l2277_227777

/-- Statements about biological experiments and research -/
inductive BiologicalStatement
| A : BiologicalStatement -- Ovary of locusts for observing animal cell meiosis
| B : BiologicalStatement -- Morgan's fruit fly experiment
| C : BiologicalStatement -- Hydrogen peroxide as substrate in enzyme activity experiment
| D : BiologicalStatement -- Investigating red-green color blindness incidence

/-- Predicate to determine if a biological statement is correct -/
def is_correct : BiologicalStatement → Prop
| BiologicalStatement.A => False
| BiologicalStatement.B => True
| BiologicalStatement.C => False
| BiologicalStatement.D => False

/-- Theorem stating that Morgan's fruit fly experiment statement is correct -/
theorem morgan_experiment_correct :
  is_correct BiologicalStatement.B :=
by sorry

end NUMINAMATH_CALUDE_morgan_experiment_correct_l2277_227777


namespace NUMINAMATH_CALUDE_line_slope_and_point_l2277_227750

/-- Given two points P and Q in a coordinate plane, if the slope of the line
    through P and Q is -5/4, then the y-coordinate of Q is -2. Additionally,
    if R is a point on this line and is horizontally 6 units to the right of Q,
    then R has coordinates (11, -9.5). -/
theorem line_slope_and_point (P Q R : ℝ × ℝ) : 
  P = (-3, 8) →
  Q.1 = 5 →
  (Q.2 - P.2) / (Q.1 - P.1) = -5/4 →
  R.1 = Q.1 + 6 →
  (R.2 - Q.2) / (R.1 - Q.1) = -5/4 →
  Q.2 = -2 ∧ R = (11, -9.5) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_and_point_l2277_227750


namespace NUMINAMATH_CALUDE_decimal_25_to_binary_l2277_227731

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

theorem decimal_25_to_binary :
  decimal_to_binary 25 = [true, true, false, false, true] := by
  sorry

#eval decimal_to_binary 25

end NUMINAMATH_CALUDE_decimal_25_to_binary_l2277_227731


namespace NUMINAMATH_CALUDE_male_average_height_l2277_227783

/-- Proves that the average height of males in a school is 185 cm given the following conditions:
  - The average height of all students is 180 cm
  - The average height of females is 170 cm
  - The ratio of men to women is 2:1
-/
theorem male_average_height (total_avg : ℝ) (female_avg : ℝ) (male_female_ratio : ℚ) :
  total_avg = 180 →
  female_avg = 170 →
  male_female_ratio = 2 →
  ∃ (male_avg : ℝ), male_avg = 185 := by
  sorry

end NUMINAMATH_CALUDE_male_average_height_l2277_227783


namespace NUMINAMATH_CALUDE_safari_snake_giraffe_difference_l2277_227753

/-- Represents the number of animals in Safari National Park -/
structure SafariPark where
  lions : ℕ
  snakes : ℕ
  giraffes : ℕ

/-- Represents the number of animals in Savanna National Park -/
structure SavannaPark where
  lions : ℕ
  snakes : ℕ
  giraffes : ℕ

/-- The main theorem stating the difference between snakes and giraffes in Safari National Park -/
theorem safari_snake_giraffe_difference (safari : SafariPark) (savanna : SavannaPark) :
  safari.lions = 100 →
  safari.snakes = safari.lions / 2 →
  savanna.lions = 2 * safari.lions →
  savanna.snakes = 3 * safari.snakes →
  savanna.giraffes = safari.giraffes + 20 →
  savanna.lions + savanna.snakes + savanna.giraffes = 410 →
  safari.snakes - safari.giraffes = 10 := by
  sorry


end NUMINAMATH_CALUDE_safari_snake_giraffe_difference_l2277_227753


namespace NUMINAMATH_CALUDE_subset_condition_l2277_227787

theorem subset_condition (a : ℝ) : 
  ({x : ℝ | 1 ≤ x ∧ x ≤ 2} ⊆ {x : ℝ | a < x}) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l2277_227787


namespace NUMINAMATH_CALUDE_racket_price_l2277_227786

theorem racket_price (total_spent : ℚ) (h1 : total_spent = 90) : ∃ (original_price : ℚ),
  original_price + original_price / 2 = total_spent ∧ original_price = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_racket_price_l2277_227786


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_seven_l2277_227723

theorem sum_of_sixth_powers_mod_seven :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_seven_l2277_227723


namespace NUMINAMATH_CALUDE_rebecca_pie_slices_l2277_227764

theorem rebecca_pie_slices (total_pies : ℕ) (slices_per_pie : ℕ) 
  (remaining_slices : ℕ) (rebecca_husband_slices : ℕ) 
  (family_friends_percent : ℚ) :
  total_pies = 2 →
  slices_per_pie = 8 →
  remaining_slices = 5 →
  rebecca_husband_slices = 2 →
  family_friends_percent = 1/2 →
  ∃ (rebecca_initial_slices : ℕ),
    rebecca_initial_slices = total_pies * slices_per_pie - 
      ((remaining_slices + rebecca_husband_slices) / family_friends_percent) :=
by sorry

end NUMINAMATH_CALUDE_rebecca_pie_slices_l2277_227764


namespace NUMINAMATH_CALUDE_set_difference_M_N_l2277_227767

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem set_difference_M_N : M \ N = {1, 5} := by sorry

end NUMINAMATH_CALUDE_set_difference_M_N_l2277_227767


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2277_227720

theorem fourteenth_root_of_unity : 
  ∃ n : ℕ, n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 14)) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2277_227720


namespace NUMINAMATH_CALUDE_function_minimum_value_l2277_227712

/-- Given a function f(x) = (ax + b) / (x^2 + 4) that attains a maximum value of 1 at x = -1,
    prove that the minimum value of f(x) is -1/4 -/
theorem function_minimum_value (a b : ℝ) :
  let f := fun x : ℝ => (a * x + b) / (x^2 + 4)
  (f (-1) = 1) →
  (∃ x₀, ∀ x, f x ≥ f x₀) →
  (∃ x₁, f x₁ = -1/4 ∧ ∀ x, f x ≥ -1/4) :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_value_l2277_227712


namespace NUMINAMATH_CALUDE_total_toys_l2277_227766

/-- The number of toys each person has -/
structure ToyCount where
  jaxon : ℕ
  gabriel : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def toy_conditions (t : ToyCount) : Prop :=
  t.jaxon = 15 ∧ 
  t.gabriel = 2 * t.jaxon ∧ 
  t.jerry = t.gabriel + 8

/-- The theorem stating the total number of toys -/
theorem total_toys (t : ToyCount) (h : toy_conditions t) : 
  t.jaxon + t.gabriel + t.jerry = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_l2277_227766


namespace NUMINAMATH_CALUDE_exactly_two_propositions_true_l2277_227791

-- Define the propositions
def proposition1 : Prop := ∀ x : ℝ, (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)
def proposition2 : Prop := (∀ x y : ℝ, x + y = 0 → (x = -y)) ↔ (∀ x y : ℝ, x = -y → x + y = 0)

-- Theorem statement
theorem exactly_two_propositions_true : 
  (proposition1 = true) ∧ (proposition2 = true) ∧
  (¬ proposition1 = false) ∧ (¬ proposition2 = false) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_propositions_true_l2277_227791


namespace NUMINAMATH_CALUDE_intersection_empty_range_l2277_227790

theorem intersection_empty_range (a : ℝ) : 
  (∀ x : ℝ, (|x - a| < 1 → ¬(1 < x ∧ x < 5))) ↔ (a ≤ 0 ∨ a ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_range_l2277_227790


namespace NUMINAMATH_CALUDE_lindsay_dolls_l2277_227735

theorem lindsay_dolls (blonde : ℕ) (brown black red : ℕ) : 
  blonde = 6 →
  brown = 3 * blonde →
  black = brown / 2 →
  red = 2 * black →
  (black + brown + red) - blonde = 39 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_dolls_l2277_227735


namespace NUMINAMATH_CALUDE_expand_product_l2277_227718

theorem expand_product (x : ℝ) : (2 + x^2) * (3 - x^3 + x^5) = 6 + 3*x^2 - 2*x^3 + x^5 + x^7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2277_227718


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2277_227727

theorem inequality_system_solution :
  ∀ p : ℝ, (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2277_227727


namespace NUMINAMATH_CALUDE_train_meetings_l2277_227743

-- Define the travel time in minutes
def travel_time : ℕ := 210

-- Define the departure interval in minutes
def departure_interval : ℕ := 60

-- Define the time difference between the 9:00 AM train and the first train in minutes
def time_difference : ℕ := 180

-- Define a function to calculate the number of meetings
def number_of_meetings (travel_time departure_interval time_difference : ℕ) : ℕ :=
  -- The actual calculation would go here, but we're using sorry as per instructions
  sorry

-- Theorem statement
theorem train_meetings :
  number_of_meetings travel_time departure_interval time_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_train_meetings_l2277_227743


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l2277_227719

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (1, 0)

-- Define the asymptotes of the hyperbola
def hyperbola_asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem distance_focus_to_asymptotes :
  ∃ (d : ℝ), d = Real.sqrt 3 / 2 ∧
  ∀ (x y : ℝ), hyperbola_asymptotes x y →
    d = (|Real.sqrt 3 * parabola_focus.1 + parabola_focus.2|) / Real.sqrt (3^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l2277_227719


namespace NUMINAMATH_CALUDE_corgi_dog_price_calculation_l2277_227717

/-- The price calculation for Corgi dogs with profit --/
theorem corgi_dog_price_calculation (original_price : ℝ) (profit_percentage : ℝ) (num_dogs : ℕ) :
  original_price = 1000 →
  profit_percentage = 30 →
  num_dogs = 2 →
  let profit_per_dog := original_price * (profit_percentage / 100)
  let selling_price_per_dog := original_price + profit_per_dog
  let total_cost := selling_price_per_dog * num_dogs
  total_cost = 2600 := by
  sorry


end NUMINAMATH_CALUDE_corgi_dog_price_calculation_l2277_227717


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l2277_227746

theorem opposite_reciprocal_expression_value :
  ∀ (a b c : ℤ) (m n : ℚ),
    a = -b →                          -- a and b are opposite numbers
    c = -1 →                          -- c is the smallest negative integer in absolute value
    m * n = 1 →                       -- m and n are reciprocal numbers
    (a + b) / 3 + c^2 - 4 * m * n = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l2277_227746


namespace NUMINAMATH_CALUDE_distance_to_incenter_value_l2277_227774

/-- Represents a right isosceles triangle ABC with incenter I -/
structure RightIsoscelesTriangle where
  -- Length of side AB
  side_length : ℝ
  -- Incenter of the triangle
  incenter : ℝ × ℝ

/-- The distance from vertex A to the incenter I in a right isosceles triangle -/
def distance_to_incenter (t : RightIsoscelesTriangle) : ℝ :=
  -- Define the distance calculation here
  sorry

/-- Theorem: In a right isosceles triangle ABC with AB = 6√2, 
    the distance AI from vertex A to the incenter I is 6 - 3√2 -/
theorem distance_to_incenter_value :
  ∀ (t : RightIsoscelesTriangle),
  t.side_length = 6 * Real.sqrt 2 →
  distance_to_incenter t = 6 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_incenter_value_l2277_227774


namespace NUMINAMATH_CALUDE_max_m_inequality_l2277_227741

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, (2/a + 1/b ≥ m/(2*a + b)) → m ≤ 9) ∧ 
  (∃ m : ℝ, m = 9 ∧ 2/a + 1/b ≥ m/(2*a + b)) :=
sorry

end NUMINAMATH_CALUDE_max_m_inequality_l2277_227741


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l2277_227769

theorem alcohol_mixture_percentage (original_volume : ℝ) (water_added : ℝ) (final_percentage : ℝ) :
  original_volume = 11 →
  water_added = 3 →
  final_percentage = 33 →
  (final_percentage / 100) * (original_volume + water_added) = 
    (42 / 100) * original_volume :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l2277_227769


namespace NUMINAMATH_CALUDE_yogurt_combinations_l2277_227788

/- Define the number of flavors and toppings -/
def num_flavors : ℕ := 4
def num_toppings : ℕ := 8

/- Define the function to calculate combinations -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/- Theorem statement -/
theorem yogurt_combinations :
  let no_topping := 1
  let two_toppings := choose num_toppings 2
  let combinations_per_flavor := no_topping + two_toppings
  num_flavors * combinations_per_flavor = 116 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l2277_227788


namespace NUMINAMATH_CALUDE_problem_solution_l2277_227770

theorem problem_solution (a b : ℝ) 
  (h1 : 5 + a = 6 - b) 
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2277_227770


namespace NUMINAMATH_CALUDE_trip_duration_is_six_hours_l2277_227763

/-- Represents the position of a clock hand in minutes (0-59) -/
def ClockPosition : Type := Fin 60

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Fin 24
  minutes : Fin 60

/-- Returns true if the hour and minute hands coincide at the given time -/
def hands_coincide (t : TimeOfDay) : Prop :=
  (t.hours.val * 5 + t.minutes.val / 12 : ℚ) = t.minutes.val

/-- Returns true if the hour and minute hands form a 180° angle at the given time -/
def hands_opposite (t : TimeOfDay) : Prop :=
  ((t.hours.val * 5 + t.minutes.val / 12 : ℚ) + 30) % 60 = t.minutes.val

/-- The start time of the trip -/
def start_time : TimeOfDay :=
  { hours := 8, minutes := 43 }

/-- The end time of the trip -/
def end_time : TimeOfDay :=
  { hours := 14, minutes := 43 }

theorem trip_duration_is_six_hours :
  hands_coincide start_time →
  hands_opposite end_time →
  start_time.hours.val < 9 →
  end_time.hours.val > 14 ∧ end_time.hours.val < 15 →
  (end_time.hours.val - start_time.hours.val : ℕ) = 6 :=
sorry

end NUMINAMATH_CALUDE_trip_duration_is_six_hours_l2277_227763


namespace NUMINAMATH_CALUDE_cricket_count_l2277_227701

theorem cricket_count (initial : Real) (additional : Real) :
  initial = 7.0 → additional = 11.0 → initial + additional = 18.0 := by
  sorry

end NUMINAMATH_CALUDE_cricket_count_l2277_227701


namespace NUMINAMATH_CALUDE_work_completion_time_l2277_227729

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 40

/-- The number of days it takes A and B to complete the work together -/
def ab_days : ℝ := 24

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 60

/-- Theorem stating that if A can do the work in 40 days and A and B together can do it in 24 days, 
    then B can do the work alone in 60 days -/
theorem work_completion_time : 
  (1 / a_days + 1 / b_days = 1 / ab_days) ∧ (b_days = 60) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2277_227729


namespace NUMINAMATH_CALUDE_minimal_point_in_rectangle_l2277_227728

/-- Given positive real numbers a and b, the point (a/2, b/2) minimizes the sum of distances
    to the corners of the rectangle with vertices at (0,0), (a,0), (0,b), and (a,b). -/
theorem minimal_point_in_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x y, 0 < x → x < a → 0 < y → y < b →
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + (b-y)^2) + 
  Real.sqrt ((a-x)^2 + y^2) + Real.sqrt ((a-x)^2 + (b-y)^2) ≥
  Real.sqrt ((a/2)^2 + (b/2)^2) + Real.sqrt ((a/2)^2 + (b/2)^2) + 
  Real.sqrt ((a/2)^2 + (b/2)^2) + Real.sqrt ((a/2)^2 + (b/2)^2) :=
by sorry


end NUMINAMATH_CALUDE_minimal_point_in_rectangle_l2277_227728


namespace NUMINAMATH_CALUDE_right_triangle_area_l2277_227726

theorem right_triangle_area (a c : ℝ) (h1 : a = 30) (h2 : c = 34) : 
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 240 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2277_227726


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2277_227713

theorem complex_power_magnitude : Complex.abs ((2 + Complex.I * Real.sqrt 11) ^ 4) = 225 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2277_227713


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2277_227747

/-- Given an equilateral triangle and an isosceles triangle sharing a side,
    prove that the base of the isosceles triangle is 25 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral : equilateral_perimeter = 60)
  (h_isosceles : isosceles_perimeter = 65)
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) :
  isosceles_base = 25 :=
by
  sorry

#check isosceles_triangle_base_length

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2277_227747


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l2277_227704

theorem cubic_equation_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^3 * (x - 1) * (x - 2) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l2277_227704


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l2277_227794

theorem divisible_by_eleven (n : ℕ) : n < 10 → (123 * 100000 + n * 1000 + 789) % 11 = 0 ↔ n = 10 % 11 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l2277_227794


namespace NUMINAMATH_CALUDE_equation_holds_l2277_227706

theorem equation_holds (x : ℝ) (h : x = 12) : ((17.28 / x) / (3.6 * 0.2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l2277_227706


namespace NUMINAMATH_CALUDE_town_budget_ratio_l2277_227705

theorem town_budget_ratio (total_budget education public_spaces : ℕ) 
  (h1 : total_budget = 32000000)
  (h2 : education = 12000000)
  (h3 : public_spaces = 4000000) :
  (total_budget - education - public_spaces) * 2 = total_budget :=
by sorry

end NUMINAMATH_CALUDE_town_budget_ratio_l2277_227705


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2277_227782

theorem complex_magnitude_equation :
  ∃! (x : ℝ), x > 0 ∧ Complex.abs (x - 3 * Complex.I * Real.sqrt 5) * Complex.abs (8 - 5 * Complex.I) = 50 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2277_227782


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2277_227755

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon (nonagon) contains 27 diagonals -/
theorem nonagon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2277_227755


namespace NUMINAMATH_CALUDE_last_three_digits_of_square_l2277_227765

theorem last_three_digits_of_square (n : ℕ) : ∃ n, n^2 % 1000 = 689 ∧ ¬∃ m, m^2 % 1000 = 759 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_square_l2277_227765


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2277_227768

def number_list : List ℝ := [3, 0, -5, 0.48, -(-7), -|(-8)|, -((-4)^2)]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2277_227768


namespace NUMINAMATH_CALUDE_fifth_derivative_y_l2277_227781

noncomputable def y (x : ℝ) : ℝ := (2 * x^2 - 7) * Real.log (x - 1)

theorem fifth_derivative_y (x : ℝ) (h : x ≠ 1) :
  (deriv^[5] y) x = 8 * (x^2 - 5*x - 11) / (x - 1)^5 :=
by sorry

end NUMINAMATH_CALUDE_fifth_derivative_y_l2277_227781


namespace NUMINAMATH_CALUDE_price_change_l2277_227721

theorem price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease := P * (1 - 0.2)
  let price_after_increase := price_after_decrease * (1 + 0.5)
  price_after_increase = P * 1.2 := by
sorry

end NUMINAMATH_CALUDE_price_change_l2277_227721


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2277_227736

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of its first n terms,
    if S_5, S_4, and S_6 form an arithmetic sequence, then q = -2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with common ratio q
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- S_n is the sum of first n terms
  2 * S 4 = S 5 + S 6 →  -- S_5, S_4, and S_6 form an arithmetic sequence
  q = -2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2277_227736


namespace NUMINAMATH_CALUDE_hybrid_car_trip_length_l2277_227740

theorem hybrid_car_trip_length 
  (battery_distance : ℝ) 
  (gasoline_consumption_rate : ℝ) 
  (average_efficiency : ℝ) :
  battery_distance = 75 →
  gasoline_consumption_rate = 0.05 →
  average_efficiency = 50 →
  ∃ (total_distance : ℝ),
    total_distance = 125 ∧
    average_efficiency = total_distance / (gasoline_consumption_rate * (total_distance - battery_distance)) :=
by
  sorry

end NUMINAMATH_CALUDE_hybrid_car_trip_length_l2277_227740


namespace NUMINAMATH_CALUDE_granola_cost_per_bag_l2277_227722

theorem granola_cost_per_bag 
  (total_bags : ℕ) 
  (full_price_bags : ℕ) 
  (full_price : ℚ) 
  (discounted_bags : ℕ) 
  (discounted_price : ℚ) 
  (net_profit : ℚ) 
  (h1 : total_bags = 20)
  (h2 : full_price_bags = 15)
  (h3 : full_price = 6)
  (h4 : discounted_bags = 5)
  (h5 : discounted_price = 4)
  (h6 : net_profit = 50)
  (h7 : total_bags = full_price_bags + discounted_bags) :
  (full_price_bags * full_price + discounted_bags * discounted_price - net_profit) / total_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_granola_cost_per_bag_l2277_227722


namespace NUMINAMATH_CALUDE_min_value_of_sum_roots_l2277_227733

theorem min_value_of_sum_roots (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hsum : a + b + c + d = 1) : 
  Real.sqrt (a^2 + 1/(8*a)) + Real.sqrt (b^2 + 1/(8*b)) + 
  Real.sqrt (c^2 + 1/(8*c)) + Real.sqrt (d^2 + 1/(8*d)) ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_roots_l2277_227733


namespace NUMINAMATH_CALUDE_gorillas_sent_to_different_zoo_l2277_227760

theorem gorillas_sent_to_different_zoo :
  let initial_animals : ℕ := 68
  let hippopotamus : ℕ := 1
  let rhinos : ℕ := 3
  let lion_cubs : ℕ := 8
  let meerkats : ℕ := 2 * lion_cubs
  let final_animals : ℕ := 90
  let gorillas_sent : ℕ := initial_animals + hippopotamus + rhinos + lion_cubs + meerkats - final_animals
  gorillas_sent = 6 :=
by sorry

end NUMINAMATH_CALUDE_gorillas_sent_to_different_zoo_l2277_227760


namespace NUMINAMATH_CALUDE_distance_between_4th_and_30th_red_l2277_227737

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Represents the cyclic pattern of lights -/
def lightPattern : List LightColor := 
  [LightColor.Red, LightColor.Red, LightColor.Red, 
   LightColor.Green, LightColor.Green, LightColor.Green, LightColor.Green]

/-- The distance between each light in inches -/
def lightDistance : ℕ := 8

/-- Calculates the position of the nth red light -/
def nthRedLightPosition (n : ℕ) : ℕ := sorry

/-- Calculates the distance between two positions in feet -/
def distanceInFeet (pos1 pos2 : ℕ) : ℚ := sorry

/-- Theorem: The distance between the 4th and 30th red light is 41.33 feet -/
theorem distance_between_4th_and_30th_red : 
  distanceInFeet (nthRedLightPosition 4) (nthRedLightPosition 30) = 41.33 := by sorry

end NUMINAMATH_CALUDE_distance_between_4th_and_30th_red_l2277_227737


namespace NUMINAMATH_CALUDE_intersection_distance_squared_example_l2277_227739

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculate the square of the distance between intersection points of two circles -/
def intersection_distance_squared (c1 c2 : Circle) : ℝ :=
  let x1 := c1.center.1
  let y1 := c1.center.2
  let x2 := c2.center.1
  let y2 := c2.center.2
  let r1 := c1.radius
  let r2 := c2.radius
  -- Calculate the square of the distance between intersection points
  sorry

theorem intersection_distance_squared_example : 
  let c1 : Circle := ⟨(3, -2), 5⟩
  let c2 : Circle := ⟨(3, 4), Real.sqrt 13⟩
  intersection_distance_squared c1 c2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_example_l2277_227739


namespace NUMINAMATH_CALUDE_simplify_expression_l2277_227793

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2277_227793


namespace NUMINAMATH_CALUDE_toys_sold_l2277_227715

/-- Given a selling price, cost price per toy, and a gain equal to the cost of 3 toys,
    prove that the number of toys sold is 18. -/
theorem toys_sold (selling_price : ℕ) (cost_per_toy : ℕ) (h1 : selling_price = 18900) 
    (h2 : cost_per_toy = 900) : 
  (selling_price - 3 * cost_per_toy) / cost_per_toy = 18 := by
  sorry

end NUMINAMATH_CALUDE_toys_sold_l2277_227715


namespace NUMINAMATH_CALUDE_hexagon_division_divisible_by_three_l2277_227732

/-- A regular hexagon divided into congruent parallelograms -/
structure HexagonDivision where
  /-- The number of congruent parallelograms -/
  num_parallelograms : ℕ
  /-- Assertion that the hexagon is divided into this many congruent parallelograms -/
  is_valid_division : num_parallelograms > 0

/-- Theorem stating that the number of parallelograms in a valid hexagon division is divisible by 3 -/
theorem hexagon_division_divisible_by_three (h : HexagonDivision) : 
  3 ∣ h.num_parallelograms :=
sorry

end NUMINAMATH_CALUDE_hexagon_division_divisible_by_three_l2277_227732


namespace NUMINAMATH_CALUDE_ellipse_properties_l2277_227711

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the semi-focal distance
def semi_focal_distance : ℝ := 1

-- Define the condition that a > b > 0
def size_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0

-- Define the condition that the circle with diameter F₁F₂ passes through upper and lower vertices
def circle_condition (a b : ℝ) : Prop :=
  2 * semi_focal_distance = a

-- Theorem statement
theorem ellipse_properties (a b : ℝ) 
  (h1 : size_condition a b) 
  (h2 : circle_condition a b) :
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ k : ℝ, -Real.sqrt 2 / 2 < k ∧ k < 0 ∧
    ∀ x y : ℝ, y = k * (x - 2) → ellipse a b x y → y > 0 → x = 2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2277_227711


namespace NUMINAMATH_CALUDE_radius_of_Q_l2277_227752

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
axiom P : Circle
axiom Q : Circle
axiom R : Circle
axiom S : Circle

-- Define the conditions
axiom externally_tangent : P.radius + Q.radius = dist P.center Q.center ∧
                           P.radius + R.radius = dist P.center R.center ∧
                           Q.radius + R.radius = dist Q.center R.center

axiom internally_tangent : S.radius = P.radius + dist P.center S.center ∧
                           S.radius = Q.radius + dist Q.center S.center ∧
                           S.radius = R.radius + dist R.center S.center

axiom Q_R_congruent : Q.radius = R.radius

axiom P_radius : P.radius = 2

axiom P_through_S_center : dist P.center S.center = P.radius

-- Theorem to prove
theorem radius_of_Q : Q.radius = 16/9 := by sorry

end NUMINAMATH_CALUDE_radius_of_Q_l2277_227752


namespace NUMINAMATH_CALUDE_linear_function_condition_l2277_227773

/-- A linear function f(x) = ax + b satisfying f⁽¹⁰⁾(x) ≥ 1024x + 1023 
    must have a = 2 and b ≥ 1, or a = -2 and b ≤ -3 -/
theorem linear_function_condition (a b : ℝ) (h : ∀ x, a^10 * x + b * (a^10 - 1) / (a - 1) ≥ 1024 * x + 1023) :
  (a = 2 ∧ b ≥ 1) ∨ (a = -2 ∧ b ≤ -3) := by sorry

end NUMINAMATH_CALUDE_linear_function_condition_l2277_227773


namespace NUMINAMATH_CALUDE_sum_of_squares_with_means_l2277_227745

/-- Given three positive real numbers with specific arithmetic, geometric, and harmonic means, 
    prove that the sum of their squares equals 385.5 -/
theorem sum_of_squares_with_means (x y z : ℝ) 
    (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
    (h_arithmetic : (x + y + z) / 3 = 10)
    (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
    (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_means_l2277_227745


namespace NUMINAMATH_CALUDE_no_integer_divisible_by_289_l2277_227710

theorem no_integer_divisible_by_289 :
  ∀ a : ℤ, ¬(289 ∣ (a^2 - 3*a - 19)) := by
sorry

end NUMINAMATH_CALUDE_no_integer_divisible_by_289_l2277_227710
