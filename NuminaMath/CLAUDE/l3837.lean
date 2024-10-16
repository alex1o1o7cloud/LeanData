import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3837_383725

/-- Arithmetic sequence a_n with a₁ = 8 and a₃ = 4 -/
def a (n : ℕ) : ℚ :=
  8 - 2 * (n - 1)

/-- Sum of first n terms of a_n -/
def S (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- b_n sequence -/
def b (n : ℕ+) : ℚ :=
  1 / ((n : ℚ) * (12 - a n))

/-- Sum of first n terms of b_n -/
def T (n : ℕ+) : ℚ :=
  (n : ℚ) / (2 * (n + 1))

theorem arithmetic_sequence_properties :
  (∃ n : ℕ, S n = 20 ∧ ∀ m : ℕ, S m ≤ S n) ∧
  (∀ n : ℕ+, T n = (n : ℚ) / (2 * (n + 1))) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3837_383725


namespace NUMINAMATH_CALUDE_min_value_a_l3837_383748

theorem min_value_a (a : ℝ) : 
  (a > 0 ∧ ∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) → a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l3837_383748


namespace NUMINAMATH_CALUDE_evaluate_expression_l3837_383719

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 5
  (2*x - a + 4) = (a + 14) := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3837_383719


namespace NUMINAMATH_CALUDE_frog_hop_probability_l3837_383755

-- Define the grid
def Grid := Fin 3 × Fin 3

-- Define corner squares
def is_corner (pos : Grid) : Prop :=
  (pos.1 = 0 ∧ pos.2 = 0) ∨ (pos.1 = 0 ∧ pos.2 = 2) ∨
  (pos.1 = 2 ∧ pos.2 = 0) ∨ (pos.1 = 2 ∧ pos.2 = 2)

-- Define center square
def center : Grid := (1, 1)

-- Define a single hop
def hop (pos : Grid) : Grid := sorry

-- Define the probability of reaching a corner in exactly n hops
def prob_corner_in (n : Nat) (start : Grid) : ℚ := sorry

-- Main theorem
theorem frog_hop_probability :
  prob_corner_in 2 center + prob_corner_in 3 center + prob_corner_in 4 center = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l3837_383755


namespace NUMINAMATH_CALUDE_difference_of_squares_102_99_l3837_383798

theorem difference_of_squares_102_99 : 102^2 - 99^2 = 603 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_102_99_l3837_383798


namespace NUMINAMATH_CALUDE_line_equation_proof_l3837_383770

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equation_proof (l : Line) :
  l.contains (0, 3) ∧
  l.perpendicular ⟨1, 1, 1⟩ →
  l = ⟨1, -1, 3⟩ := by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_line_equation_proof_l3837_383770


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l3837_383717

/-- Represents the remaining oil quantity in a tank as a function of time -/
def Q (t : ℝ) : ℝ := 20 - 0.2 * t

/-- The initial oil quantity in the tank -/
def initial_quantity : ℝ := 20

/-- The rate at which oil flows out of the tank (in liters per minute) -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct :
  ∀ t : ℝ, t ≥ 0 →
  Q t = initial_quantity - flow_rate * t ∧
  Q t ≥ 0 ∧
  (Q t = 0 → t = initial_quantity / flow_rate) :=
sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l3837_383717


namespace NUMINAMATH_CALUDE_story_problem_solution_l3837_383742

/-- Represents the story writing problem with given parameters -/
structure StoryProblem where
  total_words : ℕ
  num_chapters : ℕ
  total_vocab_terms : ℕ
  vocab_distribution : Fin 4 → ℕ
  words_per_line : ℕ
  lines_per_page : ℕ
  pages_filled : ℚ

/-- Calculates the number of words left to write given a StoryProblem -/
def words_left_to_write (problem : StoryProblem) : ℕ :=
  problem.total_words - (problem.words_per_line * problem.lines_per_page * problem.pages_filled.num / problem.pages_filled.den).toNat

/-- Theorem stating that given the specific problem conditions, 100 words are left to write -/
theorem story_problem_solution (problem : StoryProblem) 
  (h1 : problem.total_words = 400)
  (h2 : problem.num_chapters = 4)
  (h3 : problem.total_vocab_terms = 20)
  (h4 : problem.vocab_distribution 0 = 8)
  (h5 : problem.vocab_distribution 1 = 4)
  (h6 : problem.vocab_distribution 2 = 6)
  (h7 : problem.vocab_distribution 3 = 2)
  (h8 : problem.words_per_line = 10)
  (h9 : problem.lines_per_page = 20)
  (h10 : problem.pages_filled = 3/2) :
  words_left_to_write problem = 100 := by
  sorry


end NUMINAMATH_CALUDE_story_problem_solution_l3837_383742


namespace NUMINAMATH_CALUDE_neil_charge_theorem_l3837_383706

def trim_cost : ℕ → ℝ := λ n => 5 * n
def shape_cost : ℕ → ℝ := λ n => 15 * n

theorem neil_charge_theorem (num_trim : ℕ) (num_shape : ℕ) 
  (h1 : num_trim = 30) (h2 : num_shape = 4) : 
  trim_cost num_trim + shape_cost num_shape = 210 := by
  sorry

end NUMINAMATH_CALUDE_neil_charge_theorem_l3837_383706


namespace NUMINAMATH_CALUDE_square_ceiling_lights_l3837_383791

/-- The number of lights on each side of the square ceiling -/
def lights_per_side : ℕ := 20

/-- The minimum number of lights needed for the entire square ceiling -/
def min_lights_needed : ℕ := 4 * lights_per_side - 4

theorem square_ceiling_lights : min_lights_needed = 76 := by
  sorry

end NUMINAMATH_CALUDE_square_ceiling_lights_l3837_383791


namespace NUMINAMATH_CALUDE_polynomial_with_specific_roots_l3837_383788

theorem polynomial_with_specific_roots :
  ∃ (P : ℂ → ℂ) (r s : ℤ),
    (∀ x, P x = x^4 + (a : ℤ) * x^3 + (b : ℤ) * x^2 + (c : ℤ) * x + (d : ℤ)) ∧
    (P r = 0) ∧ (P s = 0) ∧ (P ((1 + Complex.I * Real.sqrt 15) / 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_with_specific_roots_l3837_383788


namespace NUMINAMATH_CALUDE_area_triangle_DEF_l3837_383795

/-- Triangle DEF with hypotenuse DE, angle between DF and DE is 45°, and length of DF is 4 units -/
structure Triangle_DEF where
  DE : ℝ  -- Length of hypotenuse DE
  DF : ℝ  -- Length of side DF
  EF : ℝ  -- Length of side EF
  angle_DF_DE : ℝ  -- Angle between DF and DE in radians
  hypotenuse_DE : DE = DF * Real.sqrt 2  -- DE is hypotenuse
  angle_45_deg : angle_DF_DE = π / 4  -- Angle is 45°
  DF_length : DF = 4  -- Length of DF is 4 units

/-- The area of triangle DEF is 8 square units -/
theorem area_triangle_DEF (t : Triangle_DEF) : (1 / 2) * t.DF * t.EF = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_DEF_l3837_383795


namespace NUMINAMATH_CALUDE_waiter_customers_proof_l3837_383782

/-- Calculates the number of remaining customers for a waiter given the initial number of tables,
    number of tables that left, and number of customers per table. -/
def remaining_customers (initial_tables : ℝ) (tables_left : ℝ) (customers_per_table : ℝ) : ℝ :=
  (initial_tables - tables_left) * customers_per_table

/-- Proves that the number of remaining customers for a waiter with 44.0 initial tables,
    12.0 tables that left, and 8.0 customers per table is 256.0. -/
theorem waiter_customers_proof :
  remaining_customers 44.0 12.0 8.0 = 256.0 := by
  sorry

#eval remaining_customers 44.0 12.0 8.0

end NUMINAMATH_CALUDE_waiter_customers_proof_l3837_383782


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l3837_383712

/-- The volume of a round swimming pool with given dimensions -/
theorem swimming_pool_volume (diameter : ℝ) (depth_start : ℝ) (depth_end : ℝ) :
  diameter = 20 →
  depth_start = 3 →
  depth_end = 6 →
  (π * (diameter / 2)^2 * ((depth_start + depth_end) / 2) : ℝ) = 450 * π := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l3837_383712


namespace NUMINAMATH_CALUDE_dinosaur_book_cost_l3837_383790

def dictionary_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def total_cost : ℕ := 37

theorem dinosaur_book_cost :
  ∃ (dinosaur_cost : ℕ), 
    dictionary_cost + dinosaur_cost + cookbook_cost = total_cost ∧
    dinosaur_cost = 19 :=
by sorry

end NUMINAMATH_CALUDE_dinosaur_book_cost_l3837_383790


namespace NUMINAMATH_CALUDE_range_of_g_on_large_interval_l3837_383735

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def range_of (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc a b, f x = y}

theorem range_of_g_on_large_interval
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_periodic : is_periodic f 1)
  (h_g_def : ∀ x, g x = f x + 2 * x)
  (h_range_small : range_of g 1 2 = Set.Icc (-1) 5) :
  range_of g (-2020) 2020 = Set.Icc (-4043) 4041 := by
sorry

end NUMINAMATH_CALUDE_range_of_g_on_large_interval_l3837_383735


namespace NUMINAMATH_CALUDE_exam_questions_count_l3837_383781

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


end NUMINAMATH_CALUDE_exam_questions_count_l3837_383781


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3837_383783

theorem digit_sum_problem (x y z w : ℕ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 →
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x < 10 ∧ y < 10 ∧ z < 10 ∧ w < 10 →
  100 * x + 10 * y + w + 100 * z + 10 * w + x = 1000 →
  x + y + z + w = 18 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3837_383783


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l3837_383750

open Real

theorem sin_2x_derivative (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x)
  (deriv f) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l3837_383750


namespace NUMINAMATH_CALUDE_total_silver_dollars_l3837_383732

theorem total_silver_dollars (chiu phung ha lin : ℕ) : 
  chiu = 56 →
  phung = chiu + 16 →
  ha = phung + 5 →
  lin = (chiu + phung + ha) + 25 →
  chiu + phung + ha + lin = 435 :=
by sorry

end NUMINAMATH_CALUDE_total_silver_dollars_l3837_383732


namespace NUMINAMATH_CALUDE_scale_and_rotate_complex_l3837_383723

/-- Represents a complex number rotation by 270° clockwise -/
def rotate270Clockwise (z : ℂ) : ℂ := Complex.I * z

/-- Proves that scaling -8 - 4i by 2 and then rotating 270° clockwise results in 8 - 16i -/
theorem scale_and_rotate_complex : 
  let z : ℂ := -8 - 4 * Complex.I
  let scaled : ℂ := 2 * z
  rotate270Clockwise scaled = 8 - 16 * Complex.I := by sorry

end NUMINAMATH_CALUDE_scale_and_rotate_complex_l3837_383723


namespace NUMINAMATH_CALUDE_first_fibonacci_exceeding_1000_l3837_383765

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem first_fibonacci_exceeding_1000 :
  ∃ n : ℕ, fibonacci n > 1000 ∧ ∀ m : ℕ, m < n → fibonacci m ≤ 1000 ∧ fibonacci n = 1597 :=
by
  sorry

end NUMINAMATH_CALUDE_first_fibonacci_exceeding_1000_l3837_383765


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3837_383736

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^12 - x^6 + 1 = (x^2 - 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3837_383736


namespace NUMINAMATH_CALUDE_harrys_morning_routine_time_l3837_383715

/-- Harry's morning routine time calculation -/
theorem harrys_morning_routine_time :
  let buying_time : ℕ := 15
  let eating_time : ℕ := 2 * buying_time
  let total_time : ℕ := buying_time + eating_time
  total_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_harrys_morning_routine_time_l3837_383715


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l3837_383753

theorem laptop_sticker_price :
  ∀ (x : ℝ),
  (0.8 * x - 100 = 0.7 * x - 20) →
  x = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l3837_383753


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l3837_383743

/-- Given two circles C₁ and C₂ defined by equations (x-12)²+y²=25 and (x+18)²+y²=64 respectively,
    the length of the shortest line segment RS tangent to C₁ at R and C₂ at S is 339/13. -/
theorem shortest_tangent_length (C₁ C₂ : Set (ℝ × ℝ)) (R S : ℝ × ℝ) :
  C₁ = {p : ℝ × ℝ | (p.1 - 12)^2 + p.2^2 = 25} →
  C₂ = {p : ℝ × ℝ | (p.1 + 18)^2 + p.2^2 = 64} →
  R ∈ C₁ →
  S ∈ C₂ →
  (∀ p ∈ C₁, (R.1 - p.1) * (R.1 - 12) + (R.2 - p.2) * R.2 = 0) →
  (∀ p ∈ C₂, (S.1 - p.1) * (S.1 + 18) + (S.2 - p.2) * S.2 = 0) →
  (∀ T U : ℝ × ℝ, T ∈ C₁ → U ∈ C₂ → 
    (∀ q ∈ C₁, (T.1 - q.1) * (T.1 - 12) + (T.2 - q.2) * T.2 = 0) →
    (∀ q ∈ C₂, (U.1 - q.1) * (U.1 + 18) + (U.2 - q.2) * U.2 = 0) →
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) ≤ Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2)) →
  Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 339 / 13 :=
by sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l3837_383743


namespace NUMINAMATH_CALUDE_house_c_to_a_ratio_l3837_383772

/-- Represents the real estate problem with Nigella's sales --/
structure RealEstateProblem where
  base_salary : ℝ
  commission_rate : ℝ
  houses_sold : ℕ
  total_earnings : ℝ
  house_a_cost : ℝ
  house_b_cost : ℝ
  house_c_cost : ℝ

/-- Theorem stating the ratio of House C's cost to House A's cost before subtracting $110,000 --/
theorem house_c_to_a_ratio (problem : RealEstateProblem)
  (h1 : problem.base_salary = 3000)
  (h2 : problem.commission_rate = 0.02)
  (h3 : problem.houses_sold = 3)
  (h4 : problem.total_earnings = 8000)
  (h5 : problem.house_b_cost = 3 * problem.house_a_cost)
  (h6 : problem.house_c_cost = problem.house_a_cost * 2 - 110000)
  (h7 : problem.house_a_cost = 60000) :
  (problem.house_c_cost + 110000) / problem.house_a_cost = 2 := by
  sorry


end NUMINAMATH_CALUDE_house_c_to_a_ratio_l3837_383772


namespace NUMINAMATH_CALUDE_consecutive_squares_equality_l3837_383704

theorem consecutive_squares_equality :
  ∃ (a b c d : ℝ), (b = a + 1 ∧ c = b + 1 ∧ d = c + 1) ∧ (a^2 + b^2 = c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_equality_l3837_383704


namespace NUMINAMATH_CALUDE_inequality_contradiction_l3837_383792

theorem inequality_contradiction (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬(a + b < c + d ∧ (a + b) * c * d < a * b * (c + d) ∧ (a + b) * (c + d) < a * b + c * d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l3837_383792


namespace NUMINAMATH_CALUDE_binomial_20_19_l3837_383721

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l3837_383721


namespace NUMINAMATH_CALUDE_picnic_group_size_l3837_383722

theorem picnic_group_size (initial_avg : ℝ) (new_persons : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_persons = 12 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ n : ℕ, n * initial_avg + new_persons * new_avg = (n + new_persons) * final_avg ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_picnic_group_size_l3837_383722


namespace NUMINAMATH_CALUDE_sector_area_l3837_383766

/-- The area of a circular sector with central angle π/3 and radius 2 is 2π/3 -/
theorem sector_area (α : Real) (r : Real) (h1 : α = π / 3) (h2 : r = 2) :
  (1 / 2) * α * r^2 = 2 * π / 3 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l3837_383766


namespace NUMINAMATH_CALUDE_cubic_diophantine_equation_solution_l3837_383718

theorem cubic_diophantine_equation_solution :
  ∀ x y : ℕ+, x^3 - y^3 = x * y + 61 → (x = 6 ∧ y = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_diophantine_equation_solution_l3837_383718


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l3837_383744

theorem sum_and_ratio_to_difference (a b : ℝ) 
  (h1 : a + b = 500) 
  (h2 : a / b = 0.8) : 
  b - a = 100 / 1.8 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l3837_383744


namespace NUMINAMATH_CALUDE_division_problem_l3837_383730

theorem division_problem (number : ℕ) : 
  (number / 25 = 5) ∧ (number % 25 = 2) → number = 127 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3837_383730


namespace NUMINAMATH_CALUDE_triangle_property_l3837_383710

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.a / t.b * Real.cos t.C + t.c / (2 * t.b) = 1) 
  (h2 : t.A + t.B + t.C = π) 
  (h3 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0) 
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) :
  (t.A = π / 3) ∧ 
  (t.a = 1 → ∃ l : Real, l > 2 ∧ l ≤ 3 ∧ l = t.a + t.b + t.c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l3837_383710


namespace NUMINAMATH_CALUDE_max_subgrid_sum_l3837_383749

/-- Represents a 5x5 grid filled with integers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if all numbers in the grid are unique and between 1 and 25 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 25 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Calculates the sum of a 2x2 subgrid starting at (i, j) -/
def subgrid_sum (g : Grid) (i j : Fin 4) : ℕ :=
  g i j + g i (j+1) + g (i+1) j + g (i+1) (j+1)

/-- The main theorem -/
theorem max_subgrid_sum (g : Grid) (h : valid_grid g) :
  (∀ i j : Fin 4, 45 ≤ subgrid_sum g i j) ∧
  ¬∃ N > 45, ∀ i j : Fin 4, N ≤ subgrid_sum g i j :=
sorry

end NUMINAMATH_CALUDE_max_subgrid_sum_l3837_383749


namespace NUMINAMATH_CALUDE_todd_ingredient_cost_l3837_383796

/-- Represents the financial details of Todd's snow-cone business --/
structure SnowConeBusiness where
  borrowed : ℝ
  repay : ℝ
  snowConesSold : ℕ
  pricePerSnowCone : ℝ
  remainingAfterRepay : ℝ

/-- Calculates the amount spent on ingredients for the snow-cone business --/
def ingredientCost (business : SnowConeBusiness) : ℝ :=
  business.borrowed + business.snowConesSold * business.pricePerSnowCone - business.repay - business.remainingAfterRepay

/-- Theorem stating that Todd spent $25 on ingredients --/
theorem todd_ingredient_cost :
  let business : SnowConeBusiness := {
    borrowed := 100,
    repay := 110,
    snowConesSold := 200,
    pricePerSnowCone := 0.75,
    remainingAfterRepay := 65
  }
  ingredientCost business = 25 := by
  sorry


end NUMINAMATH_CALUDE_todd_ingredient_cost_l3837_383796


namespace NUMINAMATH_CALUDE_passing_percentage_l3837_383757

theorem passing_percentage (max_marks : ℕ) (pradeep_marks : ℕ) (failed_by : ℕ) 
  (h1 : max_marks = 925)
  (h2 : pradeep_marks = 160)
  (h3 : failed_by = 25) :
  (((pradeep_marks + failed_by : ℚ) / max_marks) * 100 : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l3837_383757


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3837_383739

theorem arithmetic_expression_evaluation :
  (∀ x y z : ℤ, x + y = z → (x = 6 ∧ y = -13 ∧ z = -7) ∨ (x = -5 ∧ y = -3 ∧ z = -8)) ∧
  (6 + (-13) = -7) ∧
  (6 + (-13) ≠ 7) ∧
  (6 + (-13) ≠ -19) ∧
  (-5 + (-3) ≠ 8) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3837_383739


namespace NUMINAMATH_CALUDE_triangle_height_l3837_383713

/-- Given a triangle with base 18 meters and area 54 square meters, prove its height is 6 meters -/
theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 18 → area = 54 → area = (base * height) / 2 → height = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l3837_383713


namespace NUMINAMATH_CALUDE_recurring_decimal_ratio_l3837_383703

-- Define the recurring decimals
def recurring_81 : ℚ := 81 / 99
def recurring_54 : ℚ := 54 / 99

-- State the theorem
theorem recurring_decimal_ratio :
  recurring_81 / recurring_54 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_ratio_l3837_383703


namespace NUMINAMATH_CALUDE_safety_cost_per_mile_approx_l3837_383776

/-- Safety Rent-a-Car's daily rate -/
def safety_daily_rate : ℝ := 21.95

/-- City Rentals' daily rate -/
def city_daily_rate : ℝ := 18.95

/-- City Rentals' cost per mile -/
def city_cost_per_mile : ℝ := 0.21

/-- Number of miles for equal cost -/
def equal_cost_miles : ℝ := 150.0

/-- Safety Rent-a-Car's cost per mile -/
noncomputable def safety_cost_per_mile : ℝ := 
  (city_daily_rate + city_cost_per_mile * equal_cost_miles - safety_daily_rate) / equal_cost_miles

theorem safety_cost_per_mile_approx :
  ∃ ε > 0, abs (safety_cost_per_mile - 0.177) < ε ∧ ε < 0.001 :=
sorry

end NUMINAMATH_CALUDE_safety_cost_per_mile_approx_l3837_383776


namespace NUMINAMATH_CALUDE_jeffs_weekly_running_time_l3837_383734

/-- Represents Jeff's weekly running schedule -/
structure RunningSchedule where
  normalDays : Nat  -- Number of days with normal running time
  normalTime : Nat  -- Normal running time in minutes
  thursdayReduction : Nat  -- Minutes reduced on Thursday
  fridayIncrease : Nat  -- Minutes increased on Friday

/-- Calculates the total running time for the week given a RunningSchedule -/
def totalRunningTime (schedule : RunningSchedule) : Nat :=
  schedule.normalDays * schedule.normalTime +
  (schedule.normalTime - schedule.thursdayReduction) +
  (schedule.normalTime + schedule.fridayIncrease)

/-- Theorem stating that Jeff's total running time for the week is 290 minutes -/
theorem jeffs_weekly_running_time :
  ∀ (schedule : RunningSchedule),
    schedule.normalDays = 3 ∧
    schedule.normalTime = 60 ∧
    schedule.thursdayReduction = 20 ∧
    schedule.fridayIncrease = 10 →
    totalRunningTime schedule = 290 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_weekly_running_time_l3837_383734


namespace NUMINAMATH_CALUDE_image_of_two_is_five_l3837_383705

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem image_of_two_is_five : f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_image_of_two_is_five_l3837_383705


namespace NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l3837_383745

theorem half_plus_seven_equals_seventeen (n : ℝ) : (1/2 * n + 7 = 17) → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l3837_383745


namespace NUMINAMATH_CALUDE_john_got_36_rolls_l3837_383759

/-- The number of rolls John got given the price and amount spent -/
def rolls_bought (price_per_dozen : ℚ) (amount_spent : ℚ) : ℚ :=
  (amount_spent / price_per_dozen) * 12

/-- Theorem: John got 36 rolls -/
theorem john_got_36_rolls :
  rolls_bought 5 15 = 36 := by
  sorry

end NUMINAMATH_CALUDE_john_got_36_rolls_l3837_383759


namespace NUMINAMATH_CALUDE_density_not_vector_l3837_383738

/-- A type representing physical quantities --/
inductive PhysicalQuantity
| Buoyancy
| WindSpeed
| Displacement
| Density

/-- Definition of a vector --/
def isVector (q : PhysicalQuantity) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ × ℝ × ℝ), True

/-- Theorem stating that density is not a vector --/
theorem density_not_vector : ¬ isVector PhysicalQuantity.Density := by
  sorry

end NUMINAMATH_CALUDE_density_not_vector_l3837_383738


namespace NUMINAMATH_CALUDE_jake_monday_sales_l3837_383762

/-- The number of candy pieces Jake sold on Monday -/
def monday_sales : ℕ := 15

/-- The initial number of candy pieces Jake had -/
def initial_candy : ℕ := 80

/-- The number of candy pieces Jake sold on Tuesday -/
def tuesday_sales : ℕ := 58

/-- The number of candy pieces Jake had left by Wednesday -/
def wednesday_leftover : ℕ := 7

/-- Theorem stating that the number of candy pieces Jake sold on Monday is 15 -/
theorem jake_monday_sales : 
  monday_sales = initial_candy - tuesday_sales - wednesday_leftover := by
  sorry

end NUMINAMATH_CALUDE_jake_monday_sales_l3837_383762


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3837_383775

-- Define the polynomials A, B, and C
def A (x : ℝ) : ℝ := 5 * x^2 + 4 * x - 1
def B (x : ℝ) : ℝ := -x^2 - 3 * x + 3
def C (x : ℝ) : ℝ := 8 - 7 * x - 6 * x^2

-- Theorem statement
theorem polynomial_simplification (x : ℝ) : A x - B x + C x = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3837_383775


namespace NUMINAMATH_CALUDE_paper_length_equals_days_until_due_l3837_383773

/-- The number of pages in Stacy's history paper -/
def paper_length : ℕ := sorry

/-- The number of days until the paper is due -/
def days_until_due : ℕ := 12

/-- The number of pages Stacy needs to write per day to finish on time -/
def pages_per_day : ℕ := 1

/-- Theorem stating that the paper length is equal to the number of days until due -/
theorem paper_length_equals_days_until_due : 
  paper_length = days_until_due := by sorry

end NUMINAMATH_CALUDE_paper_length_equals_days_until_due_l3837_383773


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3837_383711

theorem modulus_of_complex_number (z : ℂ) : z = 2 / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3837_383711


namespace NUMINAMATH_CALUDE_desired_depth_calculation_l3837_383746

/-- Calculates the desired depth to be dug given initial and new working conditions -/
theorem desired_depth_calculation
  (initial_men : ℕ)
  (initial_hours : ℕ)
  (initial_depth : ℝ)
  (new_hours : ℕ)
  (extra_men : ℕ)
  (h1 : initial_men = 72)
  (h2 : initial_hours = 8)
  (h3 : initial_depth = 30)
  (h4 : new_hours = 6)
  (h5 : extra_men = 88)
  : ∃ (desired_depth : ℝ), desired_depth = 50 := by
  sorry


end NUMINAMATH_CALUDE_desired_depth_calculation_l3837_383746


namespace NUMINAMATH_CALUDE_compare_powers_l3837_383754

theorem compare_powers : 9^61 < 27^41 ∧ 27^41 < 81^31 := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l3837_383754


namespace NUMINAMATH_CALUDE_complement_intersect_equal_l3837_383702

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}

theorem complement_intersect_equal : (U \ B) ∩ A = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersect_equal_l3837_383702


namespace NUMINAMATH_CALUDE_vegetable_selection_theorem_l3837_383767

/-- Represents a basket containing vegetables -/
structure Basket :=
  (cucumbers : ℕ)
  (eggplants : ℕ)
  (tomatoes : ℕ)

/-- The total number of baskets -/
def total_baskets : ℕ := 50

/-- The number of baskets to be selected -/
def selected_baskets : ℕ := 26

/-- Checks if a subset of baskets contains at least half of each vegetable type -/
def contains_half (all_baskets : Finset Basket) (subset : Finset Basket) : Prop :=
  let total := all_baskets.sum (λ b => b.cucumbers)
  let subset_sum := subset.sum (λ b => b.cucumbers)
  subset_sum * 2 ≥ total ∧
  (all_baskets.sum (λ b => b.eggplants) ≤ 2 * subset.sum (λ b => b.eggplants)) ∧
  (all_baskets.sum (λ b => b.tomatoes) ≤ 2 * subset.sum (λ b => b.tomatoes))

theorem vegetable_selection_theorem (baskets : Finset Basket) 
    (h : baskets.card = total_baskets) :
    ∃ (subset : Finset Basket), subset.card = selected_baskets ∧ 
    contains_half baskets subset := by
  sorry

end NUMINAMATH_CALUDE_vegetable_selection_theorem_l3837_383767


namespace NUMINAMATH_CALUDE_eleven_only_divisor_l3837_383786

theorem eleven_only_divisor : ∃! n : ℕ, 
  (∃ k : ℕ, n = (10^k - 1) / 9) ∧ 
  (∃ m : ℕ, (10^m + 1) % n = 0) ∧
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_eleven_only_divisor_l3837_383786


namespace NUMINAMATH_CALUDE_number_of_pupils_l3837_383747

theorem number_of_pupils (total : ℕ) (parents : ℕ) (teachers : ℕ) 
  (h1 : total = 1541)
  (h2 : parents = 73)
  (h3 : teachers = 744) :
  total - (parents + teachers) = 724 := by
sorry

end NUMINAMATH_CALUDE_number_of_pupils_l3837_383747


namespace NUMINAMATH_CALUDE_range_of_m_l3837_383751

/-- Given an increasing function f on ℝ and the condition f(m^2) > f(-m),
    the range of m is (-∞, -1) ∪ (0, +∞) -/
theorem range_of_m (f : ℝ → ℝ) (h_incr : Monotone f) (m : ℝ) (h_cond : f (m^2) > f (-m)) :
  m ∈ Set.Iio (-1) ∪ Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3837_383751


namespace NUMINAMATH_CALUDE_workshop_workers_l3837_383729

/-- Proves that the total number of workers is 12 given the conditions in the problem -/
theorem workshop_workers (total_average : ℕ) (tech_average : ℕ) (non_tech_average : ℕ) 
  (num_technicians : ℕ) (h1 : total_average = 9000) (h2 : tech_average = 12000) 
  (h3 : non_tech_average = 6000) (h4 : num_technicians = 6) : 
  ∃ (total_workers : ℕ), total_workers = 12 ∧ 
    total_average * total_workers = 
      num_technicians * tech_average + (total_workers - num_technicians) * non_tech_average :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l3837_383729


namespace NUMINAMATH_CALUDE_equation_solution_l3837_383707

theorem equation_solution :
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 55 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3837_383707


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3837_383724

theorem polynomial_evaluation : 
  let a : ℤ := 2999
  let b : ℤ := 3000
  b^3 - a*b^2 - a^2*b + a^3 = b + a := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3837_383724


namespace NUMINAMATH_CALUDE_BC_vector_l3837_383728

def complex_vector (a b : ℂ) : ℂ := b - a

theorem BC_vector (OA OC AB : ℂ) 
  (h1 : OA = -2 + I) 
  (h2 : OC = 3 + 2*I) 
  (h3 : AB = 1 + 5*I) : 
  complex_vector (OA + AB) OC = 4 - 4*I := by
  sorry

end NUMINAMATH_CALUDE_BC_vector_l3837_383728


namespace NUMINAMATH_CALUDE_expression_value_l3837_383731

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3837_383731


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3837_383794

theorem equilateral_triangle_side_length 
  (total_wire_length : ℝ) 
  (h1 : total_wire_length = 63) : 
  total_wire_length / 3 = 21 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3837_383794


namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l3837_383756

theorem abc_sum_mod_five (a b c : ℕ) : 
  0 < a ∧ a < 5 ∧
  0 < b ∧ b < 5 ∧
  0 < c ∧ c < 5 ∧
  (a * b * c) % 5 = 1 ∧
  (4 * c) % 5 = 3 ∧
  (3 * b) % 5 = (2 + b) % 5 →
  (a + b + c) % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l3837_383756


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l3837_383778

theorem greatest_multiple_of_four (x : ℕ) : x > 0 ∧ 4 ∣ x ∧ x^3 < 4096 → x ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l3837_383778


namespace NUMINAMATH_CALUDE_earnings_difference_l3837_383769

/-- Given Paul's and Vinnie's earnings, prove the difference between them. -/
theorem earnings_difference (paul_earnings vinnie_earnings : ℕ) 
  (h1 : paul_earnings = 14)
  (h2 : vinnie_earnings = 30) : 
  vinnie_earnings - paul_earnings = 16 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l3837_383769


namespace NUMINAMATH_CALUDE_x_satisfies_equation_x_is_approximately_69_28_l3837_383758

/-- The number that satisfies the given equation -/
def x : ℝ := 69.28

/-- The given approximation of q -/
def q_approx : ℝ := 9.237333333333334

/-- Theorem stating that x satisfies the equation within a small margin of error -/
theorem x_satisfies_equation : 
  abs ((x * 0.004) / 0.03 - q_approx) < 0.000001 := by
  sorry

/-- Theorem stating that x is approximately equal to 69.28 -/
theorem x_is_approximately_69_28 : 
  abs (x - 69.28) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_x_satisfies_equation_x_is_approximately_69_28_l3837_383758


namespace NUMINAMATH_CALUDE_min_value_fraction_l3837_383701

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y - 1 = 0) :
  (x + 2*y) / (x*y) ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + y₀ - 1 = 0 ∧ (x₀ + 2*y₀) / (x₀*y₀) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3837_383701


namespace NUMINAMATH_CALUDE_janes_breakfast_problem_l3837_383737

/-- Represents the number of breakfast items bought -/
structure BreakfastItems where
  muffins : ℕ
  bagels : ℕ
  croissants : ℕ

/-- Calculates the total cost in cents -/
def totalCost (items : BreakfastItems) : ℕ :=
  50 * items.muffins + 75 * items.bagels + 65 * items.croissants

theorem janes_breakfast_problem :
  ∃ (items : BreakfastItems),
    items.muffins + items.bagels + items.croissants = 6 ∧
    items.bagels = 2 ∧
    (totalCost items) % 100 = 0 ∧
    items.muffins = 4 :=
  sorry

end NUMINAMATH_CALUDE_janes_breakfast_problem_l3837_383737


namespace NUMINAMATH_CALUDE_checkerboard_fraction_sum_l3837_383741

/-- The number of squares in a n×n grid -/
def squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles in a (n+1)×(n+1) grid -/
def rectangles (n : ℕ) : ℕ := (n * (n + 1) / 2)^2

theorem checkerboard_fraction_sum :
  let s := squares 7
  let r := rectangles 7
  let g := Nat.gcd s r
  (s / g) + (r / g) = 33 := by sorry

end NUMINAMATH_CALUDE_checkerboard_fraction_sum_l3837_383741


namespace NUMINAMATH_CALUDE_recurrence_relation_solution_l3837_383752

def a (n : ℕ) : ℤ := -4 + 17 * n - 21 * n^2 + 5 * n^3 + n^4

theorem recurrence_relation_solution :
  (∀ n : ℕ, n ≥ 3 → a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) + 24 * n - 6) ∧
  a 0 = -4 ∧
  a 1 = -2 ∧
  a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_relation_solution_l3837_383752


namespace NUMINAMATH_CALUDE_composite_sum_of_power_l3837_383720

theorem composite_sum_of_power (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_power_l3837_383720


namespace NUMINAMATH_CALUDE_initial_carpet_amount_l3837_383768

/-- Given a rectangular room and additional carpet needed, calculate the initial amount of carpet --/
theorem initial_carpet_amount 
  (length width : ℝ) 
  (additional_needed : ℝ) 
  (h1 : length = 4)
  (h2 : width = 20)
  (h3 : additional_needed = 62) :
  length * width - additional_needed = 18 := by
  sorry

#check initial_carpet_amount

end NUMINAMATH_CALUDE_initial_carpet_amount_l3837_383768


namespace NUMINAMATH_CALUDE_fraction_equality_l3837_383774

theorem fraction_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3837_383774


namespace NUMINAMATH_CALUDE_coal_extraction_theorem_l3837_383740

/-- Represents the working time ratio and coal extraction for a year -/
structure YearData where
  ratio : Fin 4 → ℚ
  coal_extracted : ℚ

/-- Given the data for three years, calculate the total coal extraction for 4 months -/
def total_coal_extraction (year1 year2 year3 : YearData) : ℚ :=
  4 * (year1.coal_extracted * (year1.ratio 0 + year1.ratio 1 + year1.ratio 2 + year1.ratio 3) / 
      (year1.ratio 0 + year1.ratio 1 + year1.ratio 2 + year1.ratio 3) +
      year2.coal_extracted * (year2.ratio 0 + year2.ratio 1 + year2.ratio 2 + year2.ratio 3) / 
      (year2.ratio 0 + year2.ratio 1 + year2.ratio 2 + year2.ratio 3) +
      year3.coal_extracted * (year3.ratio 0 + year3.ratio 1 + year3.ratio 2 + year3.ratio 3) / 
      (year3.ratio 0 + year3.ratio 1 + year3.ratio 2 + year3.ratio 3)) / 3

theorem coal_extraction_theorem (year1 year2 year3 : YearData) 
  (h1 : year1.ratio 0 = 4 ∧ year1.ratio 1 = 1 ∧ year1.ratio 2 = 2 ∧ year1.ratio 3 = 5 ∧ year1.coal_extracted = 10)
  (h2 : year2.ratio 0 = 2 ∧ year2.ratio 1 = 3 ∧ year2.ratio 2 = 2 ∧ year2.ratio 3 = 1 ∧ year2.coal_extracted = 7)
  (h3 : year3.ratio 0 = 5 ∧ year3.ratio 1 = 2 ∧ year3.ratio 2 = 1 ∧ year3.ratio 3 = 4 ∧ year3.coal_extracted = 14) :
  total_coal_extraction year1 year2 year3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coal_extraction_theorem_l3837_383740


namespace NUMINAMATH_CALUDE_charles_city_population_l3837_383787

theorem charles_city_population (C G : ℕ) : 
  G + 119666 = C → 
  C + G = 845640 → 
  C = 482653 := by
sorry

end NUMINAMATH_CALUDE_charles_city_population_l3837_383787


namespace NUMINAMATH_CALUDE_equivalent_operation_l3837_383760

theorem equivalent_operation (x : ℚ) : x * (4/5) / (4/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l3837_383760


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l3837_383780

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


end NUMINAMATH_CALUDE_distance_between_circle_centers_l3837_383780


namespace NUMINAMATH_CALUDE_min_value_theorem_l3837_383700

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 4) :
  (∀ u v : ℝ, u > 0 → v > 0 → Real.log 2 * u + Real.log 8 * v = Real.log 4 → 
    1/x + 1/(3*y) ≤ 1/u + 1/(3*v)) ∧ 
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 4 ∧ 
    1/x₀ + 1/(3*y₀) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3837_383700


namespace NUMINAMATH_CALUDE_no_8002_integers_divisibility_property_l3837_383761

theorem no_8002_integers_divisibility_property (P : ℕ → ℕ) 
  (h_P : ∀ x : ℕ, P x = x^2000 - x^1000 + 1) : 
  ¬ ∃ (a : Fin 8002 → ℕ), 
    (∀ i j : Fin 8002, i ≠ j → a i ≠ a j) ∧ 
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → 
      (a i * a j * a k) ∣ (P (a i) * P (a j) * P (a k))) :=
sorry

end NUMINAMATH_CALUDE_no_8002_integers_divisibility_property_l3837_383761


namespace NUMINAMATH_CALUDE_standard_deviation_measures_stability_l3837_383785

/-- A measure of stability for a set of numbers -/
def stability_measure (data : List ℝ) : ℝ := sorry

/-- Standard deviation of a list of real numbers -/
def standard_deviation (data : List ℝ) : ℝ := sorry

/-- Theorem stating that the standard deviation is a valid measure of stability for crop yields -/
theorem standard_deviation_measures_stability 
  (n : ℕ) 
  (yields : List ℝ) 
  (h1 : yields.length = n) 
  (h2 : n > 0) :
  stability_measure yields = standard_deviation yields := by sorry

end NUMINAMATH_CALUDE_standard_deviation_measures_stability_l3837_383785


namespace NUMINAMATH_CALUDE_consecutive_integer_averages_l3837_383733

theorem consecutive_integer_averages (a b : ℤ) : 
  (a > 0) →
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5) →
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4)) / 5 = a + 4) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_averages_l3837_383733


namespace NUMINAMATH_CALUDE_journey_distance_total_distance_value_l3837_383793

/-- Represents the total distance travelled by a family in a journey -/
def total_distance : ℝ := sorry

/-- The total travel time in hours -/
def total_time : ℝ := 18

/-- The speed for the first third of the journey in km/h -/
def speed1 : ℝ := 35

/-- The speed for the second third of the journey in km/h -/
def speed2 : ℝ := 40

/-- The speed for the last third of the journey in km/h -/
def speed3 : ℝ := 45

/-- Theorem stating the relationship between distance, time, and speeds -/
theorem journey_distance : 
  (total_distance / 3) / speed1 + 
  (total_distance / 3) / speed2 + 
  (total_distance / 3) / speed3 = total_time :=
by sorry

/-- Theorem stating that the total distance is approximately 712.46 km -/
theorem total_distance_value : 
  ∃ ε > 0, abs (total_distance - 712.46) < ε :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_total_distance_value_l3837_383793


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3837_383777

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side_length : ℝ
  /-- The ratio of EJ to JF -/
  ej_jf_ratio : ℝ
  /-- side_length is positive -/
  side_length_pos : 0 < side_length
  /-- ej_jf_ratio is greater than 1 -/
  ej_jf_ratio_gt_one : 1 < ej_jf_ratio

/-- The theorem stating the length of the base of the isosceles triangle -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle)
    (h1 : t.side_length = 6)
    (h2 : t.ej_jf_ratio = 2) :
  ∃ (base_length : ℝ), base_length = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3837_383777


namespace NUMINAMATH_CALUDE_jamie_yellow_balls_l3837_383709

theorem jamie_yellow_balls (initial_red : ℕ) (total_after : ℕ) : 
  initial_red = 16 →
  total_after = 74 →
  (initial_red - 6) + (2 * initial_red) + (total_after - ((initial_red - 6) + (2 * initial_red))) = total_after :=
by
  sorry

end NUMINAMATH_CALUDE_jamie_yellow_balls_l3837_383709


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l3837_383779

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- The distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

theorem car_speed_time_relation (m n : Car) 
  (h1 : n.speed = 2 * m.speed) 
  (h2 : distance n = distance m) : 
  n.time = m.time / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_time_relation_l3837_383779


namespace NUMINAMATH_CALUDE_equal_distance_at_time_l3837_383763

/-- The time in minutes past 3 o'clock when the minute hand is at the same distance 
    to the left of 12 as the hour hand is to the right of 12 -/
def time_equal_distance : ℚ := 13 + 11/13

theorem equal_distance_at_time (t : ℚ) : 
  t = time_equal_distance →
  (180 - 6 * t = 90 + 0.5 * t) := by sorry


end NUMINAMATH_CALUDE_equal_distance_at_time_l3837_383763


namespace NUMINAMATH_CALUDE_sodium_bicarbonate_required_l3837_383714

-- Define the chemical reaction
structure Reaction where
  NaHCO₃ : ℕ
  HCl : ℕ
  NaCl : ℕ
  H₂O : ℕ
  CO₂ : ℕ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.NaHCO₃ = r.HCl ∧ r.NaHCO₃ = r.NaCl ∧ r.NaHCO₃ = r.H₂O ∧ r.NaHCO₃ = r.CO₂

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.HCl = 3 ∧ r.H₂O = 3 ∧ r.CO₂ = 3 ∧ r.NaCl = 3

-- Theorem to prove
theorem sodium_bicarbonate_required (r : Reaction) 
  (h1 : balanced_equation r) (h2 : given_conditions r) : 
  r.NaHCO₃ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sodium_bicarbonate_required_l3837_383714


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l3837_383726

theorem complex_sum_of_parts (z : ℂ) (h : z * Complex.I = 1 + Complex.I) :
  z.re + z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l3837_383726


namespace NUMINAMATH_CALUDE_skier_race_l3837_383771

/-- Two skiers race with given conditions -/
theorem skier_race (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (9 / y + 9 = 9 / x) ∧ (29 / y + 9 = 25 / x) → y = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_skier_race_l3837_383771


namespace NUMINAMATH_CALUDE_cost_of_candies_l3837_383716

def candies_per_box : ℕ := 30
def cost_per_box : ℕ := 8
def total_candies : ℕ := 450

theorem cost_of_candies :
  (total_candies / candies_per_box) * cost_per_box = 120 := by
sorry

end NUMINAMATH_CALUDE_cost_of_candies_l3837_383716


namespace NUMINAMATH_CALUDE_intersection_problem_l3837_383784

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_problem (a : ℝ) :
  (a = 1/2 → A a ∩ B = {x | 0 < x ∧ x < 1}) ∧
  (A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_problem_l3837_383784


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3837_383708

/-- Given a geometric sequence {aₙ} with sum of first n terms Sₙ, 
    if S₆/S₃ = -19/8 and a₄ - a₂ = -15/8, then a₃ = 9/4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℚ)  -- The geometric sequence
  (S : ℕ → ℚ)  -- The sum function
  (h1 : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))  -- Definition of sum for geometric sequence
  (h2 : S 6 / S 3 = -19/8)  -- Given condition
  (h3 : a 4 - a 2 = -15/8)  -- Given condition
  : a 3 = 9/4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3837_383708


namespace NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l3837_383797

theorem two_fifths_of_n_is_80 (n : ℚ) : n = 5 / 6 * 240 → 2 / 5 * n = 80 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l3837_383797


namespace NUMINAMATH_CALUDE_pyramid_volume_l3837_383789

/-- 
Given a pyramid with a rhombic base:
- d₁ and d₂ are the diagonals of the rhombus base
- d₁ > d₂
- The height of the pyramid passes through the vertex of the acute angle of the rhombus
- Q is the area of the diagonal section conducted through the shorter diagonal

This theorem states that the volume of such a pyramid is (d₁ / 12) * √(16Q² - d₁²d₂²)
-/
theorem pyramid_volume (d₁ d₂ Q : ℝ) (h₁ : d₁ > d₂) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : Q > 0) :
  let volume := d₁ / 12 * Real.sqrt (16 * Q^2 - d₁^2 * d₂^2)
  volume > 0 ∧ volume^3 = (d₁^3 / 1728) * (16 * Q^2 - d₁^2 * d₂^2) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3837_383789


namespace NUMINAMATH_CALUDE_initial_persimmons_l3837_383799

/-- The number of persimmons eaten -/
def eaten : ℕ := 5

/-- The number of persimmons left -/
def left : ℕ := 12

/-- The initial number of persimmons -/
def initial : ℕ := eaten + left

theorem initial_persimmons : initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_initial_persimmons_l3837_383799


namespace NUMINAMATH_CALUDE_layla_goals_l3837_383764

theorem layla_goals (layla kristin : ℕ) (h1 : kristin = layla - 24) (h2 : layla + kristin = 368) : layla = 196 := by
  sorry

end NUMINAMATH_CALUDE_layla_goals_l3837_383764


namespace NUMINAMATH_CALUDE_recycling_points_theorem_l3837_383727

/-- Calculates the points earned for recycling paper -/
def points_earned (pounds_per_point : ℕ) (chloe_pounds : ℕ) (friends_pounds : ℕ) : ℕ :=
  (chloe_pounds + friends_pounds) / pounds_per_point

/-- Theorem: Given the conditions, the total points earned is 5 -/
theorem recycling_points_theorem : 
  points_earned 6 28 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_theorem_l3837_383727
