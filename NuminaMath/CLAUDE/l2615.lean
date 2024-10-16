import Mathlib

namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l2615_261517

theorem gasoline_tank_capacity :
  ∀ x : ℚ,
  (5/6 : ℚ) * x - (2/3 : ℚ) * x = 15 →
  x = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l2615_261517


namespace NUMINAMATH_CALUDE_y_relationship_l2615_261559

/-- A quadratic function of the form y = -(x+2)² + h -/
def f (x h : ℝ) : ℝ := -(x + 2)^2 + h

/-- The y-coordinate of point A -/
def y₁ (h : ℝ) : ℝ := f (-3) h

/-- The y-coordinate of point B -/
def y₂ (h : ℝ) : ℝ := f 2 h

/-- The y-coordinate of point C -/
def y₃ (h : ℝ) : ℝ := f 3 h

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem y_relationship (h : ℝ) : y₃ h < y₂ h ∧ y₂ h < y₁ h := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l2615_261559


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2615_261556

theorem purely_imaginary_complex_number (m : ℝ) :
  (((m * (m + 2)) / (m - 1) : ℂ) + (m^2 + m - 2) * I).re = 0 ∧
  (((m * (m + 2)) / (m - 1) : ℂ) + (m^2 + m - 2) * I).im ≠ 0 →
  m = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2615_261556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_inequality_l2615_261562

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d
def S (a₁ d : ℚ) (n : ℕ) : ℚ := n * a₁ + n * (n - 1 : ℚ) / 2 * d

-- State the theorem
theorem arithmetic_sequence_sum_inequality 
  (p q : ℕ) (a₁ d : ℚ) (hp : p ≠ q) (hSp : S a₁ d p = p / q) (hSq : S a₁ d q = q / p) :
  S a₁ d (p + q) > 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_inequality_l2615_261562


namespace NUMINAMATH_CALUDE_adult_meal_cost_l2615_261561

/-- Proves that the cost of each adult meal is $6 given the conditions of the restaurant bill. -/
theorem adult_meal_cost (num_adults num_children : ℕ) (child_meal_cost soda_cost total_bill : ℚ) :
  num_adults = 6 →
  num_children = 2 →
  child_meal_cost = 4 →
  soda_cost = 2 →
  total_bill = 60 →
  ∃ (adult_meal_cost : ℚ),
    adult_meal_cost * num_adults + child_meal_cost * num_children + soda_cost * (num_adults + num_children) = total_bill ∧
    adult_meal_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l2615_261561


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_right_triangle_geometric_progression_l2615_261540

-- Define a right triangle with sides a, b, c
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

-- Define arithmetic progression for three numbers
def is_arithmetic_progression (x y z : ℝ) : Prop :=
  y - x = z - y

-- Define geometric progression for three numbers
def is_geometric_progression (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

theorem right_triangle_arithmetic_progression :
  ∃ t : RightTriangle, is_arithmetic_progression t.a t.b t.c ∧ t.a = 3 ∧ t.b = 4 ∧ t.c = 5 :=
sorry

theorem right_triangle_geometric_progression :
  ∃ t : RightTriangle, is_geometric_progression t.a t.b t.c ∧
    t.a = 1 ∧ t.b = Real.sqrt ((1 + Real.sqrt 5) / 2) ∧ t.c = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_right_triangle_geometric_progression_l2615_261540


namespace NUMINAMATH_CALUDE_max_value_at_zero_l2615_261599

/-- The function f(x) = x³ - 3x² + 1 reaches its maximum value at x = 0 -/
theorem max_value_at_zero (f : ℝ → ℝ) (h : f = λ x => x^3 - 3*x^2 + 1) :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≤ f x₀ ∧ x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_at_zero_l2615_261599


namespace NUMINAMATH_CALUDE_company_workforce_after_hiring_l2615_261582

theorem company_workforce_after_hiring 
  (initial_female_percentage : Real)
  (additional_male_workers : Nat)
  (new_female_percentage : Real) :
  initial_female_percentage = 0.60 →
  additional_male_workers = 22 →
  new_female_percentage = 0.55 →
  (initial_female_percentage * (264 - additional_male_workers)) / 264 = new_female_percentage :=
by sorry

end NUMINAMATH_CALUDE_company_workforce_after_hiring_l2615_261582


namespace NUMINAMATH_CALUDE_unique_k_term_l2615_261503

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n^2 - 7*n

/-- The k-th term of the sequence -/
def a (k : ℕ) : ℤ := S k - S (k-1)

theorem unique_k_term (k : ℕ) (h : 9 < a k ∧ a k < 12) : k = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_term_l2615_261503


namespace NUMINAMATH_CALUDE_three_number_problem_l2615_261530

theorem three_number_problem (a b c : ℝ) 
  (sum_30 : a + b + c = 30)
  (first_twice_sum : a = 2 * (b + c))
  (second_five_third : b = 5 * c)
  (sum_first_third : a + c = 22) :
  a * b * c = 2500 / 9 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l2615_261530


namespace NUMINAMATH_CALUDE_boar_sausages_problem_l2615_261566

theorem boar_sausages_problem (S : ℕ) : 
  (S > 0) →  -- Ensure S is positive
  (3 / 40 : ℚ) * S = 45 → 
  S = 600 := by 
sorry

end NUMINAMATH_CALUDE_boar_sausages_problem_l2615_261566


namespace NUMINAMATH_CALUDE_function_transformation_l2615_261533

/-- Given a function f where f(2) = 0, prove that g(x) = f(x-3)+1 passes through (5, 1) -/
theorem function_transformation (f : ℝ → ℝ) (h : f 2 = 0) :
  let g := λ x => f (x - 3) + 1
  g 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l2615_261533


namespace NUMINAMATH_CALUDE_boat_savings_l2615_261596

/-- The cost of traveling by plane in dollars -/
def plane_cost : ℚ := 600

/-- The cost of traveling by boat in dollars -/
def boat_cost : ℚ := 254

/-- The amount saved by taking a boat instead of a plane -/
def money_saved : ℚ := plane_cost - boat_cost

theorem boat_savings : money_saved = 346 := by
  sorry

end NUMINAMATH_CALUDE_boat_savings_l2615_261596


namespace NUMINAMATH_CALUDE_expression_evaluation_l2615_261545

theorem expression_evaluation :
  let a : ℚ := -1/2
  (3*a + 2) * (a - 1) - 4*a * (a + 1) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2615_261545


namespace NUMINAMATH_CALUDE_faiths_weekly_earnings_l2615_261560

/-- Faith's weekly earnings calculation --/
theorem faiths_weekly_earnings
  (hourly_rate : ℝ)
  (regular_hours_per_day : ℕ)
  (working_days_per_week : ℕ)
  (overtime_hours_per_day : ℕ)
  (h1 : hourly_rate = 13.5)
  (h2 : regular_hours_per_day = 8)
  (h3 : working_days_per_week = 5)
  (h4 : overtime_hours_per_day = 2) :
  let regular_pay := hourly_rate * regular_hours_per_day * working_days_per_week
  let overtime_pay := hourly_rate * overtime_hours_per_day * working_days_per_week
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 675 := by sorry

end NUMINAMATH_CALUDE_faiths_weekly_earnings_l2615_261560


namespace NUMINAMATH_CALUDE_triangle_focus_property_l2615_261563

/-- Given a triangle ABC with vertices corresponding to complex numbers z₁, z₂, and z₃,
    and a point F corresponding to complex number z, prove that:
    (z - z₁)(z - z₂) + (z - z₂)(z - z₃) + (z - z₃)(z - z₁) = 0 -/
theorem triangle_focus_property (z z₁ z₂ z₃ : ℂ) : 
  (z - z₁) * (z - z₂) + (z - z₂) * (z - z₃) + (z - z₃) * (z - z₁) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_focus_property_l2615_261563


namespace NUMINAMATH_CALUDE_square_diff_inequality_l2615_261552

theorem square_diff_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (a^2 + b^2) * (a - b) > (a^2 - b^2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_diff_inequality_l2615_261552


namespace NUMINAMATH_CALUDE_birds_on_fence_l2615_261520

/-- Given a number of initial birds, additional birds, and additional storks,
    calculate the total number of birds on the fence. -/
def total_birds (initial : ℕ) (additional : ℕ) (storks : ℕ) : ℕ :=
  initial + additional + storks

/-- Theorem stating that with 6 initial birds, 4 additional birds, and 8 storks,
    the total number of birds on the fence is 18. -/
theorem birds_on_fence :
  total_birds 6 4 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2615_261520


namespace NUMINAMATH_CALUDE_problem_statement_l2615_261564

/-- Given real numbers a, b, and c satisfying certain conditions, 
    prove that c^2 * (a + b) = 2008 -/
theorem problem_statement (a b c : ℝ) 
    (h1 : a^2 * (b + c) = 2008)
    (h2 : b^2 * (a + c) = 2008)
    (h3 : a ≠ b) :
  c^2 * (a + b) = 2008 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2615_261564


namespace NUMINAMATH_CALUDE_no_n_satisfies_condition_l2615_261544

def T_n (n : ℕ+) : Set ℕ+ :=
  {a | ∃ k h : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ 1 ≤ h ∧ h ≤ 10 ∧
    a = 11 * (k + h) + 10 * (n ^ k + n ^ h)}

theorem no_n_satisfies_condition :
  ∀ n : ℕ+, ∃ a b : ℕ+, a ∈ T_n n ∧ b ∈ T_n n ∧ a ≠ b ∧ a ≡ b [MOD 110] :=
sorry

end NUMINAMATH_CALUDE_no_n_satisfies_condition_l2615_261544


namespace NUMINAMATH_CALUDE_log_expression_equals_zero_l2615_261542

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_equals_zero (x : ℝ) (h : x > 10) :
  (log x) ^ (log (log (log x))) - (log (log x)) ^ (log (log x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_zero_l2615_261542


namespace NUMINAMATH_CALUDE_artist_paintings_l2615_261576

def june_paintings : ℕ := 2

def july_paintings : ℕ := 2 * june_paintings

def august_paintings : ℕ := 3 * july_paintings

def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem artist_paintings : total_paintings = 18 := by
  sorry

end NUMINAMATH_CALUDE_artist_paintings_l2615_261576


namespace NUMINAMATH_CALUDE_f_range_and_tan_A_l2615_261577

noncomputable section

def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_range_and_tan_A :
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x ∈ Set.Icc 0 3) ∧
  (∀ A B C : ℝ, 
    f C = 2 → 
    2 * Real.sin B = Real.cos (A - C) - Real.cos (A + C) → 
    Real.tan A = (3 + Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_range_and_tan_A_l2615_261577


namespace NUMINAMATH_CALUDE_coloring_book_problem_l2615_261580

theorem coloring_book_problem (book1 book2 book3 book4 colored : ℕ) 
  (h1 : book1 = 44)
  (h2 : book2 = 35)
  (h3 : book3 = 52)
  (h4 : book4 = 48)
  (h5 : colored = 37) :
  book1 + book2 + book3 + book4 - colored = 142 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l2615_261580


namespace NUMINAMATH_CALUDE_hexagon_side_length_equals_square_side_l2615_261500

/-- Represents a hexagon with side length y -/
structure Hexagon where
  y : ℝ

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with side length s -/
structure Square where
  s : ℝ

/-- Given a 12 × 12 rectangle divided into two congruent hexagons that can form a square without overlap,
    the side length of each hexagon is 12. -/
theorem hexagon_side_length_equals_square_side 
  (rect : Rectangle)
  (hex1 hex2 : Hexagon)
  (sq : Square)
  (h1 : rect.length = 12 ∧ rect.width = 12)
  (h2 : hex1 = hex2)
  (h3 : rect.length * rect.width = sq.s * sq.s)
  (h4 : hex1.y = sq.s) :
  hex1.y = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_equals_square_side_l2615_261500


namespace NUMINAMATH_CALUDE_parabola_equation_l2615_261554

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 9 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -3)

-- Define the parabola properties
structure Parabola where
  -- The coordinate axes are the axes of symmetry
  symmetry_axes : Prop
  -- The origin is the vertex
  vertex_at_origin : Prop
  -- The parabola passes through the center of the circle
  passes_through_center : Prop

-- Theorem statement
theorem parabola_equation (p : Parabola) :
  (∀ x y : ℝ, y^2 = 9*x) ∨ (∀ x y : ℝ, x^2 = -1/3*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2615_261554


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l2615_261583

theorem mean_of_combined_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 53 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l2615_261583


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l2615_261549

theorem similar_triangles_perimeter (h_small h_large : ℝ) (p_small p_large : ℝ) :
  h_small / h_large = 3 / 5 →
  p_small = 12 →
  p_small / p_large = h_small / h_large →
  p_large = 20 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l2615_261549


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_four_l2615_261589

def numbers : List Nat := [4628, 4638, 4648, 4658, 4662]

theorem product_of_digits_not_divisible_by_four : 
  ∃ n ∈ numbers, 
    ¬(n % 4 = 0) ∧ 
    ((n % 100) % 10 * ((n % 100) / 10 % 10) = 24) := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_four_l2615_261589


namespace NUMINAMATH_CALUDE_marbles_problem_l2615_261574

theorem marbles_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) 
  (h1 : angela = a)
  (h2 : brian = 3 * a)
  (h3 : caden = 6 * a)
  (h4 : daryl = 24 * a)
  (h5 : angela + brian + caden + daryl = 156) : 
  a = 78 / 17 := by
sorry

end NUMINAMATH_CALUDE_marbles_problem_l2615_261574


namespace NUMINAMATH_CALUDE_smallest_a_with_50_squares_l2615_261585

theorem smallest_a_with_50_squares : ∃ (a : ℕ), 
  (a = 4486) ∧ 
  (∀ k : ℕ, k < a → (∃ (n : ℕ), n * n > k ∧ n * n < 3 * k) → 
    (∃ (m : ℕ), m < 50)) ∧
  (∃ (l : ℕ), l = 50 ∧ 
    (∀ i : ℕ, i ≤ l → ∃ (s : ℕ), s * s > a ∧ s * s < 3 * a)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_with_50_squares_l2615_261585


namespace NUMINAMATH_CALUDE_vessel_width_proof_l2615_261541

/-- The width of a rectangular vessel's base when a cube is immersed in it -/
theorem vessel_width_proof (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) 
  (h_cube_edge : cube_edge = 16)
  (h_vessel_length : vessel_length = 20)
  (h_water_rise : water_rise = 13.653333333333334) : 
  (cube_edge ^ 3) / (vessel_length * water_rise) = 15 := by
  sorry

end NUMINAMATH_CALUDE_vessel_width_proof_l2615_261541


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l2615_261550

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


end NUMINAMATH_CALUDE_workshop_salary_problem_l2615_261550


namespace NUMINAMATH_CALUDE_even_function_range_l2615_261524

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_function_range (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_f_neg_two : f (-2) = 1) :
  {x : ℝ | f (x - 2) ≤ 1} = Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l2615_261524


namespace NUMINAMATH_CALUDE_sticker_distribution_l2615_261508

/-- The number of ways to distribute n identical objects into k groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem sticker_distribution :
  distribute 10 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2615_261508


namespace NUMINAMATH_CALUDE_equation_proof_l2615_261557

theorem equation_proof : 300 * 2 + (12 + 4) * (1 / 8) = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2615_261557


namespace NUMINAMATH_CALUDE_gibi_score_is_59_percent_l2615_261568

/-- Represents the exam scores of four students -/
structure ExamScores where
  max_score : ℕ
  jigi_percent : ℕ
  mike_percent : ℕ
  lizzy_percent : ℕ
  average_mark : ℕ

/-- Calculates Gibi's score percentage given the exam scores -/
def gibi_score_percent (scores : ExamScores) : ℕ :=
  let total_marks := 4 * scores.average_mark
  let other_scores := (scores.jigi_percent * scores.max_score / 100) +
                      (scores.mike_percent * scores.max_score / 100) +
                      (scores.lizzy_percent * scores.max_score / 100)
  let gibi_score := total_marks - other_scores
  (gibi_score * 100) / scores.max_score

/-- Theorem stating that Gibi's score percentage is 59% given the exam conditions -/
theorem gibi_score_is_59_percent (scores : ExamScores)
  (h1 : scores.max_score = 700)
  (h2 : scores.jigi_percent = 55)
  (h3 : scores.mike_percent = 99)
  (h4 : scores.lizzy_percent = 67)
  (h5 : scores.average_mark = 490) :
  gibi_score_percent scores = 59 := by
  sorry

end NUMINAMATH_CALUDE_gibi_score_is_59_percent_l2615_261568


namespace NUMINAMATH_CALUDE_kitten_weight_l2615_261516

theorem kitten_weight (kitten lighter_dog heavier_dog : ℝ) 
  (h1 : kitten + lighter_dog + heavier_dog = 36)
  (h2 : kitten + heavier_dog = 3 * lighter_dog)
  (h3 : kitten + lighter_dog = (1/2) * heavier_dog) :
  kitten = 3 := by
  sorry

end NUMINAMATH_CALUDE_kitten_weight_l2615_261516


namespace NUMINAMATH_CALUDE_number_raised_to_fourth_l2615_261567

theorem number_raised_to_fourth : ∃ x : ℝ, 121 * x^4 = 75625 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_raised_to_fourth_l2615_261567


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l2615_261518

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_ten : ℕ
  rows_with_nine : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 67 ∧
  s.rows_with_ten * 10 + s.rows_with_nine * 9 = s.total_people

theorem seating_arrangement_theorem :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_with_ten = 4 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l2615_261518


namespace NUMINAMATH_CALUDE_train_length_problem_l2615_261586

/-- The length of a train given its speed and time to cross a fixed point. -/
def trainLength (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train traveling at 30 m/s that takes 12 seconds to cross a fixed point has a length of 360 meters. -/
theorem train_length_problem : trainLength 30 12 = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l2615_261586


namespace NUMINAMATH_CALUDE_max_value_2q_minus_r_l2615_261594

theorem max_value_2q_minus_r : 
  ∃ (q r : ℕ+), 1024 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1024 = 23 * q' + r' → 2 * q - r ≥ 2 * q' - r' ∧
  2 * q - r = 76 := by
sorry

end NUMINAMATH_CALUDE_max_value_2q_minus_r_l2615_261594


namespace NUMINAMATH_CALUDE_divisible_by_2_4_5_under_300_l2615_261578

theorem divisible_by_2_4_5_under_300 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2_4_5_under_300_l2615_261578


namespace NUMINAMATH_CALUDE_bank_savings_exceed_target_l2615_261571

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

def initial_amount : ℚ := 5
def ratio : ℚ := 2
def target_amount : ℚ := 1600

theorem bank_savings_exceed_target :
  ∃ n : ℕ, 
    (geometric_sum initial_amount ratio n > target_amount) ∧ 
    (∀ m : ℕ, m < n → geometric_sum initial_amount ratio m ≤ target_amount) ∧
    n = 9 := by
  sorry

#check bank_savings_exceed_target

end NUMINAMATH_CALUDE_bank_savings_exceed_target_l2615_261571


namespace NUMINAMATH_CALUDE_income_record_l2615_261525

/-- Represents the recording of a financial transaction -/
def record (amount : ℤ) : ℤ := amount

/-- An expenditure is recorded as a negative number -/
axiom expenditure_record : record (-200) = -200

/-- Theorem: An income is recorded as a positive number -/
theorem income_record : record 60 = 60 := by sorry

end NUMINAMATH_CALUDE_income_record_l2615_261525


namespace NUMINAMATH_CALUDE_impossible_c_nine_l2615_261588

/-- An obtuse triangle with sides a, b, and c -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_obtuse : (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)
  h_triangle_inequality : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
  h_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The theorem stating that c = 9 is impossible for the given obtuse triangle -/
theorem impossible_c_nine (t : ObtuseTriangle) (h1 : t.a = 6) (h2 : t.b = 8) : t.c ≠ 9 := by
  sorry

#check impossible_c_nine

end NUMINAMATH_CALUDE_impossible_c_nine_l2615_261588


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2615_261592

/-- Given an ellipse with equation x²/16 + y²/9 = 1, the distance between its foci is 2√7. -/
theorem ellipse_foci_distance :
  ∀ (F₁ F₂ : ℝ × ℝ),
  (∀ (x y : ℝ), x^2/16 + y^2/9 = 1 → (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 4 * (4 + 3)) →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 28 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2615_261592


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l2615_261598

/-- Represents a point in 3D Cartesian coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in 3D Cartesian coordinates -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

/-- Theorem: The point symmetric to (-1, 2, 1) with respect to xOz plane is (-1, -2, 1) -/
theorem symmetric_point_xoz :
  let original := Point3D.mk (-1) 2 1
  symmetricPointXOZ original = Point3D.mk (-1) (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l2615_261598


namespace NUMINAMATH_CALUDE_library_visitors_average_l2615_261581

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 26
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

theorem library_visitors_average :
  averageVisitorsPerDay 500 140 = 188 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l2615_261581


namespace NUMINAMATH_CALUDE_stephens_bicycle_trip_l2615_261570

/-- Calculates the total distance of Stephen's bicycle trip to church -/
theorem stephens_bicycle_trip (first_speed second_speed third_speed : ℝ)
  (time_per_third : ℝ) (h1 : first_speed = 16)
  (h2 : second_speed = 12) (h3 : third_speed = 20)
  (h4 : time_per_third = 15 / 60) :
  let distance_first := first_speed * time_per_third
  let distance_second := second_speed * time_per_third
  let distance_third := third_speed * time_per_third
  let total_distance := distance_first + distance_second + distance_third
  total_distance = 12 := by
  sorry

#check stephens_bicycle_trip

end NUMINAMATH_CALUDE_stephens_bicycle_trip_l2615_261570


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2615_261507

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = 2 * a →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 2000 →
  c = 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2615_261507


namespace NUMINAMATH_CALUDE_commute_days_is_22_l2615_261593

/-- Represents the commuting options for a day -/
inductive CommuteOption
  | MorningCarEveningBike
  | MorningBikeEveningCar
  | BothCar

/-- Represents the commute data over a period of days -/
structure CommuteData where
  totalDays : ℕ
  morningCar : ℕ
  eveningBike : ℕ
  totalCarCommutes : ℕ

/-- The commute data satisfies the given conditions -/
def validCommuteData (data : CommuteData) : Prop :=
  data.morningCar = 10 ∧
  data.eveningBike = 12 ∧
  data.totalCarCommutes = 14

theorem commute_days_is_22 (data : CommuteData) (h : validCommuteData data) :
  data.totalDays = 22 := by
  sorry

#check commute_days_is_22

end NUMINAMATH_CALUDE_commute_days_is_22_l2615_261593


namespace NUMINAMATH_CALUDE_friends_distribution_unique_solution_l2615_261519

/-- The number of friends that satisfies the given conditions -/
def number_of_friends : ℕ := 20

/-- The total amount of money distributed (in rupees) -/
def total_amount : ℕ := 100

/-- Theorem stating that the number of friends satisfies the given conditions -/
theorem friends_distribution (n : ℕ) (h : n = number_of_friends) :
  (total_amount / n : ℚ) - (total_amount / (n + 5) : ℚ) = 1 := by
  sorry

/-- Theorem proving that the number of friends is unique -/
theorem unique_solution (n : ℕ) :
  (total_amount / n : ℚ) - (total_amount / (n + 5) : ℚ) = 1 → n = number_of_friends := by
  sorry

end NUMINAMATH_CALUDE_friends_distribution_unique_solution_l2615_261519


namespace NUMINAMATH_CALUDE_find_a_l2615_261543

theorem find_a (a b : ℚ) (h1 : a / 3 = b / 2) (h2 : a + b = 10) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2615_261543


namespace NUMINAMATH_CALUDE_problem_statement_l2615_261513

noncomputable section

variables (a : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := x^2 - a*x
def g (x : ℝ) : ℝ := Real.log x
def h (x : ℝ) : ℝ := f a x + g x

theorem problem_statement :
  (∀ x > 0, f a x ≥ g x) ↔ a ≤ 1 ∧
  ∃ m : ℝ, m = 3/4 - Real.log 2 ∧
    (0 < x₁ ∧ x₁ < 1/2 ∧ 
     h a x₁ - h a x₂ > m ∧
     (∀ m' : ℝ, h a x₁ - h a x₂ > m' → m' ≤ m)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2615_261513


namespace NUMINAMATH_CALUDE_problem_solution_l2615_261528

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^2 + 12
def g (x : ℝ) : ℝ := x^2 - 6

-- State the theorem
theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 12) : a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2615_261528


namespace NUMINAMATH_CALUDE_pyramid_volume_l2615_261565

/-- The volume of a pyramid with a triangular base and lateral faces forming 45° dihedral angles with the base -/
theorem pyramid_volume (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5) : 
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := S / p
  let H := r
  let V := (1/3) * S * H
  V = 6 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2615_261565


namespace NUMINAMATH_CALUDE_max_advancing_teams_for_specific_tournament_l2615_261591

/-- Represents a football tournament with specified rules --/
structure FootballTournament where
  num_teams : Nat
  min_points_to_advance : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculates the maximum number of teams that can advance in the tournament --/
def max_advancing_teams (tournament : FootballTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of advancing teams for the specific tournament --/
theorem max_advancing_teams_for_specific_tournament :
  let tournament : FootballTournament := {
    num_teams := 7,
    min_points_to_advance := 12,
    points_for_win := 3,
    points_for_draw := 1,
    points_for_loss := 0
  }
  max_advancing_teams tournament = 5 := by sorry

end NUMINAMATH_CALUDE_max_advancing_teams_for_specific_tournament_l2615_261591


namespace NUMINAMATH_CALUDE_candy_sales_l2615_261523

theorem candy_sales (x y z : ℝ) : 
  x + y + z = 100 →
  20 * x + 25 * y + 30 * z = 2570 →
  25 * y + 30 * z = 1970 →
  y = 26 := by
sorry

end NUMINAMATH_CALUDE_candy_sales_l2615_261523


namespace NUMINAMATH_CALUDE_eggs_not_eaten_is_six_l2615_261597

/-- Represents the number of eggs not eaten in a week given the following conditions:
  * Rhea buys 2 trays of eggs every week
  * Each tray has 24 eggs
  * Her son and daughter eat 2 eggs every morning
  * Rhea and her husband eat 4 eggs every night
  * There are 7 days in a week
-/
def eggs_not_eaten : ℕ :=
  let trays_per_week : ℕ := 2
  let eggs_per_tray : ℕ := 24
  let children_eggs_per_day : ℕ := 2
  let parents_eggs_per_day : ℕ := 4
  let days_per_week : ℕ := 7
  
  let total_eggs_bought := trays_per_week * eggs_per_tray
  let children_eggs_eaten := children_eggs_per_day * days_per_week
  let parents_eggs_eaten := parents_eggs_per_day * days_per_week
  let total_eggs_eaten := children_eggs_eaten + parents_eggs_eaten
  
  total_eggs_bought - total_eggs_eaten

theorem eggs_not_eaten_is_six : eggs_not_eaten = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_not_eaten_is_six_l2615_261597


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_k_value_l2615_261573

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  root_property : a 4 ^ 2 - 20 * a 4 + 99 = 0 ∧ a 5 ^ 2 - 20 * a 5 + 99 = 0
  sum_property : ∀ n : ℕ, 0 < n → S n ≤ S k

/-- The theorem stating that k equals 9 for the special arithmetic sequence -/
theorem special_arithmetic_sequence_k_value 
  (seq : SpecialArithmeticSequence) : 
  ∃ k : ℕ, k = 9 ∧ (∀ n : ℕ, 0 < n → seq.S n ≤ seq.S k) :=
sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_k_value_l2615_261573


namespace NUMINAMATH_CALUDE_train_passing_pole_l2615_261584

/-- Proves that a train of given length and speed takes a specific time to pass a pole -/
theorem train_passing_pole (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) : 
  train_length = 500 → 
  train_speed_kmh = 90 → 
  time = train_length / (train_speed_kmh * (1000 / 3600)) → 
  time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_l2615_261584


namespace NUMINAMATH_CALUDE_three_triples_l2615_261529

/-- Least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

/-- Theorem stating that there are exactly 3 ordered triples satisfying the conditions -/
theorem three_triples : count_triples = 3 := by sorry

end NUMINAMATH_CALUDE_three_triples_l2615_261529


namespace NUMINAMATH_CALUDE_sum_of_linear_equations_l2615_261521

theorem sum_of_linear_equations (x y : ℝ) 
  (h1 : 2*x - 1 = 5) 
  (h2 : 3*y + 2 = 17) : 
  2*x + 3*y = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_linear_equations_l2615_261521


namespace NUMINAMATH_CALUDE_train_length_l2615_261512

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * 1000 / 3600 →
  platform_length = 270 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 250 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2615_261512


namespace NUMINAMATH_CALUDE_passengers_in_nine_buses_l2615_261506

/-- Given that 110 passengers fit in 5 buses, prove that 198 passengers fit in 9 buses. -/
theorem passengers_in_nine_buses :
  ∀ (passengers_per_bus : ℕ),
    110 = 5 * passengers_per_bus →
    9 * passengers_per_bus = 198 := by
  sorry

end NUMINAMATH_CALUDE_passengers_in_nine_buses_l2615_261506


namespace NUMINAMATH_CALUDE_smallest_ambiguous_weight_correct_l2615_261532

/-- The smallest total weight of kittens for which the number of kittens is not uniquely determined -/
def smallest_ambiguous_weight : ℕ := 480

/-- The total weight of the two lightest kittens -/
def lightest_two_weight : ℕ := 80

/-- The total weight of the four heaviest kittens -/
def heaviest_four_weight : ℕ := 200

/-- Predicate to check if a given total weight allows for a unique determination of the number of kittens -/
def is_uniquely_determined (total_weight : ℕ) : Prop :=
  ∀ n m : ℕ, 
    (n ≠ m) → 
    (∃ (weights_n weights_m : List ℕ),
      (weights_n.length = n ∧ weights_m.length = m) ∧
      (weights_n.sum = total_weight ∧ weights_m.sum = total_weight) ∧
      (weights_n.take 2).sum = lightest_two_weight ∧
      (weights_m.take 2).sum = lightest_two_weight ∧
      (weights_n.reverse.take 4).sum = heaviest_four_weight ∧
      (weights_m.reverse.take 4).sum = heaviest_four_weight) →
    False

theorem smallest_ambiguous_weight_correct :
  (∀ w : ℕ, w < smallest_ambiguous_weight → is_uniquely_determined w) ∧
  ¬is_uniquely_determined smallest_ambiguous_weight :=
sorry

end NUMINAMATH_CALUDE_smallest_ambiguous_weight_correct_l2615_261532


namespace NUMINAMATH_CALUDE_grants_age_fraction_l2615_261510

theorem grants_age_fraction (grant_current_age hospital_current_age : ℕ) 
  (h1 : grant_current_age = 25) (h2 : hospital_current_age = 40) :
  (grant_current_age + 5 : ℚ) / (hospital_current_age + 5 : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_grants_age_fraction_l2615_261510


namespace NUMINAMATH_CALUDE_average_marks_l2615_261539

theorem average_marks (avg_five : ℝ) (sixth_mark : ℝ) : 
  avg_five = 74 → sixth_mark = 80 → 
  ((avg_five * 5 + sixth_mark) / 6 : ℝ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l2615_261539


namespace NUMINAMATH_CALUDE_range_of_a_l2615_261569

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a ↔ -3 ≤ a ∧ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2615_261569


namespace NUMINAMATH_CALUDE_orthographic_projection_properties_l2615_261509

-- Define the basic structure for a view in orthographic projection
structure View where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the orthographic projection
structure OrthographicProjection where
  main_view : View
  top_view : View
  left_view : View

-- Define the properties of orthographic projection
def is_valid_orthographic_projection (op : OrthographicProjection) : Prop :=
  -- Main view and top view have aligned lengths
  op.main_view.length = op.top_view.length ∧
  -- Main view and left view are height level
  op.main_view.height = op.left_view.height ∧
  -- Left view and top view have equal widths
  op.left_view.width = op.top_view.width

-- Theorem statement
theorem orthographic_projection_properties (op : OrthographicProjection) 
  (h : is_valid_orthographic_projection op) :
  op.main_view.length = op.top_view.length ∧
  op.main_view.height = op.left_view.height ∧
  op.left_view.width = op.top_view.width := by
  sorry

end NUMINAMATH_CALUDE_orthographic_projection_properties_l2615_261509


namespace NUMINAMATH_CALUDE_grasshopper_can_return_to_start_l2615_261587

/-- Represents the position of the grasshopper on a 2D plane -/
structure Position where
  x : Int
  y : Int

/-- Represents a single jump of the grasshopper -/
structure Jump where
  distance : Nat
  direction : Nat  -- 0: right, 1: up, 2: left, 3: down

/-- Applies a jump to a position -/
def applyJump (pos : Position) (jump : Jump) : Position :=
  match jump.direction % 4 with
  | 0 => ⟨pos.x + jump.distance, pos.y⟩
  | 1 => ⟨pos.x, pos.y + jump.distance⟩
  | 2 => ⟨pos.x - jump.distance, pos.y⟩
  | _ => ⟨pos.x, pos.y - jump.distance⟩

/-- Generates the nth jump -/
def nthJump (n : Nat) : Jump :=
  ⟨n, n - 1⟩

/-- Theorem: The grasshopper can return to the starting point -/
theorem grasshopper_can_return_to_start :
  ∃ (jumps : List Jump), 
    let finalPos := jumps.foldl applyJump ⟨0, 0⟩
    finalPos.x = 0 ∧ finalPos.y = 0 :=
  sorry


end NUMINAMATH_CALUDE_grasshopper_can_return_to_start_l2615_261587


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2615_261531

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2615_261531


namespace NUMINAMATH_CALUDE_factors_of_expression_l2615_261547

theorem factors_of_expression : 
  ∃ (a b : ℕ), 60 < a ∧ a < b ∧ b < 70 ∧ 
  (29 * 26 * (2^48 - 1)) % a = 0 ∧ 
  (29 * 26 * (2^48 - 1)) % b = 0 ∧
  (∀ c, 60 < c ∧ c < 70 ∧ (29 * 26 * (2^48 - 1)) % c = 0 → c = a ∨ c = b) ∧
  a = 63 ∧ b = 65 :=
sorry

end NUMINAMATH_CALUDE_factors_of_expression_l2615_261547


namespace NUMINAMATH_CALUDE_star_operation_result_l2615_261590

def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

def star_operation (X Y : Set Nat) : Set Nat :=
  {x | x ∈ X ∧ x ∉ Y}

theorem star_operation_result :
  star_operation A B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l2615_261590


namespace NUMINAMATH_CALUDE_christen_peeled_twenty_l2615_261551

/-- The number of potatoes Christen peeled --/
def christenPotatoes (initialPile : ℕ) (homerRate christenRate : ℕ) (timeBeforeJoin : ℕ) : ℕ :=
  let homerPotatoes := homerRate * timeBeforeJoin
  let remainingPotatoes := initialPile - homerPotatoes
  let combinedRate := homerRate + christenRate
  let timeAfterJoin := remainingPotatoes / combinedRate
  christenRate * timeAfterJoin

/-- Theorem stating that Christen peeled 20 potatoes --/
theorem christen_peeled_twenty :
  christenPotatoes 60 4 5 6 = 20 := by
  sorry

#eval christenPotatoes 60 4 5 6

end NUMINAMATH_CALUDE_christen_peeled_twenty_l2615_261551


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2615_261502

/-- Given a line y = kx + m intersecting a parabola y^2 = 4x at two points,
    if the midpoint of these intersection points has y-coordinate 2,
    then k = 1. -/
theorem line_parabola_intersection (k m x₀ : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    -- Line equation
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    -- Parabola equation
    y₁^2 = 4 * x₁ ∧
    y₂^2 = 4 * x₂ ∧
    -- Midpoint condition
    (x₁ + x₂) / 2 = x₀ ∧
    (y₁ + y₂) / 2 = 2) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2615_261502


namespace NUMINAMATH_CALUDE_interest_rate_equation_l2615_261575

/-- Proves that the interest rate R satisfies the equation for the given conditions -/
theorem interest_rate_equation (P : ℝ) (n : ℝ) (R : ℝ) : 
  P = 10000 → n = 2 → P * ((1 + R/100)^n - (1 + n*R/100)) = 36 → R = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l2615_261575


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l2615_261527

theorem quadratic_discriminant_nonnegative (x : ℤ) :
  x^2 * (25 - 24*x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l2615_261527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2615_261526

/-- Given two arithmetic sequences and their sum properties, prove the ratio of their 7th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n / T n = (3 * n + 5 : ℚ) / (2 * n + 3)) →
  (∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2) →
  (∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2) →
  a 7 / b 7 = 44 / 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2615_261526


namespace NUMINAMATH_CALUDE_orange_bin_theorem_l2615_261505

def final_orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : ℕ :=
  initial - thrown_away + added

theorem orange_bin_theorem (initial : ℕ) (thrown_away : ℕ) (added : ℕ) 
  (h1 : thrown_away ≤ initial) :
  final_orange_count initial thrown_away added = initial - thrown_away + added :=
by
  sorry

#eval final_orange_count 31 9 38

end NUMINAMATH_CALUDE_orange_bin_theorem_l2615_261505


namespace NUMINAMATH_CALUDE_inequality_proof_l2615_261553

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 ∧ 
  x * y + y * z + z * x - 3 * x * y * z ≤ 1/4 := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l2615_261553


namespace NUMINAMATH_CALUDE_min_value_abs_sum_min_value_abs_sum_achieved_l2615_261501

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 3| ≥ 2 := by sorry

theorem min_value_abs_sum_achieved : ∃ x : ℝ, |x - 1| + |x - 3| = 2 := by sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_min_value_abs_sum_achieved_l2615_261501


namespace NUMINAMATH_CALUDE_tv_show_episode_duration_l2615_261548

/-- Prove that the duration of each episode is 0.5 hours -/
theorem tv_show_episode_duration :
  let regular_seasons : ℕ := 9
  let episodes_per_regular_season : ℕ := 22
  let episodes_in_last_season : ℕ := 26
  let total_watching_time : ℝ := 112

  let total_episodes : ℕ := regular_seasons * episodes_per_regular_season + episodes_in_last_season
  let episode_duration : ℝ := total_watching_time / total_episodes

  episode_duration = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_episode_duration_l2615_261548


namespace NUMINAMATH_CALUDE_exponent_division_l2615_261546

theorem exponent_division (a : ℝ) : a^4 / a^3 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2615_261546


namespace NUMINAMATH_CALUDE_total_animals_savanna_l2615_261555

-- Define the number of animals in Safari National Park
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10
def safari_elephants : ℕ := safari_lions / 4

-- Define the number of animals in Savanna National Park
def savanna_lions : ℕ := safari_lions * 2
def savanna_snakes : ℕ := safari_snakes * 3
def savanna_giraffes : ℕ := safari_giraffes + 20
def savanna_elephants : ℕ := safari_elephants * 5
def savanna_zebras : ℕ := (savanna_lions + savanna_snakes) / 2

-- Theorem statement
theorem total_animals_savanna : 
  savanna_lions + savanna_snakes + savanna_giraffes + savanna_elephants + savanna_zebras = 710 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_savanna_l2615_261555


namespace NUMINAMATH_CALUDE_cartesian_oval_properties_l2615_261537

-- Define the Cartesian oval
def cartesian_oval (x y : ℝ) : Prop := x^3 + y^3 - 3*x*y = 0

theorem cartesian_oval_properties :
  -- 1. The curve does not pass through the third quadrant
  (∀ x y : ℝ, cartesian_oval x y → ¬(x < 0 ∧ y < 0)) ∧
  -- 2. The curve is symmetric about the line y = x
  (∀ x y : ℝ, cartesian_oval x y ↔ cartesian_oval y x) ∧
  -- 3. The curve has no common point with the line x + y = -1
  (∀ x y : ℝ, cartesian_oval x y → x + y ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_cartesian_oval_properties_l2615_261537


namespace NUMINAMATH_CALUDE_car_travel_time_l2615_261535

theorem car_travel_time (speed_x speed_y distance_after_y : ℝ) 
  (hx : speed_x = 35)
  (hy : speed_y = 70)
  (hd : distance_after_y = 42)
  (h_same_distance : ∀ t : ℝ, speed_x * (t + (distance_after_y / speed_x)) = speed_y * t) :
  (distance_after_y / speed_x) * 60 = 72 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l2615_261535


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2615_261511

theorem polynomial_division_theorem (x : ℝ) :
  8 * x^3 - 4 * x^2 + 6 * x - 15 = (x - 3) * (8 * x^2 + 20 * x + 66) + 183 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2615_261511


namespace NUMINAMATH_CALUDE_square_side_length_l2615_261504

theorem square_side_length (s : ℝ) (h : s > 0) : s^2 = 2 * (4 * s) → s = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2615_261504


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2615_261522

/-- Represents a rectangular field with given properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ
  length_constraint : length = breadth + 30
  perimeter_constraint : perimeter = 2 * (length + breadth)

/-- Theorem: Area of the rectangular field with given constraints is 18000 square meters -/
theorem rectangular_field_area (field : RectangularField) (h : field.perimeter = 540) :
  field.length * field.breadth = 18000 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l2615_261522


namespace NUMINAMATH_CALUDE_museum_exhibit_group_size_l2615_261538

/-- Represents the ticket sales data for a museum exhibit --/
structure TicketSales where
  regular_price : ℕ
  student_price : ℕ
  total_revenue : ℕ
  regular_to_student_ratio : ℕ
  start_time : ℕ  -- in minutes since midnight
  end_time : ℕ    -- in minutes since midnight
  interval : ℕ    -- in minutes

/-- Calculates the number of people in each group for the given ticket sales data --/
def people_per_group (sales : TicketSales) : ℕ :=
  let student_tickets := sales.total_revenue / (sales.regular_price * sales.regular_to_student_ratio + sales.student_price)
  let regular_tickets := student_tickets * sales.regular_to_student_ratio
  let total_tickets := student_tickets + regular_tickets
  let num_groups := (sales.end_time - sales.start_time) / sales.interval
  total_tickets / num_groups

/-- Theorem stating that for the given conditions, the number of people in each group is 30 --/
theorem museum_exhibit_group_size :
  let sales : TicketSales := {
    regular_price := 10,
    student_price := 5,
    total_revenue := 28350,
    regular_to_student_ratio := 3,
    start_time := 9 * 60,      -- 9:00 AM in minutes
    end_time := 17 * 60 + 55,  -- 5:55 PM in minutes
    interval := 5
  }
  people_per_group sales = 30 := by
  sorry


end NUMINAMATH_CALUDE_museum_exhibit_group_size_l2615_261538


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2615_261558

def M : Set ℤ := {x | Real.log (x - 1) ≤ 0}
def N : Set ℤ := {x | Int.natAbs x < 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2615_261558


namespace NUMINAMATH_CALUDE_matrix_product_equals_C_l2615_261536

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, -1, 2; 1, 0, 5; 4, 1, -2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![2, -3, 4; -1, 5, -2; 0, 2, 7]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![7, -10, 28; 2, 7, 39; 7, -11, 0]

theorem matrix_product_equals_C : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_equals_C_l2615_261536


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l2615_261595

/-- The number of ways to distribute books to people -/
def distribute_books (total_people : ℕ) (math_books : ℕ) (chinese_books : ℕ) : ℕ :=
  Nat.choose total_people chinese_books

/-- Theorem stating that the number of ways to distribute 6 math books and 3 Chinese books
    to 9 people is equal to C(9,3) -/
theorem book_distribution_theorem :
  distribute_books 9 6 3 = Nat.choose 9 3 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l2615_261595


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2615_261572

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 5 * p - 8 = 0) → 
  (3 * q^2 + 5 * q - 8 = 0) → 
  (p - 2) * (q - 2) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2615_261572


namespace NUMINAMATH_CALUDE_vector_addition_l2615_261534

/-- Given two vectors AB and BC in ℝ², prove that AC = AB + BC -/
theorem vector_addition (AB BC : ℝ × ℝ) : 
  AB = (2, -1) → BC = (-4, 1) → AB + BC = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_vector_addition_l2615_261534


namespace NUMINAMATH_CALUDE_money_sharing_problem_l2615_261514

/-- Represents the ratio of money distribution among three people -/
structure MoneyRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents the money distribution among three people -/
structure MoneyDistribution :=
  (amanda : ℕ) (ben : ℕ) (carlos : ℕ)

/-- Theorem stating that given a money ratio of 2:3:8 and Amanda's share of $30, 
    the total amount shared is $195 -/
theorem money_sharing_problem 
  (ratio : MoneyRatio) 
  (dist : MoneyDistribution) :
  ratio.a = 2 ∧ ratio.b = 3 ∧ ratio.c = 8 ∧ 
  dist.amanda = 30 ∧
  dist.amanda * ratio.b = dist.ben * ratio.a ∧
  dist.amanda * ratio.c = dist.carlos * ratio.a →
  dist.amanda + dist.ben + dist.carlos = 195 :=
by sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l2615_261514


namespace NUMINAMATH_CALUDE_adult_meal_cost_l2615_261579

/-- Calculates the cost of an adult meal given the total number of people,
    number of kids, and total cost for a group at a restaurant where kids eat free. -/
theorem adult_meal_cost (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) :
  total_people = 9 →
  num_kids = 2 →
  total_cost = 14 →
  (total_cost / (total_people - num_kids : ℚ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l2615_261579


namespace NUMINAMATH_CALUDE_initial_men_is_50_l2615_261515

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  completedLength : ℝ
  completedDays : ℝ
  extraMen : ℕ

/-- Calculates the initial number of men for a given road project -/
def initialMen (project : RoadProject) : ℕ :=
  sorry

/-- The theorem stating that for the given project conditions, the initial number of men is 50 -/
theorem initial_men_is_50 (project : RoadProject) 
  (h1 : project.totalLength = 15)
  (h2 : project.totalDays = 300)
  (h3 : project.completedLength = 2.5)
  (h4 : project.completedDays = 100)
  (h5 : project.extraMen = 75)
  : initialMen project = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_is_50_l2615_261515
