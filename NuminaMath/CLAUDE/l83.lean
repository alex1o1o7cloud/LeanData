import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l83_8362

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 + 6*x + 2
  ∃ x1 x2 : ℝ, x1 = -3 + Real.sqrt 7 ∧ x2 = -3 - Real.sqrt 7 ∧ f x1 = 0 ∧ f x2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l83_8362


namespace NUMINAMATH_CALUDE_desk_purchase_price_l83_8379

/-- Proves that the purchase price of a desk is $100 given the specified conditions -/
theorem desk_purchase_price (purchase_price selling_price : ℝ) : 
  selling_price = purchase_price + 0.5 * selling_price →
  selling_price - purchase_price = 100 →
  purchase_price = 100 := by
sorry

end NUMINAMATH_CALUDE_desk_purchase_price_l83_8379


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l83_8340

theorem quadratic_square_of_binomial (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 4 * x^2 + 18 * x + a = (2 * x + b)^2) → a = 81 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l83_8340


namespace NUMINAMATH_CALUDE_sam_total_dimes_l83_8344

def initial_dimes : ℕ := 9
def given_dimes : ℕ := 7

theorem sam_total_dimes : initial_dimes + given_dimes = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_dimes_l83_8344


namespace NUMINAMATH_CALUDE_inequality_theorem_l83_8313

theorem inequality_theorem (p q r : ℝ) : 
  (∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x < -6 ∨ |x - 20| ≤ 2) →
  p < q →
  p + 2*q + 3*r = 44 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l83_8313


namespace NUMINAMATH_CALUDE_number_equation_l83_8382

theorem number_equation (x : ℝ) : 3 * x - 1 = 2 * x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l83_8382


namespace NUMINAMATH_CALUDE_special_heptagon_perimeter_l83_8342

/-- A heptagon with six sides of length 3 and one side of length 5 -/
structure SpecialHeptagon where
  side_length_six : ℝ
  side_length_one : ℝ
  is_heptagon : side_length_six = 3 ∧ side_length_one = 5

/-- The perimeter of a SpecialHeptagon -/
def perimeter (h : SpecialHeptagon) : ℝ :=
  6 * h.side_length_six + h.side_length_one

theorem special_heptagon_perimeter (h : SpecialHeptagon) :
  perimeter h = 23 := by
  sorry

end NUMINAMATH_CALUDE_special_heptagon_perimeter_l83_8342


namespace NUMINAMATH_CALUDE_trivia_game_points_per_question_l83_8397

/-- Given a trivia game where a player answers questions correctly and receives a total score,
    this theorem proves that if the player answers 10 questions correctly and scores 50 points,
    then each question is worth 5 points. -/
theorem trivia_game_points_per_question 
  (total_questions : ℕ) 
  (total_score : ℕ) 
  (points_per_question : ℕ) 
  (h1 : total_questions = 10) 
  (h2 : total_score = 50) : 
  points_per_question = 5 := by
  sorry

#check trivia_game_points_per_question

end NUMINAMATH_CALUDE_trivia_game_points_per_question_l83_8397


namespace NUMINAMATH_CALUDE_train_crossing_time_l83_8317

/-- Given a train and platform with specific dimensions and time to pass the platform,
    calculate the time it takes for the train to cross a point object (tree). -/
theorem train_crossing_time (train_length platform_length time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 500)
  (h3 : time_to_pass_platform = 170) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l83_8317


namespace NUMINAMATH_CALUDE_point_on_curve_l83_8337

/-- Curve C is defined by the parametric equations x = 4t² and y = t -/
def curve_C (t : ℝ) : ℝ × ℝ := (4 * t^2, t)

/-- Point P has coordinates (m, 2) -/
def point_P (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Theorem: If point P(m, 2) lies on curve C, then m = 16 -/
theorem point_on_curve (m : ℝ) : 
  (∃ t : ℝ, curve_C t = point_P m) → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l83_8337


namespace NUMINAMATH_CALUDE_geometric_series_sum_l83_8354

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 6) :
  ∑' n, 2*a / (a + b)^n = 12/7 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l83_8354


namespace NUMINAMATH_CALUDE_regular_pentagon_not_seamless_l83_8343

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

def is_divisor_of_360 (angle : ℚ) : Prop := ∃ k : ℕ, 360 = k * angle

theorem regular_pentagon_not_seamless :
  ¬(is_divisor_of_360 (interior_angle 5)) ∧
  (is_divisor_of_360 (interior_angle 3)) ∧
  (is_divisor_of_360 (interior_angle 4)) ∧
  (is_divisor_of_360 (interior_angle 6)) :=
sorry

end NUMINAMATH_CALUDE_regular_pentagon_not_seamless_l83_8343


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l83_8325

def f (x : ℝ) := -2 * (x + 3)^2 + 1

theorem quadratic_function_properties :
  let opens_downward := ∀ x y : ℝ, f ((x + y) / 2) > (f x + f y) / 2
  let axis_of_symmetry := 3
  let vertex := (3, 1)
  let decreases_after_three := ∀ x₁ x₂ : ℝ, x₁ > 3 → x₂ > x₁ → f x₂ < f x₁
  
  (opens_downward ∧ ¬(f axis_of_symmetry = f (-axis_of_symmetry)) ∧
   ¬(f (vertex.1) = vertex.2) ∧ decreases_after_three) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l83_8325


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_simplification_and_evaluation_l83_8310

-- Question 1
theorem factorization_1 (m n : ℝ) : 
  m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := by sorry

-- Question 2
theorem factorization_2 (p : ℝ) :
  (p - 3) * (p - 1) + 1 = (p - 2)^2 := by sorry

-- Question 3
theorem simplification_and_evaluation (x : ℝ) 
  (h : x^2 + x + 1/4 = 0) :
  ((2*x + 1) / (x + 1) + x - 1) / ((x + 2) / (x^2 + 2*x + 1)) = -1/4 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_simplification_and_evaluation_l83_8310


namespace NUMINAMATH_CALUDE_correct_scores_theorem_l83_8368

/-- Represents a class with exam scores -/
structure ExamClass where
  studentCount : Nat
  initialAverage : ℝ
  initialVariance : ℝ
  studentAInitialScore : ℝ
  studentAActualScore : ℝ
  studentBInitialScore : ℝ
  studentBActualScore : ℝ

/-- Calculates the new average and variance after correcting two scores -/
def correctScores (c : ExamClass) : ℝ × ℝ :=
  let newAverage := c.initialAverage
  let newVariance := c.initialVariance - 25
  (newAverage, newVariance)

theorem correct_scores_theorem (c : ExamClass) 
  (h1 : c.studentCount = 48)
  (h2 : c.initialAverage = 70)
  (h3 : c.initialVariance = 75)
  (h4 : c.studentAInitialScore = 50)
  (h5 : c.studentAActualScore = 80)
  (h6 : c.studentBInitialScore = 100)
  (h7 : c.studentBActualScore = 70) :
  correctScores c = (70, 50) := by
  sorry

end NUMINAMATH_CALUDE_correct_scores_theorem_l83_8368


namespace NUMINAMATH_CALUDE_total_dog_weight_l83_8358

/-- The weight of Evan's dog in pounds -/
def evans_dog_weight : ℝ := 63

/-- The ratio of Evan's dog weight to Ivan's dog weight -/
def weight_ratio : ℝ := 7

/-- Theorem: The total weight of Evan's and Ivan's dogs is 72 pounds -/
theorem total_dog_weight : evans_dog_weight + evans_dog_weight / weight_ratio = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_dog_weight_l83_8358


namespace NUMINAMATH_CALUDE_school_merger_ratio_l83_8384

theorem school_merger_ratio (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (8 * a) / (7 * a) = 8 / 7 →
  (30 * b) / (31 * b) = 30 / 31 →
  (8 * a + 30 * b) / (7 * a + 31 * b) = 27 / 26 →
  (8 * a + 7 * a) / (30 * b + 31 * b) = 27 / 26 :=
by sorry

end NUMINAMATH_CALUDE_school_merger_ratio_l83_8384


namespace NUMINAMATH_CALUDE_intersection_value_l83_8324

theorem intersection_value (a : ℝ) : 
  let M : Set ℝ := {a^2, a+1, -3}
  let N : Set ℝ := {a-3, 2*a-1, a^2+1}
  M ∩ N = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l83_8324


namespace NUMINAMATH_CALUDE_exists_large_remainder_sum_l83_8329

/-- Given positive integers N and a, generates a sequence of remainders by repeatedly dividing N by the last remainder, starting with a, until 0 is reached. -/
def remainderSequence (N a : ℕ+) : List ℕ :=
  sorry

/-- The theorem states that there exist positive integers N and a such that the sum of the remainder sequence is greater than 100N. -/
theorem exists_large_remainder_sum : ∃ N a : ℕ+, 
  (remainderSequence N a).sum > 100 * N.val := by
  sorry

end NUMINAMATH_CALUDE_exists_large_remainder_sum_l83_8329


namespace NUMINAMATH_CALUDE_percent_of_whole_l83_8356

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 50 ↔ part = (1/2) * whole :=
sorry

end NUMINAMATH_CALUDE_percent_of_whole_l83_8356


namespace NUMINAMATH_CALUDE_odd_function_with_property_M_even_function_with_property_M_l83_8347

def has_property_M (f : ℝ → ℝ) (A : Set ℝ) :=
  ∃ c : ℝ, ∀ x ∈ A, Real.exp x * (f x - Real.exp x) = c

def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

theorem odd_function_with_property_M (f : ℝ → ℝ) (h1 : is_odd f) (h2 : has_property_M f Set.univ) :
  ∀ x, f x = Real.exp x - 1 / Real.exp x := by sorry

theorem even_function_with_property_M (g : ℝ → ℝ) (h1 : is_even g)
    (h2 : has_property_M g (Set.Icc (-1) 1))
    (h3 : ∀ x ∈ Set.Icc (-1) 1, g (2 * x) - 2 * Real.exp 1 * g x + n > 0) :
  n > Real.exp 2 + 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_with_property_M_even_function_with_property_M_l83_8347


namespace NUMINAMATH_CALUDE_abs_reciprocal_of_neg_three_halves_l83_8391

theorem abs_reciprocal_of_neg_three_halves :
  |(((-1 : ℚ) - (1 : ℚ) / (2 : ℚ))⁻¹)| = (2 : ℚ) / (3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_abs_reciprocal_of_neg_three_halves_l83_8391


namespace NUMINAMATH_CALUDE_complex_modulus_one_l83_8323

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l83_8323


namespace NUMINAMATH_CALUDE_index_difference_proof_l83_8352

/-- Calculates the index for a subgroup within a larger group -/
def calculate_index (n k x : ℕ) : ℚ :=
  (n - k : ℚ) / n * (n - x : ℚ) / n

theorem index_difference_proof (n k x_f x_m : ℕ) 
  (h_n : n = 25)
  (h_k : k = 8)
  (h_x_f : x_f = 6)
  (h_x_m : x_m = 10) :
  calculate_index n k x_f - calculate_index n (n - k) x_m = 203 / 625 := by
  sorry

#eval calculate_index 25 8 6 - calculate_index 25 17 10

end NUMINAMATH_CALUDE_index_difference_proof_l83_8352


namespace NUMINAMATH_CALUDE_philatelist_stamps_problem_l83_8393

theorem philatelist_stamps_problem :
  ∃! x : ℕ,
    x % 2 = 1 ∧
    x % 3 = 1 ∧
    x % 5 = 3 ∧
    x % 9 = 7 ∧
    150 < x ∧
    x ≤ 300 ∧
    x = 223 := by sorry

end NUMINAMATH_CALUDE_philatelist_stamps_problem_l83_8393


namespace NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l83_8370

theorem existence_of_n_with_k_prime_factors (k m : ℕ) (hk : k > 0) (hm : m > 0) (hm_odd : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (∃ (p : Finset ℕ), p.card ≥ k ∧ ∀ q ∈ p, Prime q ∧ q ∣ (m^n + n^m)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l83_8370


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l83_8374

/-- Given two 2D vectors a and b, if a is perpendicular to (a + m*b), then m = 2/5 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (ha : a = (1, -1)) 
  (hb : b = (-2, 3)) 
  (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) : 
  m = 2/5 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l83_8374


namespace NUMINAMATH_CALUDE_possible_values_of_a_l83_8377

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | x * a - 1 = 0}

-- State the theorem
theorem possible_values_of_a :
  ∀ a : ℝ, (A ∩ B a = B a) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l83_8377


namespace NUMINAMATH_CALUDE_original_price_calculation_l83_8327

/-- Proves that if an article is sold for $130 with a 30% gain, then its original price was $100. -/
theorem original_price_calculation (sale_price : ℝ) (gain_percent : ℝ) : 
  sale_price = 130 ∧ gain_percent = 30 → 
  sale_price = (100 : ℝ) * (1 + gain_percent / 100) := by
sorry

end NUMINAMATH_CALUDE_original_price_calculation_l83_8327


namespace NUMINAMATH_CALUDE_total_nails_is_113_l83_8369

/-- The number of nails Cassie needs to cut for her pets -/
def total_nails_to_cut : ℕ :=
  let num_dogs : ℕ := 4
  let num_parrots : ℕ := 8
  let nails_per_dog_foot : ℕ := 4
  let feet_per_dog : ℕ := 4
  let claws_per_parrot_leg : ℕ := 3
  let legs_per_parrot : ℕ := 2
  let extra_nail : ℕ := 1

  let dog_nails : ℕ := num_dogs * nails_per_dog_foot * feet_per_dog
  let parrot_nails : ℕ := num_parrots * claws_per_parrot_leg * legs_per_parrot
  
  dog_nails + parrot_nails + extra_nail

/-- Theorem stating that the total number of nails Cassie needs to cut is 113 -/
theorem total_nails_is_113 : total_nails_to_cut = 113 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_is_113_l83_8369


namespace NUMINAMATH_CALUDE_five_lines_max_sections_l83_8388

/-- The maximum number of sections created by drawing n line segments through a rectangle,
    given that the first line segment separates the rectangle into 2 sections. -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 2 + (n - 1) * n / 2

/-- Theorem: The maximum number of sections created by drawing 5 line segments
    through a rectangle is 16, given that the first line segment separates
    the rectangle into 2 sections. -/
theorem five_lines_max_sections :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_max_sections_l83_8388


namespace NUMINAMATH_CALUDE_cakes_distribution_l83_8311

theorem cakes_distribution (total_cakes : ℕ) (friends : ℕ) (cakes_per_friend : ℕ) :
  total_cakes = 30 →
  friends = 2 →
  cakes_per_friend = total_cakes / friends →
  cakes_per_friend = 15 := by
sorry

end NUMINAMATH_CALUDE_cakes_distribution_l83_8311


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l83_8367

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l83_8367


namespace NUMINAMATH_CALUDE_range_of_a_l83_8359

def p (a : ℝ) : Prop := ∀ x : ℝ, (a - 3/2) ^ x > 0 ∧ (a - 3/2) ^ x < 1

def q (a : ℝ) : Prop := ∃ f : ℝ → ℝ, (∀ x ∈ [0, a], f x = x^2 - 4*x + 3) ∧
  (∀ y ∈ Set.range f, -1 ≤ y ∧ y ≤ 3)

theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) →
  ((3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l83_8359


namespace NUMINAMATH_CALUDE_winter_ball_attendance_l83_8385

theorem winter_ball_attendance 
  (total_students : ℕ) 
  (ball_attendees : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) : 
  total_students = 1500 →
  ball_attendees = 900 →
  girls + boys = total_students →
  (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = ball_attendees →
  3 * girls / 4 = 900 :=
by sorry

end NUMINAMATH_CALUDE_winter_ball_attendance_l83_8385


namespace NUMINAMATH_CALUDE_division_problem_l83_8309

theorem division_problem (a b q : ℕ) (h1 : a - b = 1360) (h2 : a = 1614) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l83_8309


namespace NUMINAMATH_CALUDE_rational_equation_solution_l83_8301

theorem rational_equation_solution (x : ℝ) : 
  (x^2 - 7*x + 10) / (x^2 - 9*x + 8) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) ↔ x = 11 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l83_8301


namespace NUMINAMATH_CALUDE_min_b_over_a_l83_8304

theorem min_b_over_a (a b : ℝ) (h : ∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + b) : 
  (∀ c : ℝ, (∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + c) → b / a ≤ c / a) → b / a = 1 - Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_b_over_a_l83_8304


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l83_8339

theorem initial_markup_percentage (initial_price : ℝ) (price_increase : ℝ) : 
  initial_price = 36 →
  price_increase = 4 →
  (initial_price + price_increase) / (initial_price - initial_price * 0.8) = 2 →
  (initial_price - (initial_price - initial_price * 0.8)) / (initial_price - initial_price * 0.8) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l83_8339


namespace NUMINAMATH_CALUDE_only_vertical_angles_always_equal_l83_8373

-- Define the types for lines and angles
def Line : Type := ℝ → ℝ → Prop
def Angle : Type := ℝ

-- Define the relationships between angles
def are_alternate_interior (a b : Angle) (l1 l2 : Line) : Prop := sorry
def are_consecutive_interior (a b : Angle) (l1 l2 : Line) : Prop := sorry
def are_vertical (a b : Angle) : Prop := sorry
def are_adjacent_supplementary (a b : Angle) : Prop := sorry
def are_corresponding (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the property of being supplementary
def are_supplementary (a b : Angle) : Prop := sorry

-- Theorem stating that only vertical angles are always equal
theorem only_vertical_angles_always_equal :
  ∀ (a b : Angle) (l1 l2 : Line),
    (are_vertical a b → a = b) ∧
    (are_alternate_interior a b l1 l2 → ¬(l1 = l2) → ¬(a = b)) ∧
    (are_consecutive_interior a b l1 l2 → ¬(l1 = l2) → ¬(are_supplementary a b)) ∧
    (are_adjacent_supplementary a b → ¬(a = b)) ∧
    (are_corresponding a b l1 l2 → ¬(l1 = l2) → ¬(a = b)) :=
by
  sorry

end NUMINAMATH_CALUDE_only_vertical_angles_always_equal_l83_8373


namespace NUMINAMATH_CALUDE_theater_capacity_filled_l83_8307

theorem theater_capacity_filled (seats : ℕ) (ticket_price : ℕ) (performances : ℕ) (total_revenue : ℕ) :
  seats = 400 →
  ticket_price = 30 →
  performances = 3 →
  total_revenue = 28800 →
  (total_revenue / ticket_price) / (seats * performances) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_theater_capacity_filled_l83_8307


namespace NUMINAMATH_CALUDE_circle_center_l83_8355

/-- A circle passing through (0,0) and tangent to y = x^2 at (1,1) has center (-1, 2) -/
theorem circle_center (c : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = (c.1)^2 + (c.2)^2 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (∃ (r : ℝ), ∀ (x y : ℝ), y = x^2 → ((x - 1)^2 + (y - 1)^2 = r^2 → x = 1 ∧ y = 1)) →
  c = (-1, 2) := by
sorry


end NUMINAMATH_CALUDE_circle_center_l83_8355


namespace NUMINAMATH_CALUDE_expression_value_l83_8315

theorem expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1)*(m - 1) + 2022 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l83_8315


namespace NUMINAMATH_CALUDE_carol_achieves_target_average_l83_8392

-- Define the inverse relationship between exercise time and test score
def inverse_relation (exercise_time : ℝ) (test_score : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ exercise_time * test_score = k

-- Define Carol's first test results
def first_test_exercise_time : ℝ := 45
def first_test_score : ℝ := 80

-- Define Carol's target average score
def target_average_score : ℝ := 85

-- Define Carol's exercise time for the second test
def second_test_exercise_time : ℝ := 40

-- Theorem to prove
theorem carol_achieves_target_average :
  inverse_relation first_test_exercise_time first_test_score →
  inverse_relation second_test_exercise_time ((2 * target_average_score * 2) - first_test_score) →
  (first_test_score + ((2 * target_average_score * 2) - first_test_score)) / 2 = target_average_score :=
by
  sorry

end NUMINAMATH_CALUDE_carol_achieves_target_average_l83_8392


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_l83_8346

theorem cos_alpha_minus_pi (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.cos (α - π) = 2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_l83_8346


namespace NUMINAMATH_CALUDE_board_numbers_l83_8366

theorem board_numbers (N : ℕ) (numbers : Finset ℝ) : 
  (N ≥ 9) →
  (Finset.card numbers = N) →
  (∀ x ∈ numbers, 0 ≤ x ∧ x < 1) →
  (∀ subset : Finset ℝ, subset ⊆ numbers → Finset.card subset = 8 → 
    ∃ y ∈ numbers, y ∉ subset ∧ 
    ∃ z : ℤ, (Finset.sum subset (λ i => i) + y = z)) →
  N = 9 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_l83_8366


namespace NUMINAMATH_CALUDE_elephant_ratio_is_three_l83_8351

/-- The number of elephants at We Preserve For Future park -/
def we_preserve_elephants : ℕ := 70

/-- The total number of elephants in both parks -/
def total_elephants : ℕ := 280

/-- The ratio of elephants at Gestures For Good park to We Preserve For Future park -/
def elephant_ratio : ℚ := (total_elephants - we_preserve_elephants) / we_preserve_elephants

theorem elephant_ratio_is_three : elephant_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_elephant_ratio_is_three_l83_8351


namespace NUMINAMATH_CALUDE_grass_sheet_cost_per_cubic_meter_l83_8333

/-- The cost of a grass sheet per cubic meter, given the area of a playground,
    the depth of the grass sheet, and the total cost to cover the playground. -/
theorem grass_sheet_cost_per_cubic_meter
  (area : ℝ) (depth_cm : ℝ) (total_cost : ℝ)
  (h_area : area = 5900)
  (h_depth : depth_cm = 1)
  (h_total_cost : total_cost = 165.2) :
  total_cost / (area * depth_cm / 100) = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_grass_sheet_cost_per_cubic_meter_l83_8333


namespace NUMINAMATH_CALUDE_frogs_in_pond_a_l83_8305

theorem frogs_in_pond_a (frogs_b : ℕ) : 
  frogs_b + 2 * frogs_b = 48 → 2 * frogs_b = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_frogs_in_pond_a_l83_8305


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l83_8328

/-- Given an integer N represented as 777 in base b, 
    prove that 18 is the smallest positive integer b 
    such that N is the fourth power of a decimal integer -/
theorem smallest_base_for_fourth_power (N : ℤ) (b : ℕ+) : 
  (N = 7 * b^2 + 7 * b + 7) →
  (∃ (x : ℤ), N = x^4) →
  (∀ (b' : ℕ+), b' < b → ¬∃ (x : ℤ), 7 * b'^2 + 7 * b' + 7 = x^4) →
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l83_8328


namespace NUMINAMATH_CALUDE_inequality_proof_l83_8386

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 + b*c) + 1 / (b^3 + c*a) + 1 / (c^3 + a*b) ≤ (a*b + b*c + c*a)^2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l83_8386


namespace NUMINAMATH_CALUDE_sum_of_ten_angles_is_1080_l83_8302

/-- A regular pentagon inscribed in a circle --/
structure RegularPentagonInCircle where
  /-- The measure of each interior angle of the pentagon --/
  interior_angle : ℝ
  /-- The measure of each exterior angle of the pentagon --/
  exterior_angle : ℝ
  /-- The measure of each angle inscribed in the segments outside the pentagon --/
  inscribed_angle : ℝ
  /-- The number of vertices in the pentagon --/
  num_vertices : ℕ
  /-- The interior angle of a regular pentagon is 108° --/
  interior_angle_eq : interior_angle = 108
  /-- The exterior angle is supplementary to the interior angle --/
  exterior_angle_eq : exterior_angle = 180 - interior_angle
  /-- The number of vertices in a pentagon is 5 --/
  num_vertices_eq : num_vertices = 5
  /-- The inscribed angle is half of the central angle --/
  inscribed_angle_eq : inscribed_angle = (360 - exterior_angle) / 2

/-- The sum of the ten angles in a regular pentagon inscribed in a circle --/
def sum_of_ten_angles (p : RegularPentagonInCircle) : ℝ :=
  p.num_vertices * (p.inscribed_angle + p.exterior_angle)

/-- Theorem: The sum of the ten angles is 1080° --/
theorem sum_of_ten_angles_is_1080 (p : RegularPentagonInCircle) :
  sum_of_ten_angles p = 1080 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_angles_is_1080_l83_8302


namespace NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l83_8357

theorem product_of_distinct_nonzero_reals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x - 2 / x = y - 2 / y) : x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l83_8357


namespace NUMINAMATH_CALUDE_scientific_notation_410000_l83_8395

theorem scientific_notation_410000 : 410000 = 4.1 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_410000_l83_8395


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l83_8381

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l83_8381


namespace NUMINAMATH_CALUDE_triangle_perimeter_l83_8364

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a * Real.cos B = 3 ∧
  b * Real.sin A = 4 ∧
  (1/2) * a * c * Real.sin B = 10 →
  a + b + c = 10 + 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l83_8364


namespace NUMINAMATH_CALUDE_right_triangle_legs_l83_8365

theorem right_triangle_legs (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  c = 10 → -- hypotenuse length
  r = 2 → -- inscribed circle radius
  a + b - c = 2 * r → -- formula for inscribed circle radius
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  ((a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6)) :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_legs_l83_8365


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l83_8353

theorem product_remainder_by_ten : (1734 * 5389 * 80607) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l83_8353


namespace NUMINAMATH_CALUDE_system_solution_l83_8319

theorem system_solution (x y z : ℚ) : 
  x = 2/7 ∧ y = 2/5 ∧ z = 2/3 ↔ 
  1/x + 1/y = 6 ∧ 1/y + 1/z = 4 ∧ 1/z + 1/x = 5 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l83_8319


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l83_8321

theorem max_product_constrained_sum (a b : ℝ) : 
  a + b = 1 → (∀ x y : ℝ, x + y = 1 → a * b ≥ x * y) → a * b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l83_8321


namespace NUMINAMATH_CALUDE_power_sum_five_l83_8334

theorem power_sum_five (x : ℝ) (h : x + 1/x = 5) : x^5 + 1/x^5 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_five_l83_8334


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_is_three_fourths_l83_8303

/-- Represents a cube made of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_area_fraction (c : CompositeCube) : ℚ :=
  sorry

/-- Theorem: The fraction of white surface area for a specific composite cube is 3/4 -/
theorem white_surface_area_fraction_is_three_fourths :
  let c : CompositeCube := {
    edge_length := 4,
    small_cube_count := 64,
    white_cube_count := 48,
    black_cube_count := 16
  }
  white_surface_area_fraction c = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_is_three_fourths_l83_8303


namespace NUMINAMATH_CALUDE_distribute_5_3_l83_8399

/-- The number of ways to distribute n college graduates to k employers,
    with each employer receiving at least 1 graduate -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 college graduates to 3 employers,
    with each employer receiving at least 1 graduate, is 150 -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l83_8399


namespace NUMINAMATH_CALUDE_point_on_line_l83_8330

/-- A point (x, y) lies on the line passing through (2, -4) and (8, 16) if and only if y = (10/3)x - 32/3 -/
theorem point_on_line (x y : ℝ) : 
  (y = (10/3)*x - 32/3) ↔ 
  (∃ t : ℝ, x = 2 + 6*t ∧ y = -4 + 20*t) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l83_8330


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l83_8338

/-- A rectangle with perimeter 14 cm and area 12 square cm has a diagonal of length 5 cm -/
theorem rectangle_diagonal (l w : ℝ) : 
  (2 * l + 2 * w = 14) →  -- Perimeter condition
  (l * w = 12) →          -- Area condition
  Real.sqrt (l^2 + w^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l83_8338


namespace NUMINAMATH_CALUDE_set_equivalence_l83_8318

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

theorem set_equivalence : {x : ℝ | x ≥ 1} = (Set.univ : Set ℝ) \ (M ∪ N) := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l83_8318


namespace NUMINAMATH_CALUDE_iris_jacket_purchase_l83_8361

theorem iris_jacket_purchase (jacket_price shorts_price pants_price : ℕ)
  (shorts_quantity pants_quantity : ℕ) (total_spent : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  pants_price = 12 →
  shorts_quantity = 2 →
  pants_quantity = 4 →
  total_spent = 90 →
  ∃ (jacket_quantity : ℕ), 
    jacket_quantity * jacket_price + 
    shorts_quantity * shorts_price + 
    pants_quantity * pants_price = total_spent ∧
    jacket_quantity = 3 :=
by sorry

end NUMINAMATH_CALUDE_iris_jacket_purchase_l83_8361


namespace NUMINAMATH_CALUDE_vector_function_properties_l83_8398

noncomputable def a (x : ℝ) : ℝ × ℝ := (1 + Real.sin (2 * x), Real.sin x - Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (1, Real.sin x + Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_function_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x : ℝ), f x = M) ∧
  (∃ (S : Set ℝ), S = {x | ∃ (k : ℤ), x = 3 * Real.pi / 8 + k * Real.pi} ∧
    ∀ (x : ℝ), x ∈ S ↔ f x = Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_function_properties_l83_8398


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_8191_l83_8372

def greatest_prime_factor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_greatest_prime_factor_8191 :
  sum_of_digits (greatest_prime_factor 8191) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_8191_l83_8372


namespace NUMINAMATH_CALUDE_video_game_cost_is_60_l83_8375

/-- Represents the cost of the video game given Bucky's fish-catching earnings --/
def video_game_cost (last_weekend_earnings trout_price bluegill_price total_fish trout_percentage additional_savings : ℝ) : ℝ :=
  let trout_count := trout_percentage * total_fish
  let bluegill_count := total_fish - trout_count
  let sunday_earnings := trout_count * trout_price + bluegill_count * bluegill_price
  last_weekend_earnings + sunday_earnings + additional_savings

/-- Theorem stating the cost of the video game based on given conditions --/
theorem video_game_cost_is_60 :
  video_game_cost 35 5 4 5 0.6 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_is_60_l83_8375


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l83_8306

theorem reciprocal_of_sum : (1 / (1/3 + 1/5) : ℚ) = 15/8 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l83_8306


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l83_8348

/-- Two vectors in ℝ² are orthogonal if their dot product is zero -/
def orthogonal (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The vector a -/
def a : ℝ × ℝ := (4, 2)

/-- The vector b -/
def b (y : ℝ) : ℝ × ℝ := (6, y)

/-- Theorem: If a and b are orthogonal, then y = -12 -/
theorem orthogonal_vectors (y : ℝ) :
  orthogonal a (b y) → y = -12 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l83_8348


namespace NUMINAMATH_CALUDE_both_locks_stall_time_l83_8331

/-- The time (in minutes) the first lock stalls the raccoons -/
def first_lock_time : ℕ := 5

/-- The time (in minutes) the second lock stalls the raccoons -/
def second_lock_time : ℕ := 3 * first_lock_time - 3

/-- The time (in minutes) both locks together stall the raccoons -/
def both_locks_time : ℕ := 5 * second_lock_time

/-- Theorem stating that both locks together stall the raccoons for 60 minutes -/
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_both_locks_stall_time_l83_8331


namespace NUMINAMATH_CALUDE_painted_faces_count_l83_8336

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool := true

/-- Calculates the number of smaller cubes with at least two painted faces -/
def cubes_with_two_or_more_painted_faces (c : PaintedCube 4) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube 4) : 
  cubes_with_two_or_more_painted_faces c = 32 := by sorry

end NUMINAMATH_CALUDE_painted_faces_count_l83_8336


namespace NUMINAMATH_CALUDE_expand_expression_l83_8326

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l83_8326


namespace NUMINAMATH_CALUDE_product_of_numbers_l83_8332

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l83_8332


namespace NUMINAMATH_CALUDE_remainder_theorem_l83_8383

theorem remainder_theorem (n : ℕ) : (2 * n) % 4 = 2 → n % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l83_8383


namespace NUMINAMATH_CALUDE_phone_number_pricing_l83_8300

theorem phone_number_pricing (X Y : ℤ) : 
  (0 < X ∧ X < 250) →
  (0 < Y ∧ Y < 250) →
  125 * X - 64 * Y = 5 →
  ((X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205)) := by
sorry

end NUMINAMATH_CALUDE_phone_number_pricing_l83_8300


namespace NUMINAMATH_CALUDE_megan_remaining_acorns_l83_8320

def initial_acorns : ℕ := 16
def acorns_given : ℕ := 7

theorem megan_remaining_acorns :
  initial_acorns - acorns_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_megan_remaining_acorns_l83_8320


namespace NUMINAMATH_CALUDE_third_divisor_is_three_l83_8312

def smallest_number : ℕ := 1011
def diminished_number : ℕ := smallest_number - 3

theorem third_divisor_is_three :
  ∃ (x : ℕ), x ≠ 12 ∧ x ≠ 16 ∧ x ≠ 21 ∧ x ≠ 28 ∧
  diminished_number % 12 = 0 ∧
  diminished_number % 16 = 0 ∧
  diminished_number % x = 0 ∧
  diminished_number % 21 = 0 ∧
  diminished_number % 28 = 0 ∧
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_divisor_is_three_l83_8312


namespace NUMINAMATH_CALUDE_f_continuity_and_discontinuity_l83_8390

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x < -3 then (x^2 + 3*x - 1) / (x + 2)
  else if x ≤ 4 then (x + 2)^2
  else 9*x + 1

-- Define continuity at a point
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a| < ε

-- Define left and right limits
def has_limit_at_left (f : ℝ → ℝ) (a : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, a - δ < x ∧ x < a → |f x - L| < ε

def has_limit_at_right (f : ℝ → ℝ) (a : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, a < x ∧ x < a + δ → |f x - L| < ε

-- Define jump discontinuity
def jump_discontinuity (f : ℝ → ℝ) (a : ℝ) (jump : ℝ) : Prop :=
  ∃ L₁ L₂, has_limit_at_left f a L₁ ∧ has_limit_at_right f a L₂ ∧ L₂ - L₁ = jump

-- Theorem statement
theorem f_continuity_and_discontinuity :
  continuous_at f (-3) ∧ jump_discontinuity f 4 1 :=
sorry

end NUMINAMATH_CALUDE_f_continuity_and_discontinuity_l83_8390


namespace NUMINAMATH_CALUDE_no_solution_when_p_divides_x_l83_8378

theorem no_solution_when_p_divides_x (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∀ (x y : ℕ), x > 0 → y > 0 → p ∣ x → x^2 - 1 ≠ y^p := by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_p_divides_x_l83_8378


namespace NUMINAMATH_CALUDE_geometric_series_sum_l83_8389

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/4 is 4/3 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/4
  let S : ℝ := ∑' n, a * r^n
  S = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l83_8389


namespace NUMINAMATH_CALUDE_tangent_slopes_product_l83_8308

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the circle where P lies
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the tangent line from P(x₀, y₀) to C with slope k
def tangent_line (x₀ y₀ k x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the condition for a line to be tangent to C
def is_tangent (x₀ y₀ k : ℝ) : Prop :=
  ∃ x y, tangent_line x₀ y₀ k x y ∧ ellipse_C x y

-- Main theorem
theorem tangent_slopes_product (x₀ y₀ k₁ k₂ : ℝ) :
  circle_P x₀ y₀ →
  is_tangent x₀ y₀ k₁ →
  is_tangent x₀ y₀ k₂ →
  k₁ ≠ k₂ →
  k₁ * k₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slopes_product_l83_8308


namespace NUMINAMATH_CALUDE_line_point_at_47_l83_8350

/-- A line passing through three given points -/
structure Line where
  -- Define the line using two points
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  -- Ensure the third point lies on the line
  x3 : ℝ
  y3 : ℝ
  point_on_line : (y3 - y1) / (x3 - x1) = (y2 - y1) / (x2 - x1)

/-- Theorem: For the given line, when x = 47, y = 143 -/
theorem line_point_at_47 (l : Line) 
  (h1 : l.x1 = 2 ∧ l.y1 = 8)
  (h2 : l.x2 = 6 ∧ l.y2 = 20)
  (h3 : l.x3 = 10 ∧ l.y3 = 32) :
  let m := (l.y2 - l.y1) / (l.x2 - l.x1)
  let b := l.y1 - m * l.x1
  m * 47 + b = 143 := by
  sorry

end NUMINAMATH_CALUDE_line_point_at_47_l83_8350


namespace NUMINAMATH_CALUDE_four_Z_three_equals_37_l83_8394

-- Define the Z operation
def Z (a b : ℕ) : ℕ := a^2 + a*b + b^2

-- Theorem to prove
theorem four_Z_three_equals_37 : Z 4 3 = 37 := by sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_37_l83_8394


namespace NUMINAMATH_CALUDE_square_diagonal_length_l83_8345

/-- The diagonal length of a square with area 72 square meters is 12 meters. -/
theorem square_diagonal_length (area : ℝ) (side : ℝ) (diagonal : ℝ) : 
  area = 72 → 
  area = side ^ 2 → 
  diagonal ^ 2 = 2 * side ^ 2 → 
  diagonal = 12 := by
  sorry


end NUMINAMATH_CALUDE_square_diagonal_length_l83_8345


namespace NUMINAMATH_CALUDE_kids_left_playing_l83_8316

theorem kids_left_playing (initial_kids : ℝ) (kids_going_home : ℝ) :
  initial_kids = 22.0 →
  kids_going_home = 14.0 →
  initial_kids - kids_going_home = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_kids_left_playing_l83_8316


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_2023_l83_8380

theorem simplify_and_evaluate (x : ℝ) : (x + 1)^2 - x * (x + 1) = x + 1 :=
  sorry

theorem evaluate_at_2023 : (2023 + 1)^2 - 2023 * (2023 + 1) = 2024 :=
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_2023_l83_8380


namespace NUMINAMATH_CALUDE_cost_equation_holds_l83_8376

/-- Represents the cost equation for notebooks and colored pens --/
def cost_equation (x : ℕ) : Prop :=
  let total_items : ℕ := 20
  let total_cost : ℕ := 50
  let notebook_cost : ℕ := 4
  let pen_cost : ℕ := 2
  2 * (total_items - x) + notebook_cost * x = total_cost

/-- Theorem stating the cost equation holds for the given scenario --/
theorem cost_equation_holds : ∃ x : ℕ, cost_equation x := by sorry

end NUMINAMATH_CALUDE_cost_equation_holds_l83_8376


namespace NUMINAMATH_CALUDE_sqrt_equation_unique_solution_l83_8360

theorem sqrt_equation_unique_solution :
  ∃! x : ℝ, Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 6 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_unique_solution_l83_8360


namespace NUMINAMATH_CALUDE_semicircle_radius_l83_8363

theorem semicircle_radius (p : ℝ) (h : p = 108) : 
  ∃ r : ℝ, p = r * (Real.pi + 2) ∧ r = p / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l83_8363


namespace NUMINAMATH_CALUDE_circle_radius_on_right_triangle_l83_8322

theorem circle_radius_on_right_triangle (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a = 7.5 →  -- shorter leg
  b = 10 →  -- longer leg (diameter of circle)
  6^2 + (c - r)^2 = r^2 →  -- chord condition
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_on_right_triangle_l83_8322


namespace NUMINAMATH_CALUDE_amount_owed_l83_8335

-- Define the rate per room
def rate_per_room : ℚ := 11 / 2

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 7 / 3

-- Theorem statement
theorem amount_owed : rate_per_room * rooms_cleaned = 77 / 6 := by
  sorry

end NUMINAMATH_CALUDE_amount_owed_l83_8335


namespace NUMINAMATH_CALUDE_tea_bags_in_box_l83_8349

theorem tea_bags_in_box : ∀ n : ℕ,
  (2 * n ≤ 41 ∧ 41 ≤ 3 * n) ∧
  (2 * n ≤ 58 ∧ 58 ≤ 3 * n) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_tea_bags_in_box_l83_8349


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_system_solution_l83_8387

-- Part 1: Inequality solution
theorem inequality_solution (x : ℝ) :
  (2 * x - 1) / 2 ≥ 1 - (x + 1) / 3 ↔ x ≥ 7 / 8 :=
by sorry

-- Part 2: System of inequalities solution
theorem inequality_system_solution (x : ℝ) :
  (-2 * x ≤ -3 ∧ x / 2 < 2) ↔ (3 / 2 ≤ x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_system_solution_l83_8387


namespace NUMINAMATH_CALUDE_triangle_max_area_l83_8371

/-- The maximum area of a triangle ABC where b = 3a and c = 2 -/
theorem triangle_max_area (a b c : ℝ) (h1 : b = 3 * a) (h2 : c = 2) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let area := (1/2) * b * c * Real.sin A
  ∀ x > 0, area ≤ Real.sqrt 2 / 2 ∧ 
  ∃ a₀ > 0, area = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l83_8371


namespace NUMINAMATH_CALUDE_price_per_dozen_calculation_l83_8341

/-- The price per dozen of additional doughnuts -/
def price_per_dozen (first_doughnut_price : ℚ) (total_cost : ℚ) (total_doughnuts : ℕ) : ℚ :=
  (total_cost - first_doughnut_price) / ((total_doughnuts - 1 : ℚ) / 12)

/-- Theorem stating the price per dozen of additional doughnuts -/
theorem price_per_dozen_calculation (first_doughnut_price : ℚ) (total_cost : ℚ) (total_doughnuts : ℕ) 
  (h1 : first_doughnut_price = 1)
  (h2 : total_cost = 24)
  (h3 : total_doughnuts = 48) :
  price_per_dozen first_doughnut_price total_cost total_doughnuts = 276 / 47 :=
by sorry

end NUMINAMATH_CALUDE_price_per_dozen_calculation_l83_8341


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_max_sum_at_25_l83_8396

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  second_eighth_sum : a 2 + a 8 = 82
  sum_equality : S 41 = S 9

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∃ d : ℝ, ∀ n : ℕ, seq.a n = 51 - 2 * n) ∧
  (∃ n : ℕ, seq.S n = 625 ∧ ∀ m : ℕ, seq.S m ≤ seq.S n) := by
  sorry

/-- The maximum value of S_n occurs when n = 25 -/
theorem max_sum_at_25 (seq : ArithmeticSequence) :
  seq.S 25 = 625 ∧ ∀ n : ℕ, seq.S n ≤ seq.S 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_max_sum_at_25_l83_8396


namespace NUMINAMATH_CALUDE_condition_relationship_l83_8314

theorem condition_relationship :
  (∀ x : ℝ, |x - 1| ≤ 1 → 2 - x ≥ 0) ∧
  (∃ x : ℝ, 2 - x ≥ 0 ∧ |x - 1| > 1) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l83_8314
