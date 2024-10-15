import Mathlib

namespace NUMINAMATH_CALUDE_ball_selection_properties_l2296_229612

structure BallSelection where
  total_balls : Nat
  red_balls : Nat
  white_balls : Nat
  balls_drawn : Nat

def P (event : Set ℝ) : ℝ := sorry

def A (bs : BallSelection) : Set ℝ := sorry
def B (bs : BallSelection) : Set ℝ := sorry
def D (bs : BallSelection) : Set ℝ := sorry

theorem ball_selection_properties (bs : BallSelection) 
  (h1 : bs.total_balls = 4)
  (h2 : bs.red_balls = 2)
  (h3 : bs.white_balls = 2)
  (h4 : bs.balls_drawn = 2) :
  (P (A bs ∩ B bs) = P (A bs) * P (B bs)) ∧
  (P (A bs) + P (D bs) = 1) ∧
  (P (B bs ∩ D bs) = P (B bs) * P (D bs)) := by
  sorry

end NUMINAMATH_CALUDE_ball_selection_properties_l2296_229612


namespace NUMINAMATH_CALUDE_new_person_weight_l2296_229633

theorem new_person_weight 
  (n : ℕ) 
  (initial_weight : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) :
  n = 9 →
  initial_weight = 65 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (n : ℝ) * weight_increase + replaced_weight = 87.5 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2296_229633


namespace NUMINAMATH_CALUDE_gcd_888_1147_l2296_229605

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_888_1147_l2296_229605


namespace NUMINAMATH_CALUDE_minimize_expression_l2296_229690

theorem minimize_expression (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) ≥ 3 ∧ (x + 4 / (x + 1) = 3 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_minimize_expression_l2296_229690


namespace NUMINAMATH_CALUDE_min_sum_squares_l2296_229666

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2296_229666


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2296_229609

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The incident point of the light ray -/
def P : Point := ⟨5, 3⟩

/-- The point where the light ray intersects the x-axis -/
def Q : Point := ⟨2, 0⟩

/-- Function to calculate the reflected point across the x-axis -/
def reflect_across_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The reflected point of P across the x-axis -/
def P' : Point := reflect_across_x_axis P

/-- Function to create a line from two points -/
def line_from_points (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  ⟨a, b, c⟩

/-- The reflected ray line -/
def reflected_ray : Line := line_from_points Q P'

/-- Theorem stating that the reflected ray line has the equation x + y - 2 = 0 -/
theorem reflected_ray_equation :
  reflected_ray = ⟨1, 1, -2⟩ := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2296_229609


namespace NUMINAMATH_CALUDE_pencils_purchased_l2296_229655

theorem pencils_purchased (num_pens : ℕ) (total_cost : ℝ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : num_pens = 30)
  (h2 : total_cost = 510)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 12) :
  (total_cost - num_pens * pen_price) / pencil_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_purchased_l2296_229655


namespace NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l2296_229699

/-- Proves that a rectangle with area 90 cm² and length-to-width ratio 5:3 cannot fit in a 100 cm² square -/
theorem rectangle_cannot_fit_in_square : ¬ ∃ (length width : ℝ),
  (length * width = 90) ∧ 
  (length / width = 5 / 3) ∧
  (length ≤ 10) ∧
  (width ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l2296_229699


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2296_229624

theorem expand_and_simplify (x : ℝ) : 
  (1 + x^2 + x^4) * (1 - x^3 + x^5) = 1 - x^3 + x^2 + x^4 + x^9 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2296_229624


namespace NUMINAMATH_CALUDE_equation_solution_l2296_229687

theorem equation_solution :
  let f := fun x : ℝ => 2 / x - (3 / x) * (5 / x) + 1 / 2
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    f x₁ = 0 ∧ f x₂ = 0 ∧
    x₁ = -2 + Real.sqrt 34 ∧ 
    x₂ = -2 - Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2296_229687


namespace NUMINAMATH_CALUDE_yellow_leaves_count_l2296_229616

theorem yellow_leaves_count (thursday_leaves friday_leaves : ℕ) 
  (brown_percent green_percent : ℚ) :
  thursday_leaves = 12 →
  friday_leaves = 13 →
  brown_percent = 1/5 →
  green_percent = 1/5 →
  (thursday_leaves + friday_leaves : ℚ) * (1 - brown_percent - green_percent) = 15 :=
by sorry

end NUMINAMATH_CALUDE_yellow_leaves_count_l2296_229616


namespace NUMINAMATH_CALUDE_highway_intersection_probability_l2296_229686

theorem highway_intersection_probability (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  let p_enter := 1 / n
  let p_exit := 1 / n
  (k - 1) * (n - k) * p_enter * p_exit +
  p_enter * (n - k) * p_exit +
  (k - 1) * p_enter * p_exit =
  (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := by
  sorry


end NUMINAMATH_CALUDE_highway_intersection_probability_l2296_229686


namespace NUMINAMATH_CALUDE_brick_in_box_probability_l2296_229678

/-- A set of six distinct numbers from 1 to 500 -/
def SixNumbers : Type := { s : Finset ℕ // s.card = 6 ∧ ∀ n ∈ s, 1 ≤ n ∧ n ≤ 500 }

/-- The three largest numbers from a set of six numbers -/
def largestThree (s : SixNumbers) : Finset ℕ :=
  sorry

/-- The three smallest numbers from a set of six numbers -/
def smallestThree (s : SixNumbers) : Finset ℕ :=
  sorry

/-- Whether a brick with given dimensions fits in a box with given dimensions -/
def fits (brick box : Finset ℕ) : Prop :=
  sorry

/-- The probability of a brick fitting in a box -/
def fitProbability : ℚ :=
  sorry

theorem brick_in_box_probability :
  fitProbability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_brick_in_box_probability_l2296_229678


namespace NUMINAMATH_CALUDE_discount_difference_l2296_229634

theorem discount_difference (bill : ℝ) (d1 d2 d3 d4 : ℝ) :
  bill = 12000 ∧ d1 = 0.3 ∧ d2 = 0.2 ∧ d3 = 0.06 ∧ d4 = 0.04 →
  bill * (1 - d2) * (1 - d3) * (1 - d4) - bill * (1 - d1) = 263.04 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l2296_229634


namespace NUMINAMATH_CALUDE_arithmetic_mean_equidistant_l2296_229642

/-- The arithmetic mean of two real numbers is equidistant from both numbers. -/
theorem arithmetic_mean_equidistant (a b : ℝ) : 
  |((a + b) / 2) - a| = |b - ((a + b) / 2)| := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_equidistant_l2296_229642


namespace NUMINAMATH_CALUDE_vote_participation_l2296_229622

theorem vote_participation (veggie_percentage : ℝ) (veggie_votes : ℕ) (total_students : ℕ) : 
  veggie_percentage = 0.28 →
  veggie_votes = 280 →
  (veggie_percentage * total_students : ℝ) = veggie_votes →
  total_students = 1000 := by
sorry

end NUMINAMATH_CALUDE_vote_participation_l2296_229622


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2296_229614

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2296_229614


namespace NUMINAMATH_CALUDE_box_area_l2296_229658

theorem box_area (V : ℝ) (A2 A3 : ℝ) (hV : V = 720) (hA2 : A2 = 72) (hA3 : A3 = 60) :
  ∃ (L W H : ℝ), L > 0 ∧ W > 0 ∧ H > 0 ∧ 
    L * W * H = V ∧
    W * H = A2 ∧
    L * H = A3 ∧
    L * W = 120 :=
by sorry

end NUMINAMATH_CALUDE_box_area_l2296_229658


namespace NUMINAMATH_CALUDE_scientific_notation_of_million_l2296_229682

theorem scientific_notation_of_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1000000 = a * (10 : ℝ) ^ n ∧ a = 1 ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_million_l2296_229682


namespace NUMINAMATH_CALUDE_count_even_numbers_between_150_and_350_l2296_229610

theorem count_even_numbers_between_150_and_350 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ 150 < n ∧ n < 350) (Finset.range 350)).card = 99 := by
  sorry

end NUMINAMATH_CALUDE_count_even_numbers_between_150_and_350_l2296_229610


namespace NUMINAMATH_CALUDE_intersection_property_l2296_229626

-- Define the parabola and line
def parabola (x : ℝ) : ℝ := 2 * x^2
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the intersection points A and B
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

-- Define the midpoint M
def M (k : ℝ) : ℝ × ℝ := sorry

-- Define point N on x-axis
def N (k : ℝ) : ℝ × ℝ := sorry

-- Define vectors NA and NB
def NA (k : ℝ) : ℝ × ℝ := sorry
def NB (k : ℝ) : ℝ × ℝ := sorry

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

theorem intersection_property (k : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ parabola x₁ = line k x₁ ∧ parabola x₂ = line k x₂) →
  (dot_product (NA k) (NB k) = 0) →
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_property_l2296_229626


namespace NUMINAMATH_CALUDE_division_and_addition_l2296_229693

theorem division_and_addition : (12 / (1/6)) + 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l2296_229693


namespace NUMINAMATH_CALUDE_triangle_problem_l2296_229654

theorem triangle_problem (A B C : Real) (a b c : Real) :
  b = 3 * Real.sqrt 2 →
  Real.cos A = Real.sqrt 6 / 3 →
  B = A + π / 2 →
  a = 3 ∧ Real.cos (2 * C) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2296_229654


namespace NUMINAMATH_CALUDE_smallest_valid_n_l2296_229618

def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

def is_valid_n (n : ℕ) : Prop :=
  ∀ i ∈ Finset.range 6, ∃ a : ℕ, a > 0 ∧ doubling_sum a (i + 1) = n

theorem smallest_valid_n :
  is_valid_n 9765 ∧ ∀ m < 9765, ¬is_valid_n m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l2296_229618


namespace NUMINAMATH_CALUDE_customer_payment_proof_l2296_229643

-- Define the cost price of the computer table
def cost_price : ℕ := 6500

-- Define the markup percentage
def markup_percentage : ℚ := 30 / 100

-- Define the function to calculate the selling price
def selling_price (cost : ℕ) (markup : ℚ) : ℚ :=
  cost * (1 + markup)

-- Theorem statement
theorem customer_payment_proof :
  selling_price cost_price markup_percentage = 8450 := by
  sorry

end NUMINAMATH_CALUDE_customer_payment_proof_l2296_229643


namespace NUMINAMATH_CALUDE_jellybean_problem_l2296_229684

theorem jellybean_problem (initial_bags : ℕ) (initial_average : ℕ) (average_increase : ℕ) :
  initial_bags = 34 →
  initial_average = 117 →
  average_increase = 7 →
  (initial_bags * initial_average + (initial_bags + 1) * average_increase) = 362 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l2296_229684


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_sum_interior_angles_correct_l2296_229665

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180

theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2 : ℝ) * 180 :=
by
  sorry

/-- The sum of interior angles of a triangle -/
axiom triangle_sum : sum_interior_angles 3 = 180

/-- The sum of interior angles of a quadrilateral -/
axiom quadrilateral_sum : sum_interior_angles 4 = 360

theorem sum_interior_angles_correct (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2 : ℝ) * 180 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_formula_sum_interior_angles_correct_l2296_229665


namespace NUMINAMATH_CALUDE_bills_trips_l2296_229627

theorem bills_trips (total_trips : ℕ) (jeans_trips : ℕ) (h1 : total_trips = 40) (h2 : jeans_trips = 23) :
  total_trips - jeans_trips = 17 :=
by sorry

end NUMINAMATH_CALUDE_bills_trips_l2296_229627


namespace NUMINAMATH_CALUDE_special_list_median_l2296_229675

/-- Represents the list where each integer n from 1 to 300 appears exactly n times -/
def special_list : List ℕ := sorry

/-- The total number of elements in the special list -/
def total_elements : ℕ := (300 * 301) / 2

/-- The position of the median in the special list -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- The median value of the special list -/
def median_value : ℕ := 212

theorem special_list_median :
  median_value = 212 := by sorry

end NUMINAMATH_CALUDE_special_list_median_l2296_229675


namespace NUMINAMATH_CALUDE_derivative_at_one_l2296_229620

open Real

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * (deriv f 1) + log x) :
  deriv f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2296_229620


namespace NUMINAMATH_CALUDE_derivative_of_f_derivative_of_f_at_2_l2296_229667

-- Define the function f(x) = x^2 + x
def f (x : ℝ) : ℝ := x^2 + x

-- Theorem 1: The derivative of f(x) is 2x + 1
theorem derivative_of_f (x : ℝ) : deriv f x = 2 * x + 1 := by sorry

-- Theorem 2: The derivative of f(x) at x = 2 is 5
theorem derivative_of_f_at_2 : deriv f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_derivative_of_f_at_2_l2296_229667


namespace NUMINAMATH_CALUDE_initial_worksheets_count_l2296_229638

/-- Given that a teacher would have 20 worksheets to grade after grading 4 and receiving 18 more,
    prove that she initially had 6 worksheets to grade. -/
theorem initial_worksheets_count : ∀ x : ℕ, x - 4 + 18 = 20 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_worksheets_count_l2296_229638


namespace NUMINAMATH_CALUDE_license_plate_combinations_license_plate_combinations_eq_187200_l2296_229613

theorem license_plate_combinations : ℕ :=
  let total_letters : ℕ := 26
  let letter_positions : ℕ := 4
  let digit_positions : ℕ := 3
  let repeated_letter_choices : ℕ := total_letters
  let non_repeated_letter_choices : ℕ := total_letters - 1
  let repeated_letter_arrangements : ℕ := Nat.choose letter_positions (letter_positions - 1)
  let first_digit_choices : ℕ := 10
  let second_digit_choices : ℕ := 9
  let third_digit_choices : ℕ := 8

  repeated_letter_choices * non_repeated_letter_choices * repeated_letter_arrangements *
  first_digit_choices * second_digit_choices * third_digit_choices

theorem license_plate_combinations_eq_187200 : license_plate_combinations = 187200 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_license_plate_combinations_eq_187200_l2296_229613


namespace NUMINAMATH_CALUDE_muffin_combinations_l2296_229608

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of muffin types -/
def muffin_types : ℕ := 4

/-- The number of additional muffins to distribute -/
def additional_muffins : ℕ := 4

theorem muffin_combinations :
  distribute additional_muffins muffin_types = 35 := by
  sorry

end NUMINAMATH_CALUDE_muffin_combinations_l2296_229608


namespace NUMINAMATH_CALUDE_pigeonhole_zodiac_signs_pigeonhole_birthday_weekday_l2296_229617

/-- The number of zodiac signs -/
def num_zodiac_signs : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The minimum number of employees needed to ensure at least two have the same zodiac sign -/
def min_employees_same_sign : ℕ := num_zodiac_signs + 1

/-- The minimum number of employees needed to ensure at least four have birthdays on the same day of the week -/
def min_employees_same_day : ℕ := days_in_week * 3 + 1

theorem pigeonhole_zodiac_signs :
  min_employees_same_sign = 13 :=
sorry

theorem pigeonhole_birthday_weekday :
  min_employees_same_day = 22 :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_zodiac_signs_pigeonhole_birthday_weekday_l2296_229617


namespace NUMINAMATH_CALUDE_min_value_theorem_l2296_229601

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  x^2 * y^3 * z^2 ≥ 1/2268 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    1/x₀ + 1/y₀ + 1/z₀ = 9 ∧ 
    x₀^2 * y₀^3 * z₀^2 = 1/2268 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2296_229601


namespace NUMINAMATH_CALUDE_perpendicular_bisector_implies_m_equals_three_l2296_229637

/-- Given two points A and B, if the equation of the perpendicular bisector 
    of segment AB is x + 2y - 2 = 0, then the x-coordinate of B is 3. -/
theorem perpendicular_bisector_implies_m_equals_three 
  (A B : ℝ × ℝ) 
  (h1 : A = (1, -2))
  (h2 : B.2 = 2)
  (h3 : ∀ x y : ℝ, (x + 2*y - 2 = 0) ↔ 
    (x = (A.1 + B.1)/2 ∧ y = (A.2 + B.2)/2)) : 
  B.1 = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_implies_m_equals_three_l2296_229637


namespace NUMINAMATH_CALUDE_average_disk_space_per_hour_l2296_229683

/-- Proves that the average disk space per hour of music in a library
    containing 12 days of music and occupying 16,000 megabytes,
    rounded to the nearest whole number, is 56 megabytes. -/
theorem average_disk_space_per_hour (days : ℕ) (total_space : ℕ) 
  (h1 : days = 12) (h2 : total_space = 16000) : 
  round ((total_space : ℝ) / (days * 24)) = 56 := by
  sorry

#check average_disk_space_per_hour

end NUMINAMATH_CALUDE_average_disk_space_per_hour_l2296_229683


namespace NUMINAMATH_CALUDE_stratified_sampling_best_l2296_229649

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | RandomNumberTable
  | Stratified

/-- Represents a high school population -/
structure HighSchoolPopulation where
  grades : List Nat
  students : Nat

/-- Represents a survey goal -/
inductive SurveyGoal
  | PsychologicalPressure

/-- Determines the best sampling method given a high school population and survey goal -/
def bestSamplingMethod (population : HighSchoolPopulation) (goal : SurveyGoal) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the best method for the given scenario -/
theorem stratified_sampling_best 
  (population : HighSchoolPopulation) 
  (h1 : population.grades.length > 1) 
  (goal : SurveyGoal) 
  (h2 : goal = SurveyGoal.PsychologicalPressure) : 
  bestSamplingMethod population goal = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_best_l2296_229649


namespace NUMINAMATH_CALUDE_family_income_problem_l2296_229604

theorem family_income_problem (initial_avg : ℚ) (new_avg : ℚ) (deceased_income : ℚ) 
  (h1 : initial_avg = 735)
  (h2 : new_avg = 650)
  (h3 : deceased_income = 905) :
  ∃ n : ℕ, n > 0 ∧ n * initial_avg - (n - 1) * new_avg = deceased_income ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_income_problem_l2296_229604


namespace NUMINAMATH_CALUDE_remaining_practice_time_l2296_229664

/-- The total practice time in hours for the week -/
def total_practice_hours : ℝ := 7.5

/-- The number of days with known practice time -/
def known_practice_days : ℕ := 2

/-- The practice time in minutes for each of the known practice days -/
def practice_per_known_day : ℕ := 86

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem remaining_practice_time :
  hours_to_minutes total_practice_hours - (known_practice_days * practice_per_known_day) = 278 := by
  sorry

end NUMINAMATH_CALUDE_remaining_practice_time_l2296_229664


namespace NUMINAMATH_CALUDE_wendys_candy_boxes_l2296_229671

/-- Proves that Wendy had 2 boxes of candy given the problem conditions -/
theorem wendys_candy_boxes :
  ∀ (brother_candy : ℕ) (pieces_per_box : ℕ) (total_candy : ℕ) (wendys_boxes : ℕ),
    brother_candy = 6 →
    pieces_per_box = 3 →
    total_candy = 12 →
    total_candy = brother_candy + (wendys_boxes * pieces_per_box) →
    wendys_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendys_candy_boxes_l2296_229671


namespace NUMINAMATH_CALUDE_nell_card_difference_l2296_229621

/-- Represents the number of cards Nell has -/
structure CardCounts where
  initial_baseball : Nat
  initial_ace : Nat
  final_baseball : Nat
  final_ace : Nat

/-- Calculates the difference between Ace cards and baseball cards -/
def ace_baseball_difference (counts : CardCounts) : Int :=
  counts.final_ace - counts.final_baseball

/-- Theorem stating the difference between Ace cards and baseball cards -/
theorem nell_card_difference (counts : CardCounts) 
  (h1 : counts.initial_baseball = 239)
  (h2 : counts.initial_ace = 38)
  (h3 : counts.final_baseball = 111)
  (h4 : counts.final_ace = 376) :
  ace_baseball_difference counts = 265 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l2296_229621


namespace NUMINAMATH_CALUDE_stock_price_change_l2296_229685

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_decrease := initial_price * (1 - 0.05)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 4.5 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l2296_229685


namespace NUMINAMATH_CALUDE_triangle_has_two_acute_angles_l2296_229625

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the property that the sum of angles in a triangle is 180°
def validTriangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define an acute angle
def isAcute (angle : Real) : Prop := angle < 90

-- Theorem statement
theorem triangle_has_two_acute_angles (t : Triangle) (h : validTriangle t) :
  ∃ (a b : Real), (a = t.angle1 ∨ a = t.angle2 ∨ a = t.angle3) ∧
                  (b = t.angle1 ∨ b = t.angle2 ∨ b = t.angle3) ∧
                  (a ≠ b) ∧
                  isAcute a ∧ isAcute b :=
sorry

end NUMINAMATH_CALUDE_triangle_has_two_acute_angles_l2296_229625


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2296_229615

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = -x^2 + 4}
def N : Set ℝ := {x | x > 0}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ N = Set.Ioo 0 4 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2296_229615


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2296_229688

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Predicate to check if a function is a polynomial with integer coefficients -/
def is_int_polynomial (p : ℤ → ℤ) : Prop := sorry

theorem polynomial_inequality (p : IntPolynomial) (n : ℤ) 
  (h_poly : is_int_polynomial p)
  (h_ineq : p (-n) < p n ∧ p n < n) : 
  p (-n) < -n := by sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2296_229688


namespace NUMINAMATH_CALUDE_correct_divisor_l2296_229629

theorem correct_divisor (X D : ℕ) 
  (h1 : X % D = 0)
  (h2 : X / (D - 12) = 42)
  (h3 : X / D = 24) :
  D = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l2296_229629


namespace NUMINAMATH_CALUDE_optimal_station_location_l2296_229657

/-- Represents the optimal station location problem for Factory A --/
theorem optimal_station_location :
  let num_buildings : ℕ := 5
  let building_distances : List ℝ := [0, 50, 100, 150, 200]
  let worker_counts : List ℕ := [1, 2, 3, 4, 5]
  let total_workers : ℕ := worker_counts.sum
  
  -- Function to calculate total walking distance for a given station location
  let total_distance (station_location : ℝ) : ℝ :=
    List.sum (List.zipWith (fun d w => w * |station_location - d|) building_distances worker_counts)
  
  -- The optimal location minimizes the total walking distance
  ∃ (optimal_location : ℝ),
    (∀ (x : ℝ), total_distance optimal_location ≤ total_distance x) ∧
    optimal_location = 150
  := by sorry

end NUMINAMATH_CALUDE_optimal_station_location_l2296_229657


namespace NUMINAMATH_CALUDE_xt_ty_ratio_is_one_l2296_229677

/-- Represents the shape described in the problem -/
structure Shape :=
  (total_squares : ℕ)
  (rectangle_squares : ℕ)
  (terrace_rows : ℕ)
  (terrace_squares_per_row : ℕ)

/-- Represents a line segment -/
structure LineSegment :=
  (length : ℝ)

/-- The problem setup -/
def problem_setup : Shape :=
  { total_squares := 12,
    rectangle_squares := 6,
    terrace_rows := 2,
    terrace_squares_per_row := 3 }

/-- The line RS that bisects the area horizontally -/
def RS : LineSegment :=
  { length := 6 }

/-- Theorem stating the ratio XT/TY = 1 -/
theorem xt_ty_ratio_is_one (shape : Shape) (rs : LineSegment) 
  (h1 : shape = problem_setup)
  (h2 : rs = RS)
  (h3 : rs.length = shape.total_squares / 2) :
  ∃ (xt ty : ℝ), xt = ty ∧ xt + ty = rs.length ∧ xt / ty = 1 :=
sorry

end NUMINAMATH_CALUDE_xt_ty_ratio_is_one_l2296_229677


namespace NUMINAMATH_CALUDE_thirteenth_service_same_as_first_l2296_229603

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- The number of months between services -/
def service_interval : Nat := 7

/-- The month of the first service -/
def first_service_month : Month := Month.March

/-- The number of the service we're interested in -/
def target_service_number : Nat := 13

/-- Calculates the number of months between two service numbers -/
def months_between_services (start service_number : Nat) : Nat :=
  service_interval * (service_number - start)

/-- Determines if two services occur in the same month -/
def same_month (start target : Nat) : Prop :=
  (months_between_services start target) % 12 = 0

theorem thirteenth_service_same_as_first :
  same_month 1 target_service_number := by sorry

end NUMINAMATH_CALUDE_thirteenth_service_same_as_first_l2296_229603


namespace NUMINAMATH_CALUDE_dividend_calculation_l2296_229673

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 38)
  (h2 : quotient = 19)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 729 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2296_229673


namespace NUMINAMATH_CALUDE_system_solutions_l2296_229680

def system_of_equations (x y z : ℝ) : Prop :=
  3 * x * y - 5 * y * z - x * z = 3 * y ∧
  x * y + y * z = -y ∧
  -5 * x * y + 4 * y * z + x * z = -4 * y

theorem system_solutions :
  (∀ x : ℝ, system_of_equations x 0 0) ∧
  (∀ z : ℝ, system_of_equations 0 0 z) ∧
  system_of_equations 2 (-1/3) (-3) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l2296_229680


namespace NUMINAMATH_CALUDE_marks_difference_l2296_229641

/-- Represents the marks of students a, b, c, d, and e. -/
structure Marks where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The conditions of the problem and the theorem to prove. -/
theorem marks_difference (m : Marks) : m.e - m.d = 3 :=
  by
  have h1 : m.a + m.b + m.c = 48 * 3 := by sorry
  have h2 : m.a + m.b + m.c + m.d = 47 * 4 := by sorry
  have h3 : m.b + m.c + m.d + m.e = 48 * 4 := by sorry
  have h4 : m.a = 43 := by sorry
  have h5 : m.e > m.d := by sorry
  sorry

end NUMINAMATH_CALUDE_marks_difference_l2296_229641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_l2296_229695

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 1 + a 9 = 10) :
  a 5 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_l2296_229695


namespace NUMINAMATH_CALUDE_equation_holds_iff_nonpositive_l2296_229636

theorem equation_holds_iff_nonpositive (a b : ℝ) : a = |b| → (a + b = 0 ↔ b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_iff_nonpositive_l2296_229636


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l2296_229644

-- Define the equation
def equation (x : ℝ) : Prop := 2 * Real.log x = 7 - 2 * x

-- Define the inequality
def inequality (n : ℤ) : Prop := (n : ℝ) - 2 < (n : ℝ)

theorem greatest_integer_solution :
  (∃ x : ℝ, equation x) →
  (∃ n : ℤ, inequality n ∧ ∀ m : ℤ, inequality m → m ≤ n) ∧
  (∀ n : ℤ, inequality n → n ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l2296_229644


namespace NUMINAMATH_CALUDE_unique_arrangements_of_zeros_and_ones_l2296_229611

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def permutations (n : ℕ) : ℕ := factorial n

def combinations (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem unique_arrangements_of_zeros_and_ones : 
  let total_digits : ℕ := 8
  let zeros : ℕ := 4
  let ones : ℕ := 4
  permutations total_digits / (permutations zeros * permutations ones) = 70 := by
  sorry

end NUMINAMATH_CALUDE_unique_arrangements_of_zeros_and_ones_l2296_229611


namespace NUMINAMATH_CALUDE_steves_return_speed_l2296_229659

def one_way_distance : ℝ := 40
def total_travel_time : ℝ := 6

theorem steves_return_speed (v : ℝ) (h1 : v > 0) :
  (one_way_distance / v + one_way_distance / (2 * v) = total_travel_time) →
  2 * v = 20 := by
  sorry

end NUMINAMATH_CALUDE_steves_return_speed_l2296_229659


namespace NUMINAMATH_CALUDE_max_value_xy_8x_y_l2296_229652

theorem max_value_xy_8x_y (x y : ℝ) (h : x^2 + y^2 = 20) :
  x * y + 8 * x + y ≤ 42 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_8x_y_l2296_229652


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2296_229672

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 2*b = 8) (h2 : 2*a + b = 4) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2296_229672


namespace NUMINAMATH_CALUDE_probability_A_not_lose_l2296_229698

-- Define the probabilities
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Define the probability of A not losing
def prob_A_not_lose : ℝ := prob_A_win + prob_draw

-- Theorem statement
theorem probability_A_not_lose : prob_A_not_lose = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_not_lose_l2296_229698


namespace NUMINAMATH_CALUDE_triangle_area_l2296_229679

/-- Given a triangle ABC with angle A = 60°, side b = 4, and side a = 2√3,
    prove that its area is 2√3. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 →  -- 60° in radians
  b = 4 → 
  a = 2 * Real.sqrt 3 → 
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2296_229679


namespace NUMINAMATH_CALUDE_pen_cost_l2296_229691

theorem pen_cost (pen ink_refill pencil : ℝ) 
  (total_cost : pen + ink_refill + pencil = 2.35)
  (pen_ink_relation : pen = ink_refill + 1.50)
  (pencil_cost : pencil = 0.45) : 
  pen = 1.70 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l2296_229691


namespace NUMINAMATH_CALUDE_tank_filling_time_l2296_229692

theorem tank_filling_time (p q r s : ℚ) 
  (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  p + q + r + s = 1 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2296_229692


namespace NUMINAMATH_CALUDE_winning_percentage_approx_l2296_229653

/-- Represents the votes received by each candidate in an election -/
structure ElectionResults where
  candidates : Fin 3 → ℕ
  candidate1_votes : candidates 0 = 3000
  candidate2_votes : candidates 1 = 5000
  candidate3_votes : candidates 2 = 20000

/-- Calculates the total number of votes in the election -/
def totalVotes (results : ElectionResults) : ℕ :=
  (results.candidates 0) + (results.candidates 1) + (results.candidates 2)

/-- Finds the maximum number of votes received by any candidate -/
def maxVotes (results : ElectionResults) : ℕ :=
  max (results.candidates 0) (max (results.candidates 1) (results.candidates 2))

/-- Calculates the percentage of votes received by the winning candidate -/
def winningPercentage (results : ElectionResults) : ℚ :=
  (maxVotes results : ℚ) / (totalVotes results : ℚ) * 100

/-- Theorem stating that the winning percentage is approximately 71.43% -/
theorem winning_percentage_approx (results : ElectionResults) :
  ∃ ε > 0, abs (winningPercentage results - 71.43) < ε :=
sorry

end NUMINAMATH_CALUDE_winning_percentage_approx_l2296_229653


namespace NUMINAMATH_CALUDE_triangle_inequality_l2296_229694

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → a + c > b → 
  ¬(a = 5 ∧ b = 9 ∧ c = 4) :=
by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_triangle_inequality_l2296_229694


namespace NUMINAMATH_CALUDE_inverse_proportion_product_l2296_229696

/-- Theorem: For points A(x₁, y₁) and B(x₂, y₂) on the graph of y = -3/x, 
    if x₁ * x₂ = 2, then y₁ * y₂ = 9/2 -/
theorem inverse_proportion_product (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = -3 / x₁) 
    (h2 : y₂ = -3 / x₂) 
    (h3 : x₁ * x₂ = 2) : 
  y₁ * y₂ = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_product_l2296_229696


namespace NUMINAMATH_CALUDE_f_sin_A_lt_f_cos_B_l2296_229600

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) + f x = 0

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem f_sin_A_lt_f_cos_B
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_periodic : is_periodic_2 f)
  (h_increasing : is_increasing_on f 3 4)
  (A B : ℝ)
  (h_acute_A : 0 < A ∧ A < Real.pi / 2)
  (h_acute_B : 0 < B ∧ B < Real.pi / 2) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_f_sin_A_lt_f_cos_B_l2296_229600


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l2296_229650

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h1 : ∀ x, (deriv (f a)) x = (deriv (f a)) (-x))  -- f' is an odd function
  (h2 : ∃ x, (deriv (f a)) x = 3/2)  -- There exists a point with slope 3/2
  : ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log ((3 + Real.sqrt 17) / 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l2296_229650


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l2296_229602

theorem min_distance_curve_line (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) :
  ∃ (x y : ℝ), ∀ (a' b' c' d' : ℝ),
    Real.log (b' + 1) + a' - 3 * b' = 0 →
    2 * d' - c' + Real.sqrt 5 = 0 →
    (a' - c')^2 + (b' - d')^2 ≥ 1 ∧
    (x - y)^2 + (0 - Real.sqrt 5)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l2296_229602


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2296_229606

def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2296_229606


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l2296_229676

theorem triangle_radius_inequality (a b c R r : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hR : 0 < R) (hr : 0 < r)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circum : 4 * R * (a * b * c) = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b))
  (h_inradius : r * (a + b + c) = 2 * (a * b * c) / (a + b + c)) :
  1 / R^2 ≤ 1 / a^2 + 1 / b^2 + 1 / c^2 ∧ 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≤ 1 / (2 * r)^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_radius_inequality_l2296_229676


namespace NUMINAMATH_CALUDE_plane_line_relations_l2296_229669

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (intersects : Plane → Plane → Line → Prop)
variable (within : Line → Plane → Prop)
variable (ne : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem plane_line_relations
  (α β : Plane) (l m : Line)
  (h1 : intersects α β l)
  (h2 : within m α)
  (h3 : ne m l) :
  (parallel m β → parallel_lines m l) ∧
  (parallel_lines m l → parallel m β) ∧
  (perpendicular m β → perpendicular_lines m l) ∧
  ¬(perpendicular_lines m l → perpendicular m β) :=
by sorry

end NUMINAMATH_CALUDE_plane_line_relations_l2296_229669


namespace NUMINAMATH_CALUDE_probability_at_least_one_vowel_l2296_229630

def set1 : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'o', 'p'}

def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

def isVowel (c : Char) : Bool := c ∈ vowels

theorem probability_at_least_one_vowel :
  let prob_no_vowel_set1 := (set1.filter (λ c => ¬isVowel c)).card / set1.card
  let prob_no_vowel_set2 := (set2.filter (λ c => ¬isVowel c)).card / set2.card
  1 - (prob_no_vowel_set1 * prob_no_vowel_set2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_vowel_l2296_229630


namespace NUMINAMATH_CALUDE_cashier_error_l2296_229639

theorem cashier_error : ¬∃ (x y : ℕ), 9 * x + 15 * y = 485 := by
  sorry

end NUMINAMATH_CALUDE_cashier_error_l2296_229639


namespace NUMINAMATH_CALUDE_volume_of_P₃_l2296_229651

/-- Represents a polyhedron in the sequence -/
structure Polyhedron where
  index : ℕ
  volume : ℚ

/-- Constructs the next polyhedron in the sequence -/
def next_polyhedron (P : Polyhedron) : Polyhedron :=
  { index := P.index + 1,
    volume := P.volume + (3/2)^P.index }

/-- The initial regular tetrahedron -/
def P₀ : Polyhedron :=
  { index := 0,
    volume := 1 }

/-- Generates the nth polyhedron in the sequence -/
def generate_polyhedron (n : ℕ) : Polyhedron :=
  match n with
  | 0 => P₀
  | n + 1 => next_polyhedron (generate_polyhedron n)

/-- The theorem to be proved -/
theorem volume_of_P₃ :
  (generate_polyhedron 3).volume = 23/4 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_P₃_l2296_229651


namespace NUMINAMATH_CALUDE_equation_solutions_l2296_229668

theorem equation_solutions :
  (∀ x : ℝ, (2*x - 1)^2 - 25 = 0 ↔ x = 3 ∨ x = -2) ∧
  (∀ x : ℝ, (1/3) * (x + 3)^3 - 9 = 0 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2296_229668


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l2296_229635

/-- The number of ways to place balls in boxes under different conditions -/
theorem ball_placement_theorem :
  let n : ℕ := 4  -- number of balls and boxes
  -- 1. Distinct balls, exactly one empty box
  let distinct_one_empty : ℕ := n * (n - 1) * (n - 2) * 6
  -- 2. Identical balls, exactly one empty box
  let identical_one_empty : ℕ := n * (n - 1)
  -- 3. Distinct balls, empty boxes allowed
  let distinct_empty_allowed : ℕ := n^n
  -- 4. Identical balls, empty boxes allowed
  let identical_empty_allowed : ℕ := 
    1 + n * (n - 1) + (n * (n - 1) / 2) + n * (n - 1) / 2 + n
  ∀ (n : ℕ), n = 4 →
    (distinct_one_empty = 144) ∧
    (identical_one_empty = 12) ∧
    (distinct_empty_allowed = 256) ∧
    (identical_empty_allowed = 35) := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l2296_229635


namespace NUMINAMATH_CALUDE_nathan_gave_six_apples_l2296_229619

/-- The number of apples Nathan gave to Annie -/
def apples_from_nathan (initial_apples final_apples : ℕ) : ℕ :=
  final_apples - initial_apples

theorem nathan_gave_six_apples :
  apples_from_nathan 6 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nathan_gave_six_apples_l2296_229619


namespace NUMINAMATH_CALUDE_value_of_b_l2296_229607

theorem value_of_b (b : ℚ) (h : b - b/4 = 5/2) : b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2296_229607


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2296_229661

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  b = 2 * a * Real.sin B →
  A = π / 6 ∨ A = 5 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2296_229661


namespace NUMINAMATH_CALUDE_sum_calculation_l2296_229645

theorem sum_calculation : 3 * 198 + 2 * 198 + 198 + 197 = 1385 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l2296_229645


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l2296_229640

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0
def q (x : ℝ) : Prop := ∃ m : ℝ, m ∈ Set.Ioo 1 2 ∧ x = (1/2)^(m-1)

-- Theorem for part (1)
theorem range_of_x_when_a_is_quarter (x : ℝ) :
  p x (1/4) ∧ q x → x ∈ Set.Ioo (1/2) (3/4) :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_q_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) →
  a ∈ Set.Icc (1/3) (1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l2296_229640


namespace NUMINAMATH_CALUDE_smallest_perimeter_of_special_triangle_l2296_229628

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ 
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1

/-- The main theorem -/
theorem smallest_perimeter_of_special_triangle :
  ∀ a b c : ℕ,
    areConsecutiveOddPrimes a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 41 :=
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_of_special_triangle_l2296_229628


namespace NUMINAMATH_CALUDE_investment_difference_l2296_229648

/-- Represents the final value of an investment given its initial value and growth factor -/
def final_value (initial : ℝ) (growth : ℝ) : ℝ := initial * growth

/-- Theorem: Given the initial investments and their changes in value, 
    the difference between Jackson's final investment value and 
    the combined final investment values of Brandon and Meagan is $850 -/
theorem investment_difference : 
  let jackson_initial := (500 : ℝ)
  let brandon_initial := (500 : ℝ)
  let meagan_initial := (700 : ℝ)
  let jackson_growth := (4 : ℝ)
  let brandon_growth := (0.2 : ℝ)
  let meagan_growth := (1.5 : ℝ)
  final_value jackson_initial jackson_growth - 
  (final_value brandon_initial brandon_growth + final_value meagan_initial meagan_growth) = 850 := by
  sorry


end NUMINAMATH_CALUDE_investment_difference_l2296_229648


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2296_229697

theorem cos_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (-(2 / tanα) = 8/3) →  -- slope of the line 2x + (tanα)y + 1 = 0 is 8/3
  cosα = -4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2296_229697


namespace NUMINAMATH_CALUDE_andromeda_distance_scientific_notation_l2296_229674

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The distance of the Andromeda galaxy from the Milky Way in light-years -/
def andromeda_distance : ℝ := 2500000

theorem andromeda_distance_scientific_notation :
  to_scientific_notation andromeda_distance = ScientificNotation.mk 2.5 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_andromeda_distance_scientific_notation_l2296_229674


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2296_229670

def M : ℕ := 36 * 36 * 65 * 280

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M) * 254 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2296_229670


namespace NUMINAMATH_CALUDE_sequence_eventually_constant_l2296_229623

/-- A sequence of non-negative integers satisfying the given conditions -/
def Sequence (m : ℕ+) := { a : ℕ → ℕ // 
  a 0 = m ∧ 
  (∀ n : ℕ, n ≥ 1 → a n ≤ n) ∧
  (∀ n : ℕ+, (n : ℕ) ∣ (Finset.range n).sum (λ i => a i)) }

/-- The main theorem -/
theorem sequence_eventually_constant (m : ℕ+) (a : Sequence m) : 
  ∃ M : ℕ, ∀ n ≥ M, a.val n = a.val M :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_constant_l2296_229623


namespace NUMINAMATH_CALUDE_sphere_wedge_volume_l2296_229656

theorem sphere_wedge_volume (c : ℝ) (h1 : c = 16 * Real.pi) : 
  let r := c / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * r^3
  let wedge_volume := sphere_volume / 8
  wedge_volume = (256 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_wedge_volume_l2296_229656


namespace NUMINAMATH_CALUDE_perimeter_of_PQRS_l2296_229660

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the perimeter function
def perimeter (quad : Quadrilateral) : ℝ := sorry

-- Define the properties of the quadrilateral
def is_right_angle_at_Q (quad : Quadrilateral) : Prop := sorry
def PR_perpendicular_to_RS (quad : Quadrilateral) : Prop := sorry
def PQ_length (quad : Quadrilateral) : ℝ := sorry
def QR_length (quad : Quadrilateral) : ℝ := sorry
def RS_length (quad : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem perimeter_of_PQRS (quad : Quadrilateral) :
  is_right_angle_at_Q quad →
  PR_perpendicular_to_RS quad →
  PQ_length quad = 24 →
  QR_length quad = 28 →
  RS_length quad = 16 →
  perimeter quad = 68 + Real.sqrt 1616 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_PQRS_l2296_229660


namespace NUMINAMATH_CALUDE_symmetry_complex_plane_l2296_229681

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal -/
def symmetric_to_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem symmetry_complex_plane (z₁ z₂ : ℂ) :
  symmetric_to_imaginary_axis z₁ z₂ → z₁ = 1 + I → z₂ = -1 + I := by
  sorry

#check symmetry_complex_plane

end NUMINAMATH_CALUDE_symmetry_complex_plane_l2296_229681


namespace NUMINAMATH_CALUDE_gcf_of_60_and_75_l2296_229663

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_75_l2296_229663


namespace NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l2296_229632

theorem twelve_point_zero_six_million_scientific_notation :
  (12.06 : ℝ) * 1000000 = 1.206 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l2296_229632


namespace NUMINAMATH_CALUDE_z_extrema_l2296_229647

-- Define the function z(x,y)
def z (x y : ℝ) : ℝ := 2 * x^3 - 6 * x * y + 3 * y^2

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.2 ≤ 2 ∧ p.2 ≤ p.1^2 / 2}

-- State the theorem
theorem z_extrema :
  ∃ (max min : ℝ), max = 12 ∧ min = -1 ∧
  (∀ p ∈ R, z p.1 p.2 ≤ max) ∧
  (∀ p ∈ R, z p.1 p.2 ≥ min) ∧
  (∃ p ∈ R, z p.1 p.2 = max) ∧
  (∃ p ∈ R, z p.1 p.2 = min) :=
sorry

end NUMINAMATH_CALUDE_z_extrema_l2296_229647


namespace NUMINAMATH_CALUDE_larger_number_proof_l2296_229646

theorem larger_number_proof (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2296_229646


namespace NUMINAMATH_CALUDE_first_quadrant_trig_positivity_l2296_229631

theorem first_quadrant_trig_positivity (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  0 < Real.sin (2 * α) ∧ 0 < Real.tan (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_first_quadrant_trig_positivity_l2296_229631


namespace NUMINAMATH_CALUDE_road_trip_ratio_l2296_229662

/-- Road trip problem -/
theorem road_trip_ratio : 
  ∀ (total michelle_dist katie_dist tracy_dist : ℕ),
  total = 1000 →
  michelle_dist = 294 →
  michelle_dist = 3 * katie_dist →
  tracy_dist = total - michelle_dist - katie_dist →
  (tracy_dist - 20) / michelle_dist = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_road_trip_ratio_l2296_229662


namespace NUMINAMATH_CALUDE_mean_of_two_numbers_l2296_229689

def numbers : List ℕ := [1871, 1997, 2023, 2029, 2113, 2125, 2137]

def sum_of_all : ℕ := numbers.sum

def mean_of_five : ℕ := 2100

def sum_of_five : ℕ := 5 * mean_of_five

def sum_of_two : ℕ := sum_of_all - sum_of_five

theorem mean_of_two_numbers : (sum_of_two : ℚ) / 2 = 1397.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_two_numbers_l2296_229689
