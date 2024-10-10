import Mathlib

namespace magnitude_of_vector_sum_l2978_297851

/-- The magnitude of the sum of two vectors given specific conditions -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) :
  a = (1, 0) →
  ‖b‖ = Real.sqrt 2 →
  a • b = 1 →
  ‖2 • a + b‖ = Real.sqrt 10 := by
  sorry

end magnitude_of_vector_sum_l2978_297851


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_holds_l2978_297805

/-- An isosceles triangle with two sides of length 12 and a third side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 12 ∧ b = 12 ∧ c = 17) →  -- Two sides are 12, third side is 17
    (a = b)  →                    -- Isosceles triangle condition
    (a + b + c = 41)              -- Perimeter is 41

/-- The theorem holds for the given triangle. -/
theorem isosceles_triangle_perimeter_holds : isosceles_triangle_perimeter 12 12 17 := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_holds_l2978_297805


namespace same_number_of_friends_l2978_297843

theorem same_number_of_friends (n : ℕ) (h : n > 0) :
  ∃ (f : Fin n → Fin n),
    ∃ (i j : Fin n), i ≠ j ∧ f i = f j :=
by
  sorry

end same_number_of_friends_l2978_297843


namespace boxes_per_case_l2978_297860

theorem boxes_per_case (total_boxes : ℕ) (num_cases : ℕ) (h1 : total_boxes = 24) (h2 : num_cases = 3) :
  total_boxes / num_cases = 8 := by
sorry

end boxes_per_case_l2978_297860


namespace points_form_parabola_l2978_297871

-- Define the set of points (x, y) parametrically
def S : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = Real.cos t ^ 2 ∧ p.2 = Real.sin (2 * t)}

-- Define a parabola in general form
def IsParabola (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    ∀ p ∈ S, a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 = 0

-- Theorem statement
theorem points_form_parabola : IsParabola S := by
  sorry

end points_form_parabola_l2978_297871


namespace expression_simplification_and_evaluation_l2978_297890

theorem expression_simplification_and_evaluation (a b : ℝ) 
  (h1 : a = 1) (h2 : b = -1) : 
  (2*a^2*b - 2*a*b^2 - b^3) / b - (a + b)*(a - b) = 3 := by
  sorry

end expression_simplification_and_evaluation_l2978_297890


namespace women_married_long_service_fraction_l2978_297845

theorem women_married_long_service_fraction 
  (total_employees : ℕ) 
  (women_percentage : ℚ)
  (married_percentage : ℚ)
  (single_men_fraction : ℚ)
  (married_long_service_women_percentage : ℚ)
  (h1 : women_percentage = 76 / 100)
  (h2 : married_percentage = 60 / 100)
  (h3 : single_men_fraction = 2 / 3)
  (h4 : married_long_service_women_percentage = 70 / 100)
  : ℚ :=
by
  sorry

#check women_married_long_service_fraction

end women_married_long_service_fraction_l2978_297845


namespace all_or_none_triangular_l2978_297820

/-- A polynomial of degree 4 -/
structure Polynomial4 where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ

/-- Evaluate the polynomial at a given x -/
def eval (poly : Polynomial4) (x : ℝ) : ℝ :=
  x^4 + poly.p * x^3 + poly.q * x^2 + poly.r * x + poly.s

/-- Represents four points on a horizontal line intersecting the curve -/
structure FourPoints where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  h₁ : x₁ < x₂
  h₂ : x₂ < x₃
  h₃ : x₃ < x₄

/-- Check if three lengths can form a triangle -/
def isTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of four points is triangular -/
def isTriangular (pts : FourPoints) : Prop :=
  isTriangle (pts.x₂ - pts.x₁) (pts.x₃ - pts.x₁) (pts.x₄ - pts.x₁)

/-- The main theorem -/
theorem all_or_none_triangular (poly : Polynomial4) :
  (∀ y : ℝ, ∀ pts : FourPoints, eval poly pts.x₁ = y ∧ eval poly pts.x₂ = y ∧
    eval poly pts.x₃ = y ∧ eval poly pts.x₄ = y → isTriangular pts) ∨
  (∀ y : ℝ, ∀ pts : FourPoints, eval poly pts.x₁ = y ∧ eval poly pts.x₂ = y ∧
    eval poly pts.x₃ = y ∧ eval poly pts.x₄ = y → ¬isTriangular pts) :=
sorry

end all_or_none_triangular_l2978_297820


namespace train_length_train_length_alt_l2978_297822

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 9 → speed * time * (5 / 18) = 180 := by
  sorry

/-- Alternative formulation using more basic definitions -/
theorem train_length_alt (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 72 → time_s = 9 → 
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 180 := by
  sorry

end train_length_train_length_alt_l2978_297822


namespace nancy_eats_indian_food_three_times_a_week_l2978_297826

/-- Represents the number of times Nancy eats Indian food per week -/
def indian_food_times : ℕ := sorry

/-- Represents the number of times Nancy eats Mexican food per week -/
def mexican_food_times : ℕ := 2

/-- Represents the number of antacids Nancy takes when eating Indian food -/
def indian_food_antacids : ℕ := 3

/-- Represents the number of antacids Nancy takes when eating Mexican food -/
def mexican_food_antacids : ℕ := 2

/-- Represents the number of antacids Nancy takes on other days -/
def other_days_antacids : ℕ := 1

/-- Represents the total number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the number of weeks in a month (approximation) -/
def weeks_in_month : ℕ := 4

/-- Represents the total number of antacids Nancy takes per month -/
def total_antacids_per_month : ℕ := 60

/-- Theorem stating that Nancy eats Indian food 3 times a week -/
theorem nancy_eats_indian_food_three_times_a_week :
  indian_food_times = 3 :=
by sorry

end nancy_eats_indian_food_three_times_a_week_l2978_297826


namespace polynomial_roots_l2978_297889

theorem polynomial_roots : 
  let f : ℝ → ℝ := λ x => 3*x^4 + 2*x^3 - 7*x^2 + 2*x + 3
  let root1 : ℝ := ((-1 + 2*Real.sqrt 10)/3 + Real.sqrt (((-1 + 2*Real.sqrt 10)/3)^2 - 4))/2
  let root2 : ℝ := ((-1 + 2*Real.sqrt 10)/3 - Real.sqrt (((-1 + 2*Real.sqrt 10)/3)^2 - 4))/2
  let root3 : ℝ := ((-1 - 2*Real.sqrt 10)/3 + Real.sqrt (((-1 - 2*Real.sqrt 10)/3)^2 - 4))/2
  let root4 : ℝ := ((-1 - 2*Real.sqrt 10)/3 - Real.sqrt (((-1 - 2*Real.sqrt 10)/3)^2 - 4))/2
  (f root1 = 0) ∧ (f root2 = 0) ∧ (f root3 = 0) ∧ (f root4 = 0) ∧
  (∀ x : ℝ, f x = 0 → (x = root1 ∨ x = root2 ∨ x = root3 ∨ x = root4)) :=
by sorry

end polynomial_roots_l2978_297889


namespace ball_selection_count_l2978_297853

/-- Represents the set of colors available for the balls -/
inductive Color
| Red
| Yellow
| Blue

/-- Represents the set of letters used to mark the balls -/
inductive Letter
| A
| B
| C
| D
| E

/-- The total number of balls for each color -/
def ballsPerColor : Nat := 5

/-- The total number of colors -/
def numColors : Nat := 3

/-- The number of balls to be selected -/
def ballsToSelect : Nat := 5

/-- Calculates the number of ways to select the balls -/
def selectBalls : Nat := numColors ^ ballsToSelect

theorem ball_selection_count :
  selectBalls = 243 :=
sorry

end ball_selection_count_l2978_297853


namespace milk_purchase_theorem_l2978_297800

/-- The cost of one bag of milk in yuan -/
def milk_cost : ℕ := 3

/-- The number of bags paid for in the offer -/
def offer_paid : ℕ := 5

/-- The total number of bags received in the offer -/
def offer_total : ℕ := 6

/-- The amount of money mom has in yuan -/
def mom_money : ℕ := 20

/-- The maximum number of bags mom can buy -/
def max_bags : ℕ := 7

/-- The amount of money left after buying the maximum number of bags -/
def money_left : ℕ := 2

theorem milk_purchase_theorem :
  milk_cost * (offer_paid * (max_bags / offer_total) + max_bags % offer_total) ≤ mom_money ∧
  mom_money - milk_cost * (offer_paid * (max_bags / offer_total) + max_bags % offer_total) = money_left :=
by sorry

end milk_purchase_theorem_l2978_297800


namespace complement_of_A_l2978_297899

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

theorem complement_of_A : (U \ A) = {0} := by sorry

end complement_of_A_l2978_297899


namespace quadratic_equation_solution_l2978_297840

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) :=
by
  -- The proof goes here
  sorry

end quadratic_equation_solution_l2978_297840


namespace work_time_difference_l2978_297818

def monday_minutes : ℕ := 450
def wednesday_minutes : ℕ := 300

def tuesday_minutes : ℕ := monday_minutes / 2

theorem work_time_difference : wednesday_minutes - tuesday_minutes = 75 := by
  sorry

end work_time_difference_l2978_297818


namespace boys_percentage_of_school_l2978_297885

theorem boys_percentage_of_school (total_students : ℕ) (boys_representation : ℕ) 
  (h1 : total_students = 180)
  (h2 : boys_representation = 162)
  (h3 : boys_representation = (180 / 100) * (boys_percentage / 100 * total_students)) :
  boys_percentage = 50 := by
  sorry

end boys_percentage_of_school_l2978_297885


namespace segment_length_problem_l2978_297847

/-- Given a line segment AD of length 56 units, divided into three segments AB, BC, and CD,
    where AB : BC = 1 : 2 and BC : CD = 6 : 5, the length of AB is 12 units. -/
theorem segment_length_problem (AB BC CD : ℝ) : 
  AB + BC + CD = 56 → 
  AB / BC = 1 / 2 → 
  BC / CD = 6 / 5 → 
  AB = 12 := by
sorry

end segment_length_problem_l2978_297847


namespace arithmetic_calculations_l2978_297897

theorem arithmetic_calculations :
  (456 - 9 * 8 = 384) ∧
  (387 + 126 - 212 = 301) ∧
  (533 - (108 + 209) = 216) ∧
  ((746 - 710) / 6 = 6) := by
  sorry

end arithmetic_calculations_l2978_297897


namespace smallest_multiple_of_45_and_75_not_20_l2978_297886

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ n : ℕ, n > 0 ∧ 
    is_multiple n 45 ∧ 
    is_multiple n 75 ∧ 
    ¬is_multiple n 20 ∧
    ∀ m : ℕ, m > 0 → 
      is_multiple m 45 → 
      is_multiple m 75 → 
      ¬is_multiple m 20 → 
      n ≤ m ∧
  n = 225 :=
sorry

end smallest_multiple_of_45_and_75_not_20_l2978_297886


namespace equation_has_real_roots_l2978_297894

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end equation_has_real_roots_l2978_297894


namespace sandwich_filler_percentage_l2978_297829

/-- Given a sandwich with a total weight of 180 grams and filler weight of 45 grams,
    prove that the percentage of the sandwich that is not filler is 75%. -/
theorem sandwich_filler_percentage (total_weight filler_weight : ℝ) 
    (h1 : total_weight = 180)
    (h2 : filler_weight = 45) :
    (total_weight - filler_weight) / total_weight = 0.75 := by
  sorry

end sandwich_filler_percentage_l2978_297829


namespace prob_green_ball_is_13_28_l2978_297898

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers A, B, C, and D -/
def containers : List Container := [
  ⟨4, 6⟩,  -- Container A
  ⟨8, 6⟩,  -- Container B
  ⟨8, 6⟩,  -- Container C
  ⟨3, 7⟩   -- Container D
]

/-- The number of containers -/
def numContainers : ℕ := containers.length

/-- The probability of selecting a green ball -/
def probGreenBall : ℚ := 
  (1 / numContainers) * (containers.map greenProbability).sum

theorem prob_green_ball_is_13_28 : probGreenBall = 13/28 := by
  sorry

end prob_green_ball_is_13_28_l2978_297898


namespace smallest_number_with_unique_digits_divisible_by_990_l2978_297841

theorem smallest_number_with_unique_digits_divisible_by_990 : ∃ (n : ℕ), 
  (n = 1234758690) ∧ 
  (∀ m : ℕ, m < n → ¬(∀ d : Fin 10, (m.digits 10).count d = 1)) ∧
  (∀ d : Fin 10, (n.digits 10).count d = 1) ∧
  (n % 990 = 0) := by
  sorry

end smallest_number_with_unique_digits_divisible_by_990_l2978_297841


namespace age_problem_l2978_297859

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The sum of their ages is 72
    Prove that b is 28 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 72) : 
  b = 28 := by
  sorry

end age_problem_l2978_297859


namespace esteban_exercise_days_l2978_297807

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of minutes in an hour -/
def minutesInHour : ℕ := 60

/-- Represents Natasha's daily exercise time in minutes -/
def natashasDailyExercise : ℕ := 30

/-- Represents Esteban's daily exercise time in minutes -/
def estebansDailyExercise : ℕ := 10

/-- Represents the total exercise time of Natasha and Esteban in hours -/
def totalExerciseTime : ℕ := 5

/-- Theorem stating that Esteban exercised for 9 days -/
theorem esteban_exercise_days : 
  ∃ (estebanDays : ℕ), 
    estebanDays * estebansDailyExercise + 
    daysInWeek * natashasDailyExercise = 
    totalExerciseTime * minutesInHour ∧ 
    estebanDays = 9 := by
  sorry

end esteban_exercise_days_l2978_297807


namespace sum_of_roots_l2978_297888

theorem sum_of_roots (x : ℝ) : (x + 2) * (x - 3) = 16 → ∃ y : ℝ, (y + 2) * (y - 3) = 16 ∧ x + y = 1 := by
  sorry

end sum_of_roots_l2978_297888


namespace correct_divisor_l2978_297825

theorem correct_divisor : ∃ (X : ℕ) (incorrect_divisor correct_divisor : ℕ),
  incorrect_divisor = 87 ∧
  X / incorrect_divisor = 24 ∧
  X / correct_divisor = 58 ∧
  correct_divisor = 36 := by
  sorry

end correct_divisor_l2978_297825


namespace division_example_exists_l2978_297881

theorem division_example_exists : ∃ (D d q : ℕ+), 
  (D : ℚ) / (d : ℚ) = q ∧ 
  (q : ℚ) = (D : ℚ) / 5 ∧ 
  (q : ℚ) = 7 * (d : ℚ) := by
  sorry

end division_example_exists_l2978_297881


namespace logarithm_sum_simplification_l2978_297836

theorem logarithm_sum_simplification :
  let expr := (1 / (Real.log 3 / Real.log 12 + 1)) + 
              (1 / (Real.log 2 / Real.log 8 + 1)) + 
              (1 / (Real.log 3 / Real.log 9 + 1))
  expr = (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) := by
  sorry

end logarithm_sum_simplification_l2978_297836


namespace factors_of_36_l2978_297802

theorem factors_of_36 : Nat.card (Nat.divisors 36) = 9 := by
  sorry

end factors_of_36_l2978_297802


namespace purchase_cost_l2978_297814

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 1

/-- The number of sandwiches to purchase -/
def num_sandwiches : ℕ := 6

/-- The number of sodas to purchase -/
def num_sodas : ℕ := 10

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem purchase_cost : total_cost = 34 := by
  sorry

end purchase_cost_l2978_297814


namespace range_of_k_for_trigonometric_equation_l2978_297803

theorem range_of_k_for_trigonometric_equation :
  ∀ k : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 (π/2) ∧ 
    Real.sqrt 3 * Real.sin (2*x) + Real.cos (2*x) = k + 1) ↔ 
  k ∈ Set.Icc (-2) 1 := by
sorry

end range_of_k_for_trigonometric_equation_l2978_297803


namespace min_x_plus_y_l2978_297808

theorem min_x_plus_y (x y : ℝ) (h1 : x > 1) (h2 : x * y = 2 * x + y + 2) :
  x + y ≥ 7 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 1 ∧ x₀ * y₀ = 2 * x₀ + y₀ + 2 ∧ x₀ + y₀ = 7 := by
  sorry

end min_x_plus_y_l2978_297808


namespace mans_age_twice_sons_l2978_297827

/-- Proves that the number of years until a man's age is twice his son's age is 2 -/
theorem mans_age_twice_sons (
  man_age_difference : ℕ → ℕ → ℕ)
  (son_current_age : ℕ)
  (h1 : man_age_difference son_current_age son_current_age = 34)
  (h2 : son_current_age = 32)
  : ∃ (years : ℕ), years = 2 ∧
    man_age_difference (son_current_age + years) (son_current_age + years) + years =
    2 * (son_current_age + years) :=
by sorry

end mans_age_twice_sons_l2978_297827


namespace largest_fraction_l2978_297819

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 5/9, 4/11, 3/8]
  ∀ x ∈ fractions, (5:ℚ)/9 ≥ x := by
  sorry

end largest_fraction_l2978_297819


namespace circumcenter_property_l2978_297864

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is outside a plane -/
def isOutside (p : Point3D) (plane : Plane) : Prop := sorry

/-- Check if a line is perpendicular to a plane -/
def isPerpendicular (p1 p2 : Point3D) (plane : Plane) : Prop := sorry

/-- Check if a point is the foot of the perpendicular from another point to a plane -/
def isFootOfPerpendicular (o p : Point3D) (plane : Plane) : Prop := sorry

/-- Check if three distances are equal -/
def areDistancesEqual (p a b c : Point3D) : Prop := sorry

/-- Check if a point is the circumcenter of a triangle -/
def isCircumcenter (o : Point3D) (t : Triangle3D) : Prop := sorry

theorem circumcenter_property (P O : Point3D) (ABC : Triangle3D) (plane : Plane) :
  isOutside P plane →
  isPerpendicular P O plane →
  isFootOfPerpendicular O P plane →
  areDistancesEqual P ABC.A ABC.B ABC.C →
  isCircumcenter O ABC := by sorry

end circumcenter_property_l2978_297864


namespace age_difference_l2978_297876

/-- Given that the total age of a and b is 17 years more than the total age of b and c,
    prove that c is 17 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 17) : a = c + 17 := by
  sorry

end age_difference_l2978_297876


namespace complex_division_l2978_297878

/-- Given a complex number z = 1 + ai where a is a positive real number and |z| = √10,
    prove that z / (1 - 2i) = -1 + i -/
theorem complex_division (a : ℝ) (z : ℂ) (h1 : a > 0) (h2 : z = 1 + a * Complex.I) 
    (h3 : Complex.abs z = Real.sqrt 10) : 
  z / (1 - 2 * Complex.I) = -1 + Complex.I := by
  sorry

end complex_division_l2978_297878


namespace triangle_inequality_generalization_l2978_297884

theorem triangle_inequality_generalization (x y z : ℝ) :
  (|x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt 2 * Real.sqrt (x^2 + y^2 + z^2)) ∧
  ((0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) → |x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt (x^2 + y^2 + z^2)) := by
  sorry

end triangle_inequality_generalization_l2978_297884


namespace sum_to_n_equals_91_l2978_297862

theorem sum_to_n_equals_91 : ∃ n : ℕ, n * (n + 1) / 2 = 91 ∧ n = 13 := by
  sorry

end sum_to_n_equals_91_l2978_297862


namespace problem_solving_probability_l2978_297848

theorem problem_solving_probability (p_a p_either : ℝ) (h1 : p_a = 0.7) (h2 : p_either = 0.94) :
  ∃ p_b : ℝ, p_b = 0.8 ∧ p_either = 1 - (1 - p_a) * (1 - p_b) := by
  sorry

end problem_solving_probability_l2978_297848


namespace sequence_element_proof_l2978_297863

theorem sequence_element_proof :
  (∃ n : ℕ+, n^2 + 2*n = 63) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 10) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 18) ∧
  (¬ ∃ n : ℕ+, n^2 + 2*n = 26) :=
by sorry

end sequence_element_proof_l2978_297863


namespace complex_modulus_problem_l2978_297823

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l2978_297823


namespace triangle_properties_l2978_297880

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, -1)

-- Define the equation of angle bisector CD
def angle_bisector_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the equation of the perpendicular bisector of AB
def perp_bisector_eq (x y : ℝ) : Prop := 4*x + 6*y - 3 = 0

-- Define vertex C
def C : ℝ × ℝ := (-1, 2)

theorem triangle_properties :
  (∀ x y : ℝ, angle_bisector_eq x y ↔ x + y - 1 = 0) ∧
  (∀ x y : ℝ, perp_bisector_eq x y ↔ 4*x + 6*y - 3 = 0) ∧
  C = (-1, 2) :=
sorry

end triangle_properties_l2978_297880


namespace printer_z_time_l2978_297844

/-- Given printers X, Y, and Z with the following properties:
  - The ratio of time for X alone to Y and Z together is 2.25
  - X can do the job in 15 hours
  - Y can do the job in 10 hours
Prove that Z takes 20 hours to do the job alone. -/
theorem printer_z_time (tx ty tz : ℝ) : 
  tx = 15 → 
  ty = 10 → 
  tx = 2.25 * (1 / (1 / ty + 1 / tz)) → 
  tz = 20 := by
sorry

end printer_z_time_l2978_297844


namespace complement_N_Nstar_is_finite_l2978_297861

def complement_N_Nstar : Set ℕ := {0}

theorem complement_N_Nstar_is_finite :
  Set.Finite complement_N_Nstar :=
sorry

end complement_N_Nstar_is_finite_l2978_297861


namespace six_people_charity_arrangements_l2978_297869

/-- The number of ways to distribute n people into 2 charity activities,
    with each activity accommodating no more than 4 people -/
def charity_arrangements (n : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating that there are 50 ways to distribute 6 people
    into 2 charity activities with the given constraints -/
theorem six_people_charity_arrangements :
  charity_arrangements 6 = 50 := by
  sorry

end six_people_charity_arrangements_l2978_297869


namespace correct_seat_notation_l2978_297809

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (seatNum : ℕ)

/-- Defines the notation for a seat -/
def seatNotation (s : Seat) : ℕ × ℕ := (s.row, s.seatNum)

theorem correct_seat_notation :
  let example_seat := Seat.mk 10 3
  let target_seat := Seat.mk 6 16
  (seatNotation example_seat = (10, 3)) →
  (seatNotation target_seat = (6, 16)) := by
  sorry

end correct_seat_notation_l2978_297809


namespace gavin_green_shirts_l2978_297893

/-- The number of green shirts Gavin has -/
def num_green_shirts (total_shirts blue_shirts : ℕ) : ℕ :=
  total_shirts - blue_shirts

/-- Theorem stating that Gavin has 17 green shirts -/
theorem gavin_green_shirts : 
  num_green_shirts 23 6 = 17 := by
  sorry

end gavin_green_shirts_l2978_297893


namespace essay_pages_filled_l2978_297877

theorem essay_pages_filled (johnny_words madeline_words timothy_words words_per_page : ℕ) 
  (h1 : johnny_words = 150)
  (h2 : madeline_words = 2 * johnny_words)
  (h3 : timothy_words = madeline_words + 30)
  (h4 : words_per_page = 260) : 
  (johnny_words + madeline_words + timothy_words) / words_per_page = 3 := by
  sorry

end essay_pages_filled_l2978_297877


namespace largest_sample_size_l2978_297883

def population : Nat := 36

theorem largest_sample_size (X : Nat) : 
  (X > 0 ∧ 
   population % X = 0 ∧ 
   population % (X + 1) ≠ 0 ∧ 
   ∀ Y : Nat, Y > X → (population % Y = 0 → population % (Y + 1) = 0)) → 
  X = 9 := by sorry

end largest_sample_size_l2978_297883


namespace largest_sum_simplification_l2978_297830

theorem largest_sum_simplification : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/7, 1/3 + 1/8]
  (∀ s ∈ sums, s ≤ 1/3 + 1/2) ∧ 
  (1/3 + 1/2 = 5/6) := by
  sorry

end largest_sum_simplification_l2978_297830


namespace termite_ridden_not_collapsing_l2978_297833

/-- Given that 1/3 of homes are termite-ridden and 4/7 of termite-ridden homes are collapsing,
    prove that 3/21 of homes are termite-ridden but not collapsing. -/
theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = total_homes / 3) 
  (h2 : collapsing = termite_ridden * 4 / 7) : 
  (termite_ridden - collapsing) = total_homes * 3 / 21 := by
  sorry

end termite_ridden_not_collapsing_l2978_297833


namespace best_and_most_stable_values_l2978_297855

/-- Represents a student's performance data -/
structure StudentPerformance where
  average : ℝ
  variance : ℝ

/-- The given data for students B, C, and D -/
def studentB : StudentPerformance := ⟨90, 12.5⟩
def studentC : StudentPerformance := ⟨91, 14.5⟩
def studentD : StudentPerformance := ⟨88, 11⟩

/-- Conditions for Student A to be the best-performing and most stable -/
def isBestAndMostStable (m n : ℝ) : Prop :=
  m > studentB.average ∧
  m > studentC.average ∧
  m > studentD.average ∧
  n < studentB.variance ∧
  n < studentC.variance ∧
  n < studentD.variance

/-- Theorem stating that m = 92 and n = 8.5 are the only values satisfying the conditions -/
theorem best_and_most_stable_values :
  ∀ m n : ℝ, isBestAndMostStable m n ↔ m = 92 ∧ n = 8.5 := by
  sorry

end best_and_most_stable_values_l2978_297855


namespace ellipse_minor_axis_length_l2978_297834

/-- Given an ellipse with minimum distance 5 and maximum distance 15 from a point on the ellipse to a focus, 
    the length of its minor axis is 10√3. -/
theorem ellipse_minor_axis_length (min_dist max_dist : ℝ) (h1 : min_dist = 5) (h2 : max_dist = 15) :
  let a := (max_dist + min_dist) / 2
  let c := (max_dist - min_dist) / 2
  let b := Real.sqrt (a^2 - c^2)
  2 * b = 10 * Real.sqrt 3 := by sorry

end ellipse_minor_axis_length_l2978_297834


namespace parabola_transformation_sum_l2978_297858

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a transformation of a quadratic function -/
inductive Transformation
  | Reflect
  | Translate (d : ℝ)

/-- Applies a transformation to a quadratic function -/
def applyTransformation (q : QuadraticFunction) (t : Transformation) : QuadraticFunction :=
  match t with
  | Transformation.Reflect => { a := q.a, b := -q.b, c := q.c }
  | Transformation.Translate d => { a := q.a, b := q.b - 2 * q.a * d, c := q.a * d^2 - q.b * d + q.c }

/-- Sums two quadratic functions -/
def sumQuadraticFunctions (q1 q2 : QuadraticFunction) : QuadraticFunction :=
  { a := q1.a + q2.a, b := q1.b + q2.b, c := q1.c + q2.c }

theorem parabola_transformation_sum (q : QuadraticFunction) :
  let f := applyTransformation (applyTransformation q Transformation.Reflect) (Transformation.Translate (-7))
  let g := applyTransformation q (Transformation.Translate 3)
  let sum := sumQuadraticFunctions f g
  sum.a = 2 * q.a ∧ sum.b = 8 * q.a - 2 * q.b ∧ sum.c = 58 * q.a - 4 * q.b + 2 * q.c :=
by sorry

end parabola_transformation_sum_l2978_297858


namespace sum_negative_implies_at_most_one_positive_l2978_297870

theorem sum_negative_implies_at_most_one_positive (a b : ℚ) :
  a + b < 0 → (0 < a ∧ 0 < b) → False := by sorry

end sum_negative_implies_at_most_one_positive_l2978_297870


namespace ellipse_properties_l2978_297866

-- Define the ellipse C
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the condition for equilateral triangle formed by foci and minor axis endpoint
def equilateralCondition (a b c : ℝ) : Prop :=
  a = 2*c ∧ b = Real.sqrt 3 * c

-- Define the tangency condition for the circle
def tangencyCondition (a b c : ℝ) : Prop :=
  |c + 2| / Real.sqrt 2 = (Real.sqrt 6 / 2) * b

-- Define the vector addition condition
def vectorAdditionCondition (A B M : ℝ × ℝ) (t : ℝ) : Prop :=
  A.1 + B.1 = t * M.1 ∧ A.2 + B.2 = t * M.2

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  ∀ (c : ℝ), equilateralCondition a b c →
  tangencyCondition a b c →
  (∀ (A B M : ℝ × ℝ) (t : ℝ),
    A ∈ ellipse a b h →
    B ∈ ellipse a b h →
    M ∈ ellipse a b h →
    (∃ (k : ℝ), A.2 = k*(A.1 - 3) ∧ B.2 = k*(B.1 - 3)) →
    vectorAdditionCondition A B M t →
    (a = 2 ∧ b = Real.sqrt 3 ∧ c = 1) ∧
    ((a^2 - b^2) / a^2 = 1/4) ∧
    (ellipse a b h = {p : ℝ × ℝ | p.1^2/4 + p.2^2/3 = 1}) ∧
    (-2 < t ∧ t < 2)) :=
sorry

end ellipse_properties_l2978_297866


namespace max_inspector_sum_l2978_297812

/-- Represents the configuration of towers in the city of Flat -/
structure TowerConfiguration where
  one_floor : ℕ  -- Number of 1-floor towers
  two_floor : ℕ  -- Number of 2-floor towers

/-- Calculates the total height of all towers -/
def total_height (config : TowerConfiguration) : ℕ :=
  config.one_floor + 2 * config.two_floor

/-- Calculates the inspector's sum for a given configuration -/
def inspector_sum (config : TowerConfiguration) : ℕ :=
  config.one_floor * config.two_floor

/-- Theorem stating that the maximum inspector's sum is 112 -/
theorem max_inspector_sum :
  ∃ (config : TowerConfiguration),
    total_height config = 30 ∧
    inspector_sum config = 112 ∧
    ∀ (other : TowerConfiguration),
      total_height other = 30 →
      inspector_sum other ≤ 112 := by
  sorry

end max_inspector_sum_l2978_297812


namespace quadratic_factorization_l2978_297849

theorem quadratic_factorization (y : ℝ) : 3 * y^2 - 6 * y + 3 = 3 * (y - 1)^2 := by
  sorry

end quadratic_factorization_l2978_297849


namespace largest_valid_number_l2978_297846

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 = 1 ∨ n / 100 = 7 ∨ n / 100 = 0) ∧
  ((n / 10) % 10 = 1 ∨ (n / 10) % 10 = 7 ∨ (n / 10) % 10 = 0) ∧
  (n % 10 = 1 ∨ n % 10 = 7 ∨ n % 10 = 0) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 710 :=
sorry

end largest_valid_number_l2978_297846


namespace jelly_bean_probability_l2978_297810

theorem jelly_bean_probability (p_red p_orange p_green p_yellow : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_green = 0.25 →
  p_red + p_orange + p_green + p_yellow = 1 →
  p_yellow = 0.25 := by
sorry

end jelly_bean_probability_l2978_297810


namespace circle_bounds_l2978_297854

theorem circle_bounds (x y : ℝ) : 
  x^2 + (y - 1)^2 = 1 →
  (-Real.sqrt 3 / 3 ≤ (y - 1) / (x - 2) ∧ (y - 1) / (x - 2) ≤ Real.sqrt 3 / 3) ∧
  (1 - Real.sqrt 5 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 5) := by
  sorry

end circle_bounds_l2978_297854


namespace range_of_a_in_fourth_quadrant_l2978_297837

/-- Given a complex number z that corresponds to a point in the fourth quadrant,
    prove that the real parameter a in z = (a + 2i³) / (2 - i) is in the range (-1, 4) -/
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := (a + 2 * Complex.I ^ 3) / (2 - Complex.I)
  (z.re > 0 ∧ z.im < 0) → -1 < a ∧ a < 4 := by
  sorry

end range_of_a_in_fourth_quadrant_l2978_297837


namespace stuffed_animal_sales_difference_l2978_297831

theorem stuffed_animal_sales_difference (thor jake quincy : ℕ) 
  (h1 : jake = thor + 10)
  (h2 : quincy = 10 * thor)
  (h3 : quincy = 200) :
  quincy - jake = 170 := by
  sorry

end stuffed_animal_sales_difference_l2978_297831


namespace impossible_all_defective_l2978_297850

/-- Given 10 products with 2 defective ones, the probability of selecting 3 defective products
    when randomly choosing 3 is zero. -/
theorem impossible_all_defective (total : Nat) (defective : Nat) (selected : Nat)
    (h1 : total = 10)
    (h2 : defective = 2)
    (h3 : selected = 3) :
  Nat.choose defective selected / Nat.choose total selected = 0 := by
  sorry

end impossible_all_defective_l2978_297850


namespace carter_red_velvet_cakes_l2978_297838

/-- The number of red velvet cakes Carter usually bakes per week -/
def usual_red_velvet : ℕ := sorry

/-- The number of cheesecakes Carter usually bakes per week -/
def usual_cheesecakes : ℕ := 6

/-- The number of muffins Carter usually bakes per week -/
def usual_muffins : ℕ := 5

/-- The total number of additional cakes Carter baked this week -/
def additional_cakes : ℕ := 38

/-- The factor by which Carter increased his baking this week -/
def increase_factor : ℕ := 3

theorem carter_red_velvet_cakes :
  (usual_cheesecakes + usual_muffins + usual_red_velvet) + additional_cakes =
  increase_factor * (usual_cheesecakes + usual_muffins + usual_red_velvet) →
  usual_red_velvet = 8 := by
sorry

end carter_red_velvet_cakes_l2978_297838


namespace no_double_application_increment_l2978_297857

theorem no_double_application_increment :
  ¬∃ f : ℤ → ℤ, ∀ x : ℤ, f (f x) = x + 1 := by sorry

end no_double_application_increment_l2978_297857


namespace peter_chip_cost_l2978_297865

/-- Calculates the cost to consume a given number of calories from chips, given the calorie content per chip, chips per bag, and cost per bag. -/
def cost_for_calories (calories_per_chip : ℕ) (chips_per_bag : ℕ) (cost_per_bag : ℚ) (target_calories : ℕ) : ℚ :=
  let calories_per_bag := calories_per_chip * chips_per_bag
  let bags_needed := (target_calories + calories_per_bag - 1) / calories_per_bag
  bags_needed * cost_per_bag

/-- Theorem stating that Peter needs to spend $4 to consume 480 calories of chips. -/
theorem peter_chip_cost : cost_for_calories 10 24 2 480 = 4 := by
  sorry

end peter_chip_cost_l2978_297865


namespace class_size_problem_l2978_297816

theorem class_size_problem :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 29 :=
by sorry

end class_size_problem_l2978_297816


namespace percentage_problem_l2978_297835

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.20 * 1000 - 30 → x = 680 := by
  sorry

end percentage_problem_l2978_297835


namespace actress_not_lead_plays_l2978_297887

theorem actress_not_lead_plays (total_plays : ℕ) (lead_percentage : ℚ) 
  (h1 : total_plays = 100)
  (h2 : lead_percentage = 80 / 100) :
  total_plays - (total_plays * lead_percentage).floor = 20 := by
sorry

end actress_not_lead_plays_l2978_297887


namespace camel_cost_l2978_297839

theorem camel_cost (camel horse ox elephant : ℕ → ℚ) 
  (h1 : 10 * camel 1 = 24 * horse 1)
  (h2 : 16 * horse 1 = 4 * ox 1)
  (h3 : 6 * ox 1 = 4 * elephant 1)
  (h4 : 10 * elephant 1 = 120000) :
  camel 1 = 4800 := by
  sorry

end camel_cost_l2978_297839


namespace gcd_of_powers_of_79_l2978_297879

theorem gcd_of_powers_of_79 :
  Nat.Prime 79 →
  Nat.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
sorry

end gcd_of_powers_of_79_l2978_297879


namespace max_value_range_l2978_297856

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) * (x - a)

-- State the theorem
theorem max_value_range (a : ℝ) :
  (∀ x, f_derivative a x = a * (x + 1) * (x - a)) →
  (∃ x₀, ∀ x, f x ≤ f x₀) →
  (∀ x, x < a → f_derivative a x > 0) →
  (∀ x, x > a → f_derivative a x < 0) →
  a ∈ Set.Ioo (-1 : ℝ) 0 :=
sorry

end max_value_range_l2978_297856


namespace factory_growth_rate_l2978_297867

theorem factory_growth_rate (x : ℝ) : 
  (1 + x)^2 = 1.2 → x < 0.1 := by sorry

end factory_growth_rate_l2978_297867


namespace f_monotonically_decreasing_l2978_297872

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x - 2

-- Theorem statement
theorem f_monotonically_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ := by sorry

end f_monotonically_decreasing_l2978_297872


namespace arcsin_one_half_l2978_297882

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_l2978_297882


namespace concentric_circles_radii_difference_l2978_297815

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : π * R^2 / (π * r^2) = 4) :
  R - r = r :=
sorry

end concentric_circles_radii_difference_l2978_297815


namespace triathlon_speeds_correct_l2978_297832

/-- Represents the minimum speeds required for Maria to complete the triathlon within the given time limit. -/
def triathlon_speeds (swim_dist : ℝ) (cycle_dist : ℝ) (run_dist : ℝ) (time_limit : ℝ) : ℝ × ℝ × ℝ :=
  let swim_speed : ℝ := 60
  let run_speed : ℝ := 3 * swim_speed
  let cycle_speed : ℝ := 2.5 * run_speed
  (swim_speed, cycle_speed, run_speed)

/-- Theorem stating that the calculated speeds are correct for the given triathlon conditions. -/
theorem triathlon_speeds_correct 
  (swim_dist : ℝ) (cycle_dist : ℝ) (run_dist : ℝ) (time_limit : ℝ)
  (h_swim : swim_dist = 800)
  (h_cycle : cycle_dist = 20000)
  (h_run : run_dist = 4000)
  (h_time : time_limit = 80) :
  let (swim_speed, cycle_speed, run_speed) := triathlon_speeds swim_dist cycle_dist run_dist time_limit
  swim_speed = 60 ∧ cycle_speed = 450 ∧ run_speed = 180 ∧
  swim_dist / swim_speed + cycle_dist / cycle_speed + run_dist / run_speed ≤ time_limit :=
by sorry

#check triathlon_speeds_correct

end triathlon_speeds_correct_l2978_297832


namespace solve_for_x_l2978_297824

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 := by
  sorry

end solve_for_x_l2978_297824


namespace sample_customers_l2978_297896

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (leftover_samples : ℕ) :
  samples_per_box = 20 →
  boxes_opened = 12 →
  leftover_samples = 5 →
  (samples_per_box * boxes_opened - leftover_samples : ℕ) = 235 :=
by sorry

end sample_customers_l2978_297896


namespace exists_good_permutation_iff_power_of_two_l2978_297842

/-- A permutation is "good" if for any i < j < k, n doesn't divide (aᵢ + aₖ - 2aⱼ) -/
def is_good_permutation (n : ℕ) (a : Fin n → ℕ) : Prop :=
  ∀ i j k : Fin n, i < j → j < k → ¬(n ∣ a i + a k - 2 * a j)

/-- A natural number n ≥ 3 has a good permutation if and only if it's a power of 2 -/
theorem exists_good_permutation_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ (a : Fin n → ℕ), Function.Bijective a ∧ is_good_permutation n a) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end exists_good_permutation_iff_power_of_two_l2978_297842


namespace initial_money_calculation_l2978_297891

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 400 → initial_money = 1000 :=
by
  sorry

#check initial_money_calculation

end initial_money_calculation_l2978_297891


namespace splash_width_is_seven_l2978_297828

/-- The width of a splash made by a pebble in meters -/
def pebble_splash : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash : ℚ := 2

/-- The number of pebbles thrown -/
def pebbles_thrown : ℕ := 6

/-- The number of rocks thrown -/
def rocks_thrown : ℕ := 3

/-- The number of boulders thrown -/
def boulders_thrown : ℕ := 2

/-- The total width of splashes made by TreQuan's throws -/
def total_splash_width : ℚ := 
  pebble_splash * pebbles_thrown + 
  rock_splash * rocks_thrown + 
  boulder_splash * boulders_thrown

theorem splash_width_is_seven : total_splash_width = 7 := by
  sorry

end splash_width_is_seven_l2978_297828


namespace two_true_propositions_l2978_297852

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a predicate for a right angle
def is_right_angle (angle : Real) : Prop := angle = 90

-- Define a predicate for a right triangle
def is_right_triangle (t : Triangle) : Prop := ∃ angle, is_right_angle angle

-- Define the original proposition
def original_prop (t : Triangle) : Prop :=
  is_right_angle t.C → is_right_triangle t

-- Define the converse proposition
def converse_prop (t : Triangle) : Prop :=
  is_right_triangle t → is_right_angle t.C

-- Define the inverse proposition
def inverse_prop (t : Triangle) : Prop :=
  ¬(is_right_angle t.C) → ¬(is_right_triangle t)

-- Define the contrapositive proposition
def contrapositive_prop (t : Triangle) : Prop :=
  ¬(is_right_triangle t) → ¬(is_right_angle t.C)

-- Theorem stating that exactly two of these propositions are true
theorem two_true_propositions :
  ∃ (t : Triangle),
    (original_prop t ∧ contrapositive_prop t) ∧
    ¬(converse_prop t ∨ inverse_prop t) :=
  sorry

end two_true_propositions_l2978_297852


namespace quadrant_passing_implies_negative_m_l2978_297804

/-- A linear function passing through the second, third, and fourth quadrants -/
structure QuadrantPassingFunction where
  m : ℝ
  passes_second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = -3 * x + m
  passes_third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = -3 * x + m
  passes_fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = -3 * x + m

/-- Theorem: If a linear function y = -3x + m passes through the second, third, and fourth quadrants, then m is negative -/
theorem quadrant_passing_implies_negative_m (f : QuadrantPassingFunction) : f.m < 0 :=
  sorry

end quadrant_passing_implies_negative_m_l2978_297804


namespace max_area_rectangular_pen_l2978_297806

/-- Given a rectangular pen with a perimeter of 60 feet, the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * (x + y) = 60 →
  x * y ≤ 225 :=
by sorry

end max_area_rectangular_pen_l2978_297806


namespace opposite_of_sqrt3_minus_2_l2978_297868

theorem opposite_of_sqrt3_minus_2 :
  -(Real.sqrt 3 - 2) = 2 - Real.sqrt 3 := by sorry

end opposite_of_sqrt3_minus_2_l2978_297868


namespace sheets_per_student_l2978_297801

theorem sheets_per_student (num_classes : ℕ) (students_per_class : ℕ) (total_sheets : ℕ) :
  num_classes = 4 →
  students_per_class = 20 →
  total_sheets = 400 →
  total_sheets / (num_classes * students_per_class) = 5 := by
  sorry

end sheets_per_student_l2978_297801


namespace bus_total_capacity_l2978_297817

/-- Represents the seating capacity of a bus with specific seating arrangements -/
def bus_capacity (left_seats : ℕ) (right_seat_diff : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seat_diff
  let total_regular_seats := left_seats + right_seats
  let regular_capacity := total_regular_seats * people_per_seat
  regular_capacity + back_seat_capacity

/-- Theorem stating the total seating capacity of the bus -/
theorem bus_total_capacity :
  bus_capacity 15 3 3 8 = 89 := by
  sorry

#eval bus_capacity 15 3 3 8

end bus_total_capacity_l2978_297817


namespace bike_price_l2978_297873

theorem bike_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_payment = 200 → 
  upfront_percentage * total_price = upfront_payment → 
  total_price = 1000 := by
sorry

end bike_price_l2978_297873


namespace trees_difference_l2978_297892

theorem trees_difference (initial_trees : ℕ) (died_trees : ℕ) : 
  initial_trees = 14 → died_trees = 9 → died_trees - (initial_trees - died_trees) = 4 := by
  sorry

end trees_difference_l2978_297892


namespace sum_twenty_ways_l2978_297874

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the target sum
def target_sum : ℕ := 20

-- Define the minimum value on a die
def min_value : ℕ := 1

-- Define the maximum value on a die
def max_value : ℕ := 6

-- Function to calculate the number of ways to achieve the target sum
def ways_to_achieve_sum (n d s min max : ℕ) : ℕ :=
  sorry

-- Theorem stating that the number of ways to achieve a sum of 20 with 5 dice is 721
theorem sum_twenty_ways : ways_to_achieve_sum num_dice target_sum min_value max_value = 721 := by
  sorry

end sum_twenty_ways_l2978_297874


namespace interest_related_to_gender_l2978_297895

/-- Represents the chi-square statistic -/
def chi_square : ℝ := 3.918

/-- Represents the critical value -/
def critical_value : ℝ := 3.841

/-- The probability that the chi-square statistic is greater than or equal to the critical value -/
def p_value : ℝ := 0.05

/-- The confidence level -/
def confidence_level : ℝ := 1 - p_value

theorem interest_related_to_gender :
  chi_square > critical_value →
  confidence_level = 0.95 →
  ∃ (relation : Prop), relation ∧ confidence_level = 0.95 :=
by sorry

end interest_related_to_gender_l2978_297895


namespace expression_equivalence_l2978_297875

theorem expression_equivalence (a b : ℝ) : (a) - (b) - 3 * (a + b) - b = a - 8 * b := by
  sorry

end expression_equivalence_l2978_297875


namespace line_parameterization_values_l2978_297821

/-- A line parameterized by a point and a direction vector -/
structure ParametricLine (α : Type*) [Field α] where
  point : α × α
  direction : α × α

/-- The equation of a line in slope-intercept form -/
structure LineEquation (α : Type*) [Field α] where
  slope : α
  intercept : α

/-- Check if a point lies on a line given by slope-intercept equation -/
def LineEquation.contains_point {α : Type*} [Field α] (l : LineEquation α) (p : α × α) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- Check if a parametric line is equivalent to a line equation -/
def parametric_line_equiv_equation {α : Type*} [Field α] 
  (pl : ParametricLine α) (le : LineEquation α) : Prop :=
  ∀ t : α, le.contains_point (pl.point.1 + t * pl.direction.1, pl.point.2 + t * pl.direction.2)

theorem line_parameterization_values 
  (l : LineEquation ℝ) 
  (pl : ParametricLine ℝ) 
  (h_equiv : parametric_line_equiv_equation pl l) 
  (h_slope : l.slope = 2) 
  (h_intercept : l.intercept = -7) 
  (h_point : pl.point = (s, 2)) 
  (h_direction : pl.direction = (3, m)) : 
  s = 9/2 ∧ m = -1 := by
  sorry

end line_parameterization_values_l2978_297821


namespace adams_final_score_l2978_297813

/-- Calculates the final score in a trivia game -/
def final_score (correct_first_half correct_second_half points_per_question : ℕ) : ℕ :=
  (correct_first_half + correct_second_half) * points_per_question

/-- Theorem: Adam's final score in the trivia game is 50 points -/
theorem adams_final_score : 
  final_score 5 5 5 = 50 := by
  sorry

end adams_final_score_l2978_297813


namespace factorial_solutions_l2978_297811

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_solutions :
  ∀ x y z : ℕ, factorial x + 2^y = factorial z →
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 2)) :=
by sorry

end factorial_solutions_l2978_297811
