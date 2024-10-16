import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2998_299807

-- First expression
theorem simplify_expression_1 (a b : ℝ) :
  -b * (2 * a - b) + (a + b)^2 = a^2 + 2 * b^2 := by sorry

-- Second expression
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (1 - x / (2 + x)) / ((x^2 - 4) / (x^2 + 4*x + 4)) = 2 / (x - 2) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2998_299807


namespace NUMINAMATH_CALUDE_triangle_side_length_l2998_299897

/-- Triangle DEF with given properties -/
structure Triangle where
  D : ℝ  -- Angle D
  E : ℝ  -- Angle E
  F : ℝ  -- Angle F
  d : ℝ  -- Side length opposite to angle D
  e : ℝ  -- Side length opposite to angle E
  f : ℝ  -- Side length opposite to angle F

/-- The theorem stating the properties of the triangle and the value of f -/
theorem triangle_side_length 
  (t : Triangle)
  (h1 : t.d = 7)
  (h2 : t.e = 3)
  (h3 : Real.cos (t.D - t.E) = 7/8) :
  t.f = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2998_299897


namespace NUMINAMATH_CALUDE_container_volume_scaling_l2998_299817

theorem container_volume_scaling (V k : ℝ) (h : k > 0) :
  let new_volume := V * k^3
  new_volume = V * k * k * k :=
by sorry

end NUMINAMATH_CALUDE_container_volume_scaling_l2998_299817


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2998_299843

theorem binomial_divisibility (p m : ℕ) (hp : Prime p) (hm : m > 0) :
  p^m ∣ (Nat.choose (p^m) p - p^(m-1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2998_299843


namespace NUMINAMATH_CALUDE_charcoal_drawings_l2998_299892

theorem charcoal_drawings (total : Nat) (colored_pencil : Nat) (blending_marker : Nat) :
  total = 60 → colored_pencil = 24 → blending_marker = 19 →
  total - colored_pencil - blending_marker = 17 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_l2998_299892


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2998_299880

theorem arithmetic_sequence_length 
  (a : ℤ) (an : ℤ) (d : ℤ) (n : ℕ) 
  (h1 : a = -50) 
  (h2 : an = 74) 
  (h3 : d = 6) 
  (h4 : an = a + (n - 1) * d) : n = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2998_299880


namespace NUMINAMATH_CALUDE_problem_statement_l2998_299805

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + y = 5) 
  (h2 : x + 3 * y = 6) : 
  5 * x^2 + 8 * x * y + 5 * y^2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2998_299805


namespace NUMINAMATH_CALUDE_students_with_two_skills_l2998_299896

theorem students_with_two_skills (total : ℕ) (cant_paint cant_write cant_music : ℕ) : 
  total = 150 →
  cant_paint = 75 →
  cant_write = 90 →
  cant_music = 45 →
  ∃ (two_skills : ℕ), two_skills = 90 ∧ 
    two_skills = (total - cant_paint) + (total - cant_write) + (total - cant_music) - total :=
by sorry

end NUMINAMATH_CALUDE_students_with_two_skills_l2998_299896


namespace NUMINAMATH_CALUDE_weight_of_BaCl2_l2998_299891

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The number of moles of BaCl2 -/
def moles_BaCl2 : ℝ := 8

/-- The molecular weight of BaCl2 in g/mol -/
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

/-- The total weight of BaCl2 in grams -/
def total_weight_BaCl2 : ℝ := molecular_weight_BaCl2 * moles_BaCl2

theorem weight_of_BaCl2 : total_weight_BaCl2 = 1665.84 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaCl2_l2998_299891


namespace NUMINAMATH_CALUDE_legos_won_l2998_299868

def initial_legos : ℕ := 2080
def final_legos : ℕ := 2097

theorem legos_won : final_legos - initial_legos = 17 := by
  sorry

end NUMINAMATH_CALUDE_legos_won_l2998_299868


namespace NUMINAMATH_CALUDE_solve_chimney_problem_l2998_299882

def chimney_problem (brenda_time brandon_time charlie_time : ℝ)
  (output_decrease : ℝ) (combined_time : ℝ) : Prop :=
  let individual_rates := 1 / brenda_time + 1 / brandon_time + 1 / charlie_time
  let effective_rate := individual_rates - output_decrease / (brenda_time * brandon_time * charlie_time)
  effective_rate * combined_time = 330

theorem solve_chimney_problem :
  chimney_problem 9 10 12 15 3.5 := by sorry

end NUMINAMATH_CALUDE_solve_chimney_problem_l2998_299882


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2998_299801

theorem absolute_value_equation_solutions (m n k : ℝ) : 
  (∀ x : ℝ, |2*x - 3| + m ≠ 0) →
  (∃! x : ℝ, |3*x - 4| + n = 0) →
  (∃ x y : ℝ, x ≠ y ∧ |4*x - 5| + k = 0 ∧ |4*y - 5| + k = 0) →
  m > n ∧ n > k :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2998_299801


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2998_299813

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 5 > 3 ∧ x > a) ↔ x > -2) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2998_299813


namespace NUMINAMATH_CALUDE_job_land_theorem_l2998_299861

/-- Represents the total land owned by Job in hectares -/
def total_land : ℕ := 150

/-- Represents the land occupied by house and farm machinery in hectares -/
def house_and_machinery : ℕ := 25

/-- Represents the land reserved for future expansion in hectares -/
def future_expansion : ℕ := 15

/-- Represents the land dedicated to rearing cattle in hectares -/
def cattle_land : ℕ := 40

/-- Represents the land used for crop production in hectares -/
def crop_land : ℕ := 70

/-- Theorem stating that the total land is equal to the sum of all land uses -/
theorem job_land_theorem : 
  total_land = house_and_machinery + future_expansion + cattle_land + crop_land := by
  sorry

end NUMINAMATH_CALUDE_job_land_theorem_l2998_299861


namespace NUMINAMATH_CALUDE_distance_to_point_l2998_299834

theorem distance_to_point : Real.sqrt (8^2 + (-15)^2) = 17 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l2998_299834


namespace NUMINAMATH_CALUDE_chord_length_polar_curve_l2998_299895

/-- The length of the chord AB, where A is the point (3, 0) and B is the other intersection point
    of the line x = 3 with the curve ρ = 4cosθ in polar coordinates. -/
theorem chord_length_polar_curve : ∃ (A B : ℝ × ℝ),
  A = (3, 0) ∧
  B.1 = 3 ∧
  (B.1 - 2)^2 + B.2^2 = 4 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_polar_curve_l2998_299895


namespace NUMINAMATH_CALUDE_parallelogram_equation_l2998_299845

/-- Represents a parallelogram in the xy-plane -/
def is_parallelogram (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ) (h : b ≠ 0),
    ∀ (x y : ℝ), f x y = |b*x - (a+c)*y| + |b*x + (c-a)*y - b*c| - b*c

theorem parallelogram_equation :
  ∃ (f : ℝ → ℝ → ℝ), is_parallelogram f :=
sorry

end NUMINAMATH_CALUDE_parallelogram_equation_l2998_299845


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2998_299894

theorem triangle_side_lengths 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_sum : a + b = 13)
  (h_angle : C = π/3) :
  ((a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l2998_299894


namespace NUMINAMATH_CALUDE_line_intercept_sum_minimum_equality_condition_l2998_299863

theorem line_intercept_sum_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b = a * b) : a + b ≥ 4 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b = a * b) : a + b = 4 ↔ a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_minimum_equality_condition_l2998_299863


namespace NUMINAMATH_CALUDE_correct_exponent_calculation_l2998_299870

theorem correct_exponent_calculation (a : ℝ) : (-a)^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_exponent_calculation_l2998_299870


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2998_299869

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2998_299869


namespace NUMINAMATH_CALUDE_max_area_OAPF_l2998_299802

/-- The equation of ellipse C is (x^2/9) + (y^2/10) = 1 -/
def ellipse_equation (x y : ℝ) : Prop := x^2/9 + y^2/10 = 1

/-- F is the upper focus of ellipse C -/
def F : ℝ × ℝ := (0, 1)

/-- A is the right vertex of ellipse C -/
def A : ℝ × ℝ := (3, 0)

/-- P is a point on ellipse C located in the first quadrant -/
def P : ℝ × ℝ := sorry

/-- The area of quadrilateral OAPF -/
def area_OAPF (P : ℝ × ℝ) : ℝ := sorry

theorem max_area_OAPF :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ P.1 > 0 ∧ P.2 > 0 ∧
  ∀ (Q : ℝ × ℝ), ellipse_equation Q.1 Q.2 → Q.1 > 0 → Q.2 > 0 →
  area_OAPF P ≥ area_OAPF Q ∧
  area_OAPF P = (3 * Real.sqrt 11) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_area_OAPF_l2998_299802


namespace NUMINAMATH_CALUDE_range_of_a_l2998_299800

theorem range_of_a (π : ℝ) (h : π > 0) : 
  ∀ a : ℝ, (∃ x : ℝ, x < 0 ∧ (1/π)^x = (1+a)/(1-a)) → 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2998_299800


namespace NUMINAMATH_CALUDE_count_integer_solutions_l2998_299854

theorem count_integer_solutions : ∃! A : ℕ, 
  A = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 + p.2 ≥ A ∧ 
    p.1 ≤ 6 ∧ 
    p.2 ≤ 7
  ) (Finset.product (Finset.range 7) (Finset.range 8))).card ∧
  A = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l2998_299854


namespace NUMINAMATH_CALUDE_boyd_boys_percentage_l2998_299830

/-- Represents the number of friends on a social media platform -/
structure SocialMediaFriends where
  boys : ℕ
  girls : ℕ

/-- Represents a person's friends on different social media platforms -/
structure Person where
  facebook : SocialMediaFriends
  instagram : SocialMediaFriends

def Julian : Person :=
  { facebook := { boys := 48, girls := 32 },
    instagram := { boys := 45, girls := 105 } }

def Boyd : Person :=
  { facebook := { boys := 1, girls := 64 },
    instagram := { boys := 135, girls := 0 } }

def total_friends (p : Person) : ℕ :=
  p.facebook.boys + p.facebook.girls + p.instagram.boys + p.instagram.girls

def boys_percentage (p : Person) : ℚ :=
  (p.facebook.boys + p.instagram.boys : ℚ) / total_friends p

theorem boyd_boys_percentage :
  boys_percentage Boyd = 68 / 100 :=
sorry

end NUMINAMATH_CALUDE_boyd_boys_percentage_l2998_299830


namespace NUMINAMATH_CALUDE_more_seventh_graders_l2998_299888

theorem more_seventh_graders (n m : ℕ) 
  (h1 : n > 0) 
  (h2 : m > 0) 
  (h3 : 7 * n = 6 * m) : 
  m > n :=
by
  sorry

end NUMINAMATH_CALUDE_more_seventh_graders_l2998_299888


namespace NUMINAMATH_CALUDE_number_of_divisors_of_fourth_power_l2998_299804

/-- Given a positive integer n where n = p₁ * p₂² * p₃⁵ and p₁, p₂, and p₃ are different prime numbers,
    the number of positive divisors of x = n⁴ is 945. -/
theorem number_of_divisors_of_fourth_power (p₁ p₂ p₃ : Nat) (h_prime₁ : Prime p₁) (h_prime₂ : Prime p₂)
    (h_prime₃ : Prime p₃) (h_distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃) :
    let n := p₁ * p₂^2 * p₃^5
    let x := n^4
    (Nat.divisors x).card = 945 := by
  sorry

#check number_of_divisors_of_fourth_power

end NUMINAMATH_CALUDE_number_of_divisors_of_fourth_power_l2998_299804


namespace NUMINAMATH_CALUDE_problem_statement_l2998_299837

theorem problem_statement (n : ℕ+) : 
  3 * (Nat.choose (n - 1) (n - 5)) = 5 * (Nat.factorial (n - 2) / Nat.factorial (n - 4)) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2998_299837


namespace NUMINAMATH_CALUDE_expression_evaluation_l2998_299832

theorem expression_evaluation : 
  (3^2 + 5^2 + 7^2) / (2^2 + 4^2 + 6^2) - (2^2 + 4^2 + 6^2) / (3^2 + 5^2 + 7^2) = 3753/4648 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2998_299832


namespace NUMINAMATH_CALUDE_puppy_food_theorem_l2998_299884

/-- Calculates the total amount of food a puppy eats over 4 weeks given a specific feeding schedule. -/
def puppyFoodConsumption (daysInWeek : ℕ) (weeksTotal : ℕ) (foodPerDayWeek1And2 : ℚ) 
  (foodPerDayWeek3And4 : ℚ) (foodToday : ℚ) : ℚ :=
  let firstTwoWeeks := daysInWeek * 2 * foodPerDayWeek1And2
  let secondTwoWeeks := daysInWeek * 2 * foodPerDayWeek3And4
  firstTwoWeeks + secondTwoWeeks + foodToday

theorem puppy_food_theorem :
  puppyFoodConsumption 7 4 (3 * (1/4)) (2 * (1/2)) (1/2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_puppy_food_theorem_l2998_299884


namespace NUMINAMATH_CALUDE_function_composition_equality_l2998_299879

theorem function_composition_equality (a b c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = c * x + d)
  (ha : a = 2 * c) :
  (∀ x, f (g x) = g (f x)) ↔ (b = d ∨ c = 1/2) := by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2998_299879


namespace NUMINAMATH_CALUDE_division_of_fractions_l2998_299860

theorem division_of_fractions : (3/8) / (5/12) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l2998_299860


namespace NUMINAMATH_CALUDE_perpendicular_line_angle_l2998_299808

-- Define the perpendicularity condition
def isPerpendicular (θ : Real) : Prop :=
  ∃ t : Real, (1 + t * Real.cos θ = t * Real.sin θ) ∧ 
              (Real.tan θ = -1)

-- State the theorem
theorem perpendicular_line_angle :
  ∀ θ : Real, 0 ≤ θ ∧ θ < π → isPerpendicular θ → θ = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_angle_l2998_299808


namespace NUMINAMATH_CALUDE_dollar_evaluation_l2998_299841

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_evaluation (x y : ℝ) :
  dollar (2*x + 3*y) (3*x - 4*y) = x^2 - 14*x*y + 49*y^2 := by
  sorry

end NUMINAMATH_CALUDE_dollar_evaluation_l2998_299841


namespace NUMINAMATH_CALUDE_paul_prediction_accuracy_l2998_299876

/-- Represents a team in the FIFA World Cup -/
inductive Team
| Ghana
| Bolivia
| Argentina
| France

/-- The probability of a team winning the tournament -/
def winProbability (t : Team) : ℚ :=
  match t with
  | Team.Ghana => 1/2
  | Team.Bolivia => 1/6
  | Team.Argentina => 1/6
  | Team.France => 1/6

/-- The probability of Paul correctly predicting the winner -/
def paulCorrectProbability : ℚ :=
  (winProbability Team.Ghana)^2 +
  (winProbability Team.Bolivia)^2 +
  (winProbability Team.Argentina)^2 +
  (winProbability Team.France)^2

theorem paul_prediction_accuracy :
  paulCorrectProbability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_paul_prediction_accuracy_l2998_299876


namespace NUMINAMATH_CALUDE_math_competition_score_l2998_299806

theorem math_competition_score (x : ℕ) : 
  let total_problems := 8 * x + x
  let missed_problems := 2 * x
  let bonus_problems := x
  let standard_points := (total_problems - missed_problems - bonus_problems)
  let bonus_points := 2 * bonus_problems
  let total_available_points := total_problems + bonus_problems
  let scored_points := standard_points + bonus_points
  (scored_points : ℚ) / total_available_points = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_score_l2998_299806


namespace NUMINAMATH_CALUDE_equality_of_sqrt_five_terms_l2998_299816

theorem equality_of_sqrt_five_terms 
  (a b c d : ℚ) 
  (h : a + b * Real.sqrt 5 = c + d * Real.sqrt 5) : 
  a = c ∧ b = d := by
sorry

end NUMINAMATH_CALUDE_equality_of_sqrt_five_terms_l2998_299816


namespace NUMINAMATH_CALUDE_thirteen_travel_methods_l2998_299811

/-- The number of different methods to travel from Place A to Place B -/
def travel_methods (bus_services train_services ship_services : ℕ) : ℕ :=
  bus_services + train_services + ship_services

/-- Theorem: There are 13 different methods to travel from Place A to Place B -/
theorem thirteen_travel_methods :
  travel_methods 8 3 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_travel_methods_l2998_299811


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2998_299862

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b ∧ a ≠ b

-- Define the linear function
def linear_function (x : ℝ) (a b : ℝ) : ℝ := (a*b - 1)*x + a + b

-- Theorem: The linear function does not pass through the third quadrant
theorem linear_function_not_in_third_quadrant (a b : ℝ) :
  roots a b →
  ∀ x y : ℝ, y = linear_function x a b →
  ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2998_299862


namespace NUMINAMATH_CALUDE_train_passengers_with_hats_l2998_299849

theorem train_passengers_with_hats 
  (total_adults : ℕ) 
  (men_percentage : ℚ) 
  (men_with_hats_percentage : ℚ) 
  (women_with_hats_percentage : ℚ) 
  (h1 : total_adults = 3600) 
  (h2 : men_percentage = 40 / 100) 
  (h3 : men_with_hats_percentage = 15 / 100) 
  (h4 : women_with_hats_percentage = 25 / 100) : 
  ℕ := by
  sorry

#check train_passengers_with_hats

end NUMINAMATH_CALUDE_train_passengers_with_hats_l2998_299849


namespace NUMINAMATH_CALUDE_platform_length_l2998_299829

/-- The length of a platform given train specifications and crossing time -/
theorem platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 55 →
  crossing_time = 50.395968322534195 →
  ∃ platform_length : ℝ, abs (platform_length - 520) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2998_299829


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l2998_299855

/-- The hyperbola with equation x^2/16 - y^2/20 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 16 - p.2^2 / 20 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- Distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_distance 
  (P : ℝ × ℝ) 
  (h_P : P ∈ Hyperbola) 
  (h_dist : distance P F₁ = 9) : 
  distance P F₂ = 17 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l2998_299855


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2998_299825

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (s : ℝ) : 
  (∀ k < n, ¬ ∃ (t : ℝ) (l : ℕ), t > 0 ∧ t < 1/500 ∧ l^(1/3 : ℝ) = k + t) →
  s > 0 → 
  s < 1/500 → 
  m^(1/3 : ℝ) = n + s → 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2998_299825


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_64_l2998_299873

theorem arithmetic_square_root_of_64 : Real.sqrt 64 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_64_l2998_299873


namespace NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l2998_299853

theorem inverse_true_implies_negation_true (P : Prop) : 
  (¬P → ¬(¬P)) → (¬P) := by
  sorry

end NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l2998_299853


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l2998_299812

/-- Calculates the monthly income given the percentage saved and the amount saved -/
def calculate_income (percent_saved : ℚ) (amount_saved : ℚ) : ℚ :=
  amount_saved / percent_saved

/-- The percentage of income spent on various categories -/
def total_expenses : ℚ := 35 + 18 + 6 + 11 + 12 + 5 + 7

/-- The percentage of income saved -/
def percent_saved : ℚ := 100 - total_expenses

/-- The amount saved in Rupees -/
def amount_saved : ℚ := 12500

theorem monthly_income_calculation :
  calculate_income percent_saved amount_saved = 208333.33 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l2998_299812


namespace NUMINAMATH_CALUDE_square_completion_l2998_299857

theorem square_completion (x : ℝ) : x^2 + 5*x + 25/4 = (x + 5/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_completion_l2998_299857


namespace NUMINAMATH_CALUDE_sum_of_roots_l2998_299883

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 + 6*x^2 + 16*x = -15) 
  (hy : y^3 + 6*y^2 + 16*y = -17) : 
  x + y = -4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2998_299883


namespace NUMINAMATH_CALUDE_ellipse_equation_l2998_299887

/-- The standard equation of an ellipse with foci on the coordinate axes and passing through points A(√3, -2) and B(-2√3, 1) is x²/15 + y²/5 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 15 + y^2 / 5 = 1)) ∧
  (x^2 / 15 + y^2 / 5 = 1 → x = Real.sqrt 3 ∧ y = -2 ∨ x = -2 * Real.sqrt 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2998_299887


namespace NUMINAMATH_CALUDE_inequality_proof_l2998_299851

theorem inequality_proof (a b u v k : ℝ) 
  (ha : a > 0) (hb : b > 0) (huv : u < v) (hk : k > 0) :
  (a^u + b^u) / (a^v + b^v) ≥ (a^(u+k) + b^(u+k)) / (a^(v+k) + b^(v+k)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2998_299851


namespace NUMINAMATH_CALUDE_concert_stay_probability_concert_stay_probability_is_one_l2998_299864

/-- The probability that at least 4 out of 8 people stay for an entire concert, 
    given 4 are certain to stay and 4 have a 1/3 probability of staying. -/
theorem concert_stay_probability : ℝ :=
  let total_people : ℕ := 8
  let certain_stay : ℕ := 4
  let uncertain_stay : ℕ := 4
  let stay_prob : ℝ := 1/3
  
  let prob_exactly_4 := (1 - stay_prob) ^ uncertain_stay
  let prob_exactly_5 := uncertain_stay.choose 1 * stay_prob * (1 - stay_prob) ^ 3
  let prob_exactly_6 := uncertain_stay.choose 2 * stay_prob ^ 2 * (1 - stay_prob) ^ 2
  let prob_exactly_7 := uncertain_stay.choose 3 * stay_prob ^ 3 * (1 - stay_prob)
  let prob_exactly_8 := stay_prob ^ uncertain_stay

  prob_exactly_4 + prob_exactly_5 + prob_exactly_6 + prob_exactly_7 + prob_exactly_8

#check concert_stay_probability

theorem concert_stay_probability_is_one : concert_stay_probability = 1 := by
  sorry

end NUMINAMATH_CALUDE_concert_stay_probability_concert_stay_probability_is_one_l2998_299864


namespace NUMINAMATH_CALUDE_expression_value_l2998_299823

theorem expression_value (a b c : ℤ) (ha : a = 17) (hb : b = 21) (hc : c = 5) :
  (a - (b - c)) - ((a - b) - c) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2998_299823


namespace NUMINAMATH_CALUDE_retired_staff_samples_calculation_l2998_299898

/-- Calculates the number of samples for retired staff given total samples and ratio -/
def retired_staff_samples (total_samples : ℕ) (retired_ratio current_ratio student_ratio : ℕ) : ℕ :=
  let total_ratio := retired_ratio + current_ratio + student_ratio
  let unit_value := total_samples / total_ratio
  retired_ratio * unit_value

/-- Theorem stating that given 300 total samples and a ratio of 3:7:40, 
    the number of samples from retired staff is 18 -/
theorem retired_staff_samples_calculation :
  retired_staff_samples 300 3 7 40 = 18 := by
  sorry

end NUMINAMATH_CALUDE_retired_staff_samples_calculation_l2998_299898


namespace NUMINAMATH_CALUDE_unique_function_theorem_l2998_299890

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10^10}

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x ∈ S, f x ∈ S) ∧
  (∀ x ∈ S, f (x + 1) ≡ f (f x) + 1 [MOD 10^10]) ∧
  (f (10^10 + 1) = f 1)

theorem unique_function_theorem :
  ∀ f : ℕ → ℕ, is_valid_function f →
    ∀ x ∈ S, f x ≡ x [MOD 10^10] :=
by sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l2998_299890


namespace NUMINAMATH_CALUDE_regular_dodecahedron_vertex_count_l2998_299840

/-- A regular dodecahedron has 20 vertices. -/
def regular_dodecahedron_vertices : ℕ := 20

/-- The number of vertices in a regular dodecahedron is 20. -/
theorem regular_dodecahedron_vertex_count : 
  regular_dodecahedron_vertices = 20 := by sorry

end NUMINAMATH_CALUDE_regular_dodecahedron_vertex_count_l2998_299840


namespace NUMINAMATH_CALUDE_tank_capacity_l2998_299867

theorem tank_capacity (C : ℚ) : 
  (C > 0) →  -- The capacity is positive
  ((117 / 200) * C = 4680) →  -- Final volume equation
  (C = 8000) := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l2998_299867


namespace NUMINAMATH_CALUDE_sum_equals_200_l2998_299835

theorem sum_equals_200 : 148 + 32 + 18 + 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_200_l2998_299835


namespace NUMINAMATH_CALUDE_calculator_addition_correct_l2998_299839

/-- Represents a calculator button --/
inductive CalculatorButton
  | Digit (n : Nat)
  | Plus
  | Equals

/-- Represents a sequence of button presses on a calculator --/
def ButtonSequence := List CalculatorButton

/-- Evaluates a sequence of button presses and returns the result --/
def evaluate (seq : ButtonSequence) : Nat :=
  sorry

/-- The correct sequence of button presses to calculate 569 + 728 --/
def correctSequence : ButtonSequence :=
  [CalculatorButton.Digit 569, CalculatorButton.Plus, CalculatorButton.Digit 728, CalculatorButton.Equals]

theorem calculator_addition_correct :
  evaluate correctSequence = 569 + 728 :=
sorry

end NUMINAMATH_CALUDE_calculator_addition_correct_l2998_299839


namespace NUMINAMATH_CALUDE_inequality_proof_l2998_299803

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) : 
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2998_299803


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l2998_299899

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := sorry

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := 53

/-- The difference between the frog's jump and the grasshopper's jump in inches -/
def frog_grasshopper_diff : ℕ := 17

theorem grasshopper_jump_distance :
  grasshopper_jump = frog_jump - frog_grasshopper_diff :=
by sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l2998_299899


namespace NUMINAMATH_CALUDE_max_min_difference_c_l2998_299822

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 = 15) :
  (3 : ℝ) - (-7/3) = 16/3 := by sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l2998_299822


namespace NUMINAMATH_CALUDE_robotics_club_proof_l2998_299838

theorem robotics_club_proof (total : ℕ) (programming : ℕ) (electronics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : programming = 80)
  (h3 : electronics = 50)
  (h4 : both = 15) :
  total - (programming + electronics - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_proof_l2998_299838


namespace NUMINAMATH_CALUDE_total_growing_space_l2998_299821

/-- Represents a garden bed with length and width dimensions -/
structure GardenBed where
  length : ℕ
  width : ℕ

/-- Calculates the area of a garden bed -/
def area (bed : GardenBed) : ℕ := bed.length * bed.width

/-- Calculates the total area of multiple identical garden beds -/
def totalArea (bed : GardenBed) (count : ℕ) : ℕ := area bed * count

/-- The set of garden beds Amy is building -/
def amysGardenBeds : List (GardenBed × ℕ) := [
  (⟨5, 4⟩, 3),
  (⟨6, 3⟩, 4),
  (⟨7, 5⟩, 2),
  (⟨8, 4⟩, 1)
]

/-- Theorem stating that the total growing space is 234 sq ft -/
theorem total_growing_space :
  (amysGardenBeds.map (fun (bed, count) => totalArea bed count)).sum = 234 := by
  sorry

end NUMINAMATH_CALUDE_total_growing_space_l2998_299821


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2998_299852

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 1000 = n / 6)

theorem count_valid_numbers : 
  ∃ (s : Finset ℕ), (∀ n ∈ s, is_valid_number n) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2998_299852


namespace NUMINAMATH_CALUDE_invalid_diagonal_sets_l2998_299820

-- Define a function to check if a set of three numbers satisfies the condition
def isValidDiagonalSet (x y z : ℝ) : Prop :=
  x^2 + y^2 ≥ z^2 ∧ x^2 + z^2 ≥ y^2 ∧ y^2 + z^2 ≥ x^2

-- Theorem stating which sets are invalid for external diagonals of a right regular prism
theorem invalid_diagonal_sets :
  (¬ isValidDiagonalSet 3 4 6) ∧
  (¬ isValidDiagonalSet 5 5 8) ∧
  (¬ isValidDiagonalSet 7 8 12) ∧
  (isValidDiagonalSet 6 8 10) ∧
  (isValidDiagonalSet 3 4 5) :=
by sorry

end NUMINAMATH_CALUDE_invalid_diagonal_sets_l2998_299820


namespace NUMINAMATH_CALUDE_simplify_expression_l2998_299847

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  (15 / 8) * Real.sqrt (2 + 10 / 27) / Real.sqrt (25 / (12 * a^3)) = a * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2998_299847


namespace NUMINAMATH_CALUDE_school_survey_sampling_params_l2998_299856

/-- Systematic sampling parameters for a given population and sample size -/
def systematic_sampling_params (population : ℕ) (sample_size : ℕ) : ℕ × ℕ :=
  let n := population % sample_size
  let m := sample_size
  (n, m)

/-- Theorem stating the correct systematic sampling parameters for the given problem -/
theorem school_survey_sampling_params :
  systematic_sampling_params 1553 50 = (3, 50) := by
sorry

end NUMINAMATH_CALUDE_school_survey_sampling_params_l2998_299856


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l2998_299824

theorem product_ratio_theorem (a b c d e f : ℚ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l2998_299824


namespace NUMINAMATH_CALUDE_salesman_commission_problem_l2998_299827

/-- A problem about a salesman's commission schemes -/
theorem salesman_commission_problem 
  (old_commission_rate : ℝ)
  (fixed_salary : ℝ)
  (sales_threshold : ℝ)
  (total_sales : ℝ)
  (remuneration_difference : ℝ)
  (h1 : old_commission_rate = 0.05)
  (h2 : fixed_salary = 1000)
  (h3 : sales_threshold = 4000)
  (h4 : total_sales = 12000)
  (h5 : remuneration_difference = 600) :
  ∃ new_commission_rate : ℝ,
    new_commission_rate * (total_sales - sales_threshold) + fixed_salary = 
    old_commission_rate * total_sales + remuneration_difference ∧
    new_commission_rate = 0.025 := by
  sorry

end NUMINAMATH_CALUDE_salesman_commission_problem_l2998_299827


namespace NUMINAMATH_CALUDE_adjacent_xue_rong_rong_arrangements_l2998_299836

def num_bing_dung_dung : ℕ := 4
def num_xue_rong_rong : ℕ := 3

def adjacent_arrangements (n_bdd : ℕ) (n_xrr : ℕ) : ℕ :=
  2 * (n_bdd + 2).factorial * (n_bdd + 1)

theorem adjacent_xue_rong_rong_arrangements :
  adjacent_arrangements num_bing_dung_dung num_xue_rong_rong = 960 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_xue_rong_rong_arrangements_l2998_299836


namespace NUMINAMATH_CALUDE_candy_distribution_l2998_299809

theorem candy_distribution (total : Nat) (sisters : Nat) (take_away : Nat) : 
  total = 24 →
  sisters = 5 →
  take_away = 4 →
  (total - take_away) % sisters = 0 →
  ∀ x : Nat, x < take_away → (total - x) % sisters ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2998_299809


namespace NUMINAMATH_CALUDE_cubic_inequality_l2998_299893

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + 3*a*b*c > a*b*(a+b) + b*c*(b+c) + a*c*(a+c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2998_299893


namespace NUMINAMATH_CALUDE_least_possible_difference_l2998_299858

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → 
  Odd y → Odd z → 
  (∀ w, w = z - x → w ≥ 9) ∧ (∃ w, w = z - x ∧ w = 9) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l2998_299858


namespace NUMINAMATH_CALUDE_max_bribe_amount_l2998_299871

/-- Represents the bet amount in coins -/
def betAmount : ℤ := 100

/-- Represents the maximum bribe amount in coins -/
def maxBribe : ℤ := 199

/-- 
Proves that the maximum bribe a person would pay to avoid eviction is 199 coins, 
given a bet where they lose 100 coins if evicted and gain 100 coins if not evicted, 
assuming they act solely in their own financial interest.
-/
theorem max_bribe_amount : 
  ∀ (bribe : ℤ), 
    bribe ≤ maxBribe ∧ 
    bribe > betAmount ∧
    (maxBribe - betAmount ≤ betAmount) ∧
    (∀ (x : ℤ), x > maxBribe → x - betAmount > betAmount) := by
  sorry


end NUMINAMATH_CALUDE_max_bribe_amount_l2998_299871


namespace NUMINAMATH_CALUDE_field_division_l2998_299815

theorem field_division (total_area smaller_area : ℝ) (h1 : total_area = 700) (h2 : smaller_area = 315) :
  ∃ (larger_area X : ℝ),
    larger_area + smaller_area = total_area ∧
    larger_area - smaller_area = (1 / 5) * X ∧
    X = 350 := by
  sorry

end NUMINAMATH_CALUDE_field_division_l2998_299815


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2998_299889

-- Define the equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1

-- State the theorem
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ioo (-3/2 : ℝ) (-1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2998_299889


namespace NUMINAMATH_CALUDE_special_function_characterization_l2998_299850

/-- A monotonic and invertible function from ℝ to ℝ satisfying f(x) + f⁻¹(x) = 2x for all x ∈ ℝ -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ Function.Bijective f ∧ ∀ x, f x + (Function.invFun f) x = 2 * x

/-- The theorem stating that any function satisfying SpecialFunction is of the form f(x) = x + c -/
theorem special_function_characterization (f : ℝ → ℝ) (h : SpecialFunction f) :
  ∃ c : ℝ, ∀ x, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_special_function_characterization_l2998_299850


namespace NUMINAMATH_CALUDE_quiz_score_ratio_l2998_299810

/-- Given a quiz taken by three people with specific scoring conditions,
    prove that the ratio of Tatuya's score to Ivanna's score is 2:1 -/
theorem quiz_score_ratio (tatuya_score ivanna_score dorothy_score : ℚ) : 
  dorothy_score = 90 →
  ivanna_score = (3/5) * dorothy_score →
  (tatuya_score + ivanna_score + dorothy_score) / 3 = 84 →
  tatuya_score / ivanna_score = 2 := by
sorry

end NUMINAMATH_CALUDE_quiz_score_ratio_l2998_299810


namespace NUMINAMATH_CALUDE_emily_quiz_score_l2998_299833

/-- Emily's quiz scores -/
def emily_scores : List ℕ := [85, 88, 90, 94, 96, 92]

/-- The required arithmetic mean -/
def required_mean : ℕ := 91

/-- The number of quizzes including the new one -/
def total_quizzes : ℕ := 7

/-- The score Emily needs on her seventh quiz -/
def seventh_score : ℕ := 92

theorem emily_quiz_score :
  (emily_scores.sum + seventh_score) / total_quizzes = required_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_quiz_score_l2998_299833


namespace NUMINAMATH_CALUDE_bacteria_growth_problem_l2998_299826

/-- Bacteria growth problem -/
theorem bacteria_growth_problem (initial_count : ℕ) : 
  (∀ (period : ℕ), initial_count * (4 ^ period) = initial_count * 4 ^ period) →
  initial_count * 4 ^ 4 = 262144 →
  initial_count = 1024 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_problem_l2998_299826


namespace NUMINAMATH_CALUDE_floor_product_equals_49_l2998_299819

theorem floor_product_equals_49 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 49 ↔ 7 ≤ x ∧ x < 50 / 7 := by
sorry

end NUMINAMATH_CALUDE_floor_product_equals_49_l2998_299819


namespace NUMINAMATH_CALUDE_isosceles_triangle_property_l2998_299875

/-- Represents a triangle with vertices A, B, C and incentre I -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  I : ℝ × ℝ

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The squared distance between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ := sorry

/-- Check if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  distance t.A t.B = distance t.A t.C

/-- The distance from a point to a line defined by two points -/
def distanceToLine (p : ℝ × ℝ) (q r : ℝ × ℝ) : ℝ := sorry

/-- Theorem: In an isosceles triangle ABC with incentre I, 
    if AB = AC, AI = 3, and the distance from I to BC is 2, then BC² = 80 -/
theorem isosceles_triangle_property (t : Triangle) :
  isIsosceles t →
  distance t.A t.I = 3 →
  distanceToLine t.I t.B t.C = 2 →
  distanceSquared t.B t.C = 80 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_property_l2998_299875


namespace NUMINAMATH_CALUDE_algorithm_structure_logical_judgment_l2998_299877

-- Define the basic algorithm structures
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

-- Define a property for structures requiring logical judgment and different processing
def RequiresLogicalJudgment (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Conditional => true
  | AlgorithmStructure.Loop => true
  | _ => false

-- Theorem statement
theorem algorithm_structure_logical_judgment :
  ∀ (s : AlgorithmStructure),
    RequiresLogicalJudgment s ↔ (s = AlgorithmStructure.Conditional ∨ s = AlgorithmStructure.Loop) :=
by sorry

end NUMINAMATH_CALUDE_algorithm_structure_logical_judgment_l2998_299877


namespace NUMINAMATH_CALUDE_mini_cupcakes_count_l2998_299872

theorem mini_cupcakes_count (students : ℕ) (donut_holes : ℕ) (desserts_per_student : ℕ) :
  students = 13 →
  donut_holes = 12 →
  desserts_per_student = 2 →
  ∃ (mini_cupcakes : ℕ), 
    mini_cupcakes + donut_holes = students * desserts_per_student ∧
    mini_cupcakes = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_mini_cupcakes_count_l2998_299872


namespace NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l2998_299828

theorem no_numbers_satisfying_conditions : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 300 →
    (6 ∣ n ∧ 8 ∣ n) → (4 ∣ n ∨ 11 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l2998_299828


namespace NUMINAMATH_CALUDE_used_computer_cost_l2998_299846

/-- Proves the cost of each used computer given the conditions of the problem -/
theorem used_computer_cost
  (new_computer_cost : ℕ)
  (new_computer_lifespan : ℕ)
  (used_computer_lifespan : ℕ)
  (savings : ℕ)
  (h1 : new_computer_cost = 600)
  (h2 : new_computer_lifespan = 6)
  (h3 : used_computer_lifespan = 3)
  (h4 : 2 * used_computer_lifespan = new_computer_lifespan)
  (h5 : savings = 200)
  (h6 : ∃ (used_computer_cost : ℕ),
        new_computer_cost = 2 * used_computer_cost + savings) :
  ∃ (used_computer_cost : ℕ), used_computer_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_used_computer_cost_l2998_299846


namespace NUMINAMATH_CALUDE_circles_centers_form_rectangle_l2998_299874

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def inside (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 < (c2.radius - c1.radius)^2

def rectangle (a b c d : ℝ × ℝ) : Prop :=
  let ab := (b.1 - a.1, b.2 - a.2)
  let bc := (c.1 - b.1, c.2 - b.2)
  let cd := (d.1 - c.1, d.2 - c.2)
  let da := (a.1 - d.1, a.2 - d.2)
  ab.1 * bc.1 + ab.2 * bc.2 = 0 ∧
  bc.1 * cd.1 + bc.2 * cd.2 = 0 ∧
  cd.1 * da.1 + cd.2 * da.2 = 0 ∧
  da.1 * ab.1 + da.2 * ab.2 = 0 ∧
  ab.1^2 + ab.2^2 = cd.1^2 + cd.2^2 ∧
  bc.1^2 + bc.2^2 = da.1^2 + da.2^2

theorem circles_centers_form_rectangle 
  (C C1 C2 C3 C4 : Circle)
  (h1 : C.radius = 2)
  (h2 : C1.radius = 1)
  (h3 : C2.radius = 1)
  (h4 : tangent C1 C2)
  (h5 : inside C1 C)
  (h6 : inside C2 C)
  (h7 : inside C3 C)
  (h8 : inside C4 C)
  (h9 : tangent C3 C)
  (h10 : tangent C3 C1)
  (h11 : tangent C3 C2)
  (h12 : tangent C4 C)
  (h13 : tangent C4 C1)
  (h14 : tangent C4 C3)
  : rectangle C.center C1.center C3.center C4.center :=
sorry

end NUMINAMATH_CALUDE_circles_centers_form_rectangle_l2998_299874


namespace NUMINAMATH_CALUDE_bread_cost_is_30_cents_l2998_299814

/-- The cost of a sandwich in dollars -/
def sandwich_price : ℚ := 1.5

/-- The cost of a slice of ham in dollars -/
def ham_cost : ℚ := 0.25

/-- The cost of a slice of cheese in dollars -/
def cheese_cost : ℚ := 0.35

/-- The total cost to make a sandwich in dollars -/
def total_cost : ℚ := 0.9

/-- The number of slices of bread in a sandwich -/
def bread_slices : ℕ := 2

/-- Theorem: The cost of a slice of bread is $0.30 -/
theorem bread_cost_is_30_cents :
  (total_cost - ham_cost - cheese_cost) / bread_slices = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_is_30_cents_l2998_299814


namespace NUMINAMATH_CALUDE_max_log_value_l2998_299848

theorem max_log_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 4 * a - 2 * b + 25 * c = 0) : 
  (∀ x y z, x > 0 → y > 0 → z > 0 → 4 * x - 2 * y + 25 * z = 0 → 
    Real.log x + Real.log z - 2 * Real.log y ≤ Real.log a + Real.log c - 2 * Real.log b) ∧
  Real.log a + Real.log c - 2 * Real.log b = -2 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_max_log_value_l2998_299848


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l2998_299886

theorem power_function_not_through_origin (m : ℝ) : 
  (m = 1 ∨ m = 2) → 
  ∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^((m^2 - m - 2)/2) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l2998_299886


namespace NUMINAMATH_CALUDE_sin_neg_360_degrees_l2998_299818

theorem sin_neg_360_degrees : Real.sin (-(360 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_360_degrees_l2998_299818


namespace NUMINAMATH_CALUDE_steve_coins_value_l2998_299881

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the total value of coins given the number of nickels and dimes -/
def total_value (nickels dimes : ℕ) : ℕ :=
  nickels * nickel_value + dimes * dime_value

/-- Proves that given 2 nickels and 4 more dimes than nickels, the total value is 70 cents -/
theorem steve_coins_value : 
  ∀ (nickels : ℕ), nickels = 2 → total_value nickels (nickels + 4) = 70 := by
  sorry

end NUMINAMATH_CALUDE_steve_coins_value_l2998_299881


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_10_l2998_299866

theorem factorization_of_2x_squared_minus_10 :
  ∀ x : ℝ, 2 * x^2 - 10 = 2 * (x + Real.sqrt 5) * (x - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_10_l2998_299866


namespace NUMINAMATH_CALUDE_infinitely_many_primes_dividing_2_pow_plus_poly_l2998_299831

/-- A nonzero polynomial with integer coefficients -/
def nonzero_int_poly (P : ℕ → ℤ) : Prop :=
  ∃ n, P n ≠ 0 ∧ ∀ m, ∃ k : ℤ, P m = k

theorem infinitely_many_primes_dividing_2_pow_plus_poly 
  (P : ℕ → ℤ) (h : nonzero_int_poly P) :
  ∀ N : ℕ, ∃ q : ℕ, q > N ∧ Nat.Prime q ∧ 
    ∃ n : ℕ, (q : ℤ) ∣ (2^n : ℤ) + P n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_dividing_2_pow_plus_poly_l2998_299831


namespace NUMINAMATH_CALUDE_doll_count_l2998_299844

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

/-- The number of collector's edition dolls Ivy has -/
def ivy_collector_dolls : ℕ := 20

/-- The number of dolls Casey has -/
def casey_dolls : ℕ := 5 * ivy_collector_dolls

/-- The total number of dolls Dina, Ivy, and Casey have together -/
def total_dolls : ℕ := dina_dolls + ivy_dolls + casey_dolls

theorem doll_count : total_dolls = 190 ∧ 
  2 * ivy_dolls / 3 = ivy_collector_dolls := by
  sorry

end NUMINAMATH_CALUDE_doll_count_l2998_299844


namespace NUMINAMATH_CALUDE_range_of_a_l2998_299878

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_monotone_decreasing f (-2) 4)
  (h2 : f (a + 1) > f (2 * a)) :
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2998_299878


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l2998_299885

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 8x + 2 and y = (2c)x - 4 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 8 * x + 2 ↔ y = (2 * c) * x - 4) → c = 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l2998_299885


namespace NUMINAMATH_CALUDE_triangle_side_length_l2998_299865

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the conditions and conclusion about the triangle -/
theorem triangle_side_length (t : Triangle) 
  (h1 : 2 * Real.sin t.B = Real.sin t.A + Real.sin t.C)
  (h2 : Real.cos t.B = 3/5)
  (h3 : 1/2 * t.a * t.c * Real.sin t.B = 4) :
  t.b = 4 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2998_299865


namespace NUMINAMATH_CALUDE_inverse_function_decreasing_l2998_299842

theorem inverse_function_decreasing :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ > x₂ → (1 / x₁) < (1 / x₂) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_decreasing_l2998_299842


namespace NUMINAMATH_CALUDE_mixed_gender_selection_l2998_299859

def male_students : ℕ := 10
def female_students : ℕ := 6
def total_selection : ℕ := 3

def select_mixed_gender (m : ℕ) (f : ℕ) (total : ℕ) : ℕ :=
  Nat.choose m 1 * Nat.choose f 2 + Nat.choose m 2 * Nat.choose f 1

theorem mixed_gender_selection :
  select_mixed_gender male_students female_students total_selection = 420 := by
  sorry

end NUMINAMATH_CALUDE_mixed_gender_selection_l2998_299859
