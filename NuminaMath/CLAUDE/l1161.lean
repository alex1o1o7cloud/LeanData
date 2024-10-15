import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1161_116195

theorem complex_number_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := 3 - Complex.I * (a^2 + 1)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1161_116195


namespace NUMINAMATH_CALUDE_f_monotonicity_and_tangent_intersection_l1161_116140

/-- The function f(x) = x³ - x² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_tangent_intersection (a : ℝ) :
  (∀ x : ℝ, f' a x ≥ 0 → a ≥ 1/3) ∧
  (∃ t : ℝ, t * f' a 1 = f a 1 ∧ f a (-1) = -t * f' a (-1)) :=
sorry


end NUMINAMATH_CALUDE_f_monotonicity_and_tangent_intersection_l1161_116140


namespace NUMINAMATH_CALUDE_inequality_solution_l1161_116158

theorem inequality_solution (x : ℝ) : 
  (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -1) ∨ x > 5 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1161_116158


namespace NUMINAMATH_CALUDE_raffle_winnings_l1161_116188

theorem raffle_winnings (W : ℝ) (h1 : W > 0) (h2 : W / 2 - 2 + 114 = W) : 
  W - W / 2 - 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_raffle_winnings_l1161_116188


namespace NUMINAMATH_CALUDE_number_thought_of_l1161_116168

theorem number_thought_of : ∃ x : ℝ, (x / 4 + 9 = 15) ∧ (x = 24) := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l1161_116168


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1161_116125

-- Part 1: Equation solution
theorem equation_solution :
  ∃ x : ℝ, (2 * x / (x - 2) + 3 / (2 - x) = 1) ∧ (x = 1) := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∃ x : ℝ, (2 * x - 1 ≥ 3 * (x - 1)) ∧
           ((5 - x) / 2 < x + 3) ∧
           (-1/3 < x) ∧ (x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1161_116125


namespace NUMINAMATH_CALUDE_pen_collection_l1161_116150

theorem pen_collection (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) : 
  initial_pens = 25 →
  mike_pens = 22 →
  sharon_pens = 19 →
  2 * (initial_pens + mike_pens) - sharon_pens = 75 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_l1161_116150


namespace NUMINAMATH_CALUDE_fold_point_set_area_l1161_116135

/-- Triangle DEF with given side lengths and right angle -/
structure RightTriangle where
  de : ℝ
  df : ℝ
  angle_e_is_right : de^2 + ef^2 = df^2
  de_length : de = 24
  df_length : df = 48

/-- Set of fold points in the triangle -/
def FoldPointSet (t : RightTriangle) : Set (ℝ × ℝ) := sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: Area of fold point set -/
theorem fold_point_set_area (t : RightTriangle) :
  area (FoldPointSet t) = 156 * Real.pi - 144 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_fold_point_set_area_l1161_116135


namespace NUMINAMATH_CALUDE_second_class_size_l1161_116183

theorem second_class_size 
  (first_class_size : ℕ) 
  (first_class_avg : ℚ) 
  (second_class_avg : ℚ) 
  (total_avg : ℚ) 
  (h1 : first_class_size = 35)
  (h2 : first_class_avg = 40)
  (h3 : second_class_avg = 60)
  (h4 : total_avg = 51.25) :
  ∃ second_class_size : ℕ,
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size) = total_avg ∧
    second_class_size = 45 := by
  sorry


end NUMINAMATH_CALUDE_second_class_size_l1161_116183


namespace NUMINAMATH_CALUDE_sector_tangent_problem_l1161_116148

theorem sector_tangent_problem (θ φ : Real) (h1 : 0 < θ) (h2 : θ < 2 * Real.pi) : 
  (1/2 * θ * 4^2 = 2 * Real.pi) → (Real.tan (θ + φ) = 3) → Real.tan φ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_tangent_problem_l1161_116148


namespace NUMINAMATH_CALUDE_visibility_time_proof_l1161_116128

/-- The time when Jenny and Kenny become visible to each other after being blocked by a circular building -/
def visibilityTime (buildingRadius : ℝ) (pathDistance : ℝ) (jennySpeed : ℝ) (kennySpeed : ℝ) : ℝ :=
  120

theorem visibility_time_proof (buildingRadius : ℝ) (pathDistance : ℝ) (jennySpeed : ℝ) (kennySpeed : ℝ) 
    (h1 : buildingRadius = 60)
    (h2 : pathDistance = 240)
    (h3 : jennySpeed = 4)
    (h4 : kennySpeed = 2) :
  visibilityTime buildingRadius pathDistance jennySpeed kennySpeed = 120 :=
by
  sorry

#check visibility_time_proof

end NUMINAMATH_CALUDE_visibility_time_proof_l1161_116128


namespace NUMINAMATH_CALUDE_rectangle_width_l1161_116110

/-- Given a rectangular piece of metal with length 19 cm and perimeter 70 cm, 
    prove that its width is 16 cm. -/
theorem rectangle_width (length perimeter : ℝ) (h1 : length = 19) (h2 : perimeter = 70) :
  let width := (perimeter / 2) - length
  width = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l1161_116110


namespace NUMINAMATH_CALUDE_yoojung_initial_candies_l1161_116173

/-- The number of candies Yoojung gave to her older sister -/
def candies_to_older_sister : ℕ := 7

/-- The number of candies Yoojung gave to her younger sister -/
def candies_to_younger_sister : ℕ := 6

/-- The number of candies Yoojung had left over -/
def candies_left_over : ℕ := 15

/-- The initial number of candies Yoojung had -/
def initial_candies : ℕ := candies_to_older_sister + candies_to_younger_sister + candies_left_over

theorem yoojung_initial_candies : initial_candies = 28 := by
  sorry

end NUMINAMATH_CALUDE_yoojung_initial_candies_l1161_116173


namespace NUMINAMATH_CALUDE_fraction_problem_l1161_116197

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N * f^2 = 6^3 ∧ N * f^2 = 7776 → f = 1/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1161_116197


namespace NUMINAMATH_CALUDE_wills_earnings_after_deductions_l1161_116102

/-- Calculates Will's earnings after tax deductions for a 5-day work week --/
def willsEarnings (monday_wage monday_hours tuesday_wage tuesday_hours
                   wednesday_wage wednesday_hours thursday_wage thursday_hours
                   friday_wage friday_hours tax_rate : ℝ) : ℝ :=
  let total_earnings := monday_wage * monday_hours +
                        tuesday_wage * tuesday_hours +
                        wednesday_wage * wednesday_hours +
                        thursday_wage * thursday_hours +
                        friday_wage * friday_hours
  let tax_deduction := total_earnings * tax_rate
  total_earnings - tax_deduction

/-- Theorem stating Will's earnings after deductions --/
theorem wills_earnings_after_deductions :
  willsEarnings 8 8 10 2 9 6 7 4 7 4 0.12 = 170.72 := by
  sorry

end NUMINAMATH_CALUDE_wills_earnings_after_deductions_l1161_116102


namespace NUMINAMATH_CALUDE_seashells_given_away_l1161_116161

/-- Represents the number of seashells Maura collected and gave away -/
structure SeashellCollection where
  total : ℕ
  left : ℕ
  given : ℕ

/-- Theorem stating that the number of seashells given away is the difference between total and left -/
theorem seashells_given_away (collection : SeashellCollection) 
  (h1 : collection.total = 75)
  (h2 : collection.left = 57)
  (h3 : collection.given = collection.total - collection.left) :
  collection.given = 18 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_away_l1161_116161


namespace NUMINAMATH_CALUDE_triangle_properties_l1161_116156

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c * Real.sin B + (a + c^2 / a - b^2 / a) * Real.sin C = 2 * c * Real.sin A →
  (a * b * Real.sin C) / 2 = Real.sqrt 3 →
  a * Real.sin A / 2 = Real.sqrt 3 →
  b * Real.sin B / 2 = Real.sqrt 3 →
  c * Real.sin C / 2 = Real.sqrt 3 →
  C = π / 3 ∧ Real.cos (2 * A) - 2 * Real.sin B ^ 2 + 1 = -1 / 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l1161_116156


namespace NUMINAMATH_CALUDE_ball_probabilities_l1161_116170

structure Bag where
  red_balls : ℕ
  blue_balls : ℕ
  red_ones : ℕ
  blue_ones : ℕ

def total_balls (b : Bag) : ℕ := b.red_balls + b.blue_balls

def prob_one_red_sum_three (b : Bag) : ℚ := 16 / 81

def prob_first_red (b : Bag) : ℚ := b.red_balls / (total_balls b)

def prob_second_one (b : Bag) : ℚ := 1 / 3

theorem ball_probabilities (b : Bag) 
  (h1 : b.red_balls = 6) 
  (h2 : b.blue_balls = 3) 
  (h3 : b.red_ones = 2) 
  (h4 : b.blue_ones = 1) :
  prob_one_red_sum_three b = 16 / 81 ∧ 
  prob_first_red b = 2 / 3 ∧
  prob_second_one b = 1 / 3 ∧
  prob_first_red b * prob_second_one b = 
    (b.red_ones / (total_balls b)) * ((b.blue_ones) / (total_balls b - 1)) +
    ((b.red_balls - b.red_ones) / (total_balls b)) * (b.red_ones / (total_balls b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l1161_116170


namespace NUMINAMATH_CALUDE_local_science_students_percentage_l1161_116113

/-- Proves that the percentage of local science students is 25% given the conditions of the problem -/
theorem local_science_students_percentage 
  (total_arts : ℕ) 
  (total_science : ℕ) 
  (total_commerce : ℕ) 
  (local_arts_percentage : ℚ) 
  (local_commerce_percentage : ℚ) 
  (total_local_percentage : ℚ) 
  (h1 : total_arts = 400) 
  (h2 : total_science = 100) 
  (h3 : total_commerce = 120) 
  (h4 : local_arts_percentage = 1/2) 
  (h5 : local_commerce_percentage = 17/20) 
  (h6 : total_local_percentage = 327/100) : 
  ∃ (local_science_percentage : ℚ), 
    local_science_percentage = 1/4 ∧ 
    (local_arts_percentage * total_arts + local_science_percentage * total_science + local_commerce_percentage * total_commerce) / (total_arts + total_science + total_commerce) = total_local_percentage := by
  sorry


end NUMINAMATH_CALUDE_local_science_students_percentage_l1161_116113


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_101110111_base_5_l1161_116166

def base_five_to_decimal (n : ℕ) : ℕ := 
  5^8 + 5^6 + 5^5 + 5^4 + 5^3 + 5^2 + 5^1 + 5^0

theorem largest_prime_divisor_of_101110111_base_5 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ base_five_to_decimal 101110111 ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ base_five_to_decimal 101110111 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_101110111_base_5_l1161_116166


namespace NUMINAMATH_CALUDE_jane_egg_income_l1161_116177

/-- Calculates the income from selling eggs given the number of chickens, eggs per chicken per week, 
    price per dozen eggs, and number of weeks. -/
def egg_income (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  (num_chickens * eggs_per_chicken * num_weeks : ℚ) / 12 * price_per_dozen

/-- Proves that Jane's income from selling eggs in 2 weeks is $20. -/
theorem jane_egg_income :
  egg_income 10 6 2 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jane_egg_income_l1161_116177


namespace NUMINAMATH_CALUDE_sam_apple_consumption_l1161_116105

/-- Calculates the number of apples Sam eats in a week -/
def apples_eaten_in_week (apples_per_sandwich : ℕ) (sandwiches_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  apples_per_sandwich * sandwiches_per_day * days_in_week

/-- Proves that Sam eats 280 apples in one week -/
theorem sam_apple_consumption : apples_eaten_in_week 4 10 7 = 280 := by
  sorry

#eval apples_eaten_in_week 4 10 7

end NUMINAMATH_CALUDE_sam_apple_consumption_l1161_116105


namespace NUMINAMATH_CALUDE_oil_leak_height_l1161_116106

/-- The speed of oil leaking from a circular cylinder -/
def leak_speed (k : ℝ) (h : ℝ) : ℝ := k * h^2

theorem oil_leak_height (k : ℝ) (h' : ℝ) :
  (k > 0) →
  (leak_speed k 12 = 9 * leak_speed k h') →
  h' = 4 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_height_l1161_116106


namespace NUMINAMATH_CALUDE_dave_tickets_proof_l1161_116114

/-- Represents the number of tickets Dave won initially -/
def initial_tickets : ℕ := 25

/-- Represents the number of tickets spent on a beanie -/
def spent_tickets : ℕ := 22

/-- Represents the number of additional tickets won -/
def additional_tickets : ℕ := 15

/-- Represents the final number of tickets Dave has -/
def final_tickets : ℕ := 18

/-- Proves that the initial number of tickets is correct given the problem conditions -/
theorem dave_tickets_proof :
  initial_tickets - spent_tickets + additional_tickets = final_tickets :=
by sorry

end NUMINAMATH_CALUDE_dave_tickets_proof_l1161_116114


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l1161_116118

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  (a + b + c) / 3 = 20 →
  max a (max b c) = 24 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l1161_116118


namespace NUMINAMATH_CALUDE_plane_division_theorem_l1161_116116

/-- A line in the plane --/
structure Line where
  -- We don't need to define the actual properties of a line for this statement

/-- A set of lines in the plane --/
def LineSet := Set Line

/-- Predicate to check if all lines in a set are parallel to one of them --/
def allParallel (ls : LineSet) : Prop := sorry

/-- Number of regions formed by a set of lines --/
def numRegions (ls : LineSet) : ℕ := sorry

/-- Statement of the theorem --/
theorem plane_division_theorem :
  ∃ (k₀ : ℕ), ∀ (k : ℕ), k > k₀ →
    ∃ (ls : LineSet), ls.Finite ∧ ¬allParallel ls ∧ numRegions ls = k :=
by
  -- Let k₀ = 5
  use 5
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_plane_division_theorem_l1161_116116


namespace NUMINAMATH_CALUDE_complex_addition_l1161_116194

theorem complex_addition : (1 : ℂ) + 3*I + (2 : ℂ) - 4*I = 3 - I := by sorry

end NUMINAMATH_CALUDE_complex_addition_l1161_116194


namespace NUMINAMATH_CALUDE_viewing_angle_midpoint_l1161_116162

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the viewing angle function
noncomputable def viewingAngle (c : Circle) (p : Point) : ℝ := sorry

-- Define the line AB
def lineAB (A B : Point) : Set Point := sorry

-- Theorem statement
theorem viewing_angle_midpoint (O : Circle) (A B : Point) :
  let α := viewingAngle O A
  let β := viewingAngle O B
  let γ := (α + β) / 2
  ∃ (C₁ C₂ : Point), C₁ ∈ lineAB A B ∧ C₂ ∈ lineAB A B ∧
    viewingAngle O C₁ = γ ∧ viewingAngle O C₂ = γ ∧
    (α = β → (C₁ = A ∧ C₂ = B) ∨ (C₁ = B ∧ C₂ = A)) :=
by sorry


end NUMINAMATH_CALUDE_viewing_angle_midpoint_l1161_116162


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l1161_116127

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ t : ℝ, x t * y t = c

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 5 = 40 →
  x 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l1161_116127


namespace NUMINAMATH_CALUDE_product_less_than_sum_plus_one_l1161_116165

theorem product_less_than_sum_plus_one (a₁ a₂ : ℝ) 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ < a₁ + a₂ + 1 := by
  sorry

#check product_less_than_sum_plus_one

end NUMINAMATH_CALUDE_product_less_than_sum_plus_one_l1161_116165


namespace NUMINAMATH_CALUDE_square_divides_power_plus_one_l1161_116133

theorem square_divides_power_plus_one (n : ℕ) : n^2 ∣ 2^n + 1 ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_divides_power_plus_one_l1161_116133


namespace NUMINAMATH_CALUDE_negation_of_all_x_squared_plus_one_positive_l1161_116155

theorem negation_of_all_x_squared_plus_one_positive :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_x_squared_plus_one_positive_l1161_116155


namespace NUMINAMATH_CALUDE_arrangements_without_A_at_head_l1161_116164

def total_people : Nat := 5
def people_to_select : Nat := 3

def total_arrangements : Nat := total_people * (total_people - 1) * (total_people - 2)
def arrangements_with_A_at_head : Nat := (total_people - 1) * (total_people - 2)

theorem arrangements_without_A_at_head :
  total_arrangements - arrangements_with_A_at_head = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_without_A_at_head_l1161_116164


namespace NUMINAMATH_CALUDE_next_square_property_number_l1161_116104

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_square_property (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens_ones := n % 100
  is_perfect_square (hundreds * tens_ones)

theorem next_square_property_number :
  ∀ n : ℕ,
    1818 < n →
    n < 10000 →
    has_square_property n →
    (∀ m : ℕ, 1818 < m → m < n → ¬has_square_property m) →
    n = 1832 :=
sorry

end NUMINAMATH_CALUDE_next_square_property_number_l1161_116104


namespace NUMINAMATH_CALUDE_triangle_angle_A_l1161_116153

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem statement
theorem triangle_angle_A (t : Triangle) :
  t.a = 3 ∧ t.b = 24/5 ∧ Real.cos t.B = 3/5 → t.A = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l1161_116153


namespace NUMINAMATH_CALUDE_ABC_reflection_collinear_l1161_116154

-- Define the basic structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

-- Define the triangle ABC
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

-- Define point P and line γ
def P : Point := sorry
def γ : Line := sorry

-- Define the reflection of a line with respect to another line
def reflect (l₁ l₂ : Line) : Line := sorry

-- Define the intersection of two lines
def intersect (l₁ l₂ : Line) : Point := sorry

-- Define lines PA, PB, PC
def PA : Line := sorry
def PB : Line := sorry
def PC : Line := sorry

-- Define lines BC, AC, AB
def BC : Line := sorry
def AC : Line := sorry
def AB : Line := sorry

-- Define points A', B', C'
def A' : Point := intersect (reflect PA γ) BC
def B' : Point := intersect (reflect PB γ) AC
def C' : Point := intersect (reflect PC γ) AB

-- Define collinearity
def collinear (p q r : Point) : Prop := sorry

-- The theorem to be proved
theorem ABC_reflection_collinear : collinear A' B' C' := by sorry

end NUMINAMATH_CALUDE_ABC_reflection_collinear_l1161_116154


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1161_116176

theorem final_sum_after_operations (a b S : ℝ) : 
  a + b = S → 3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1161_116176


namespace NUMINAMATH_CALUDE_problem_solution_l1161_116117

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 7) : 
  (x + y)^2 - x*y = 1183/36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1161_116117


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l1161_116139

theorem min_value_product_quotient (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k ≥ 2) :
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x*y*z) ≥ (2+k)^3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l1161_116139


namespace NUMINAMATH_CALUDE_cuboid_height_l1161_116191

theorem cuboid_height (volume : ℝ) (base_area : ℝ) (height : ℝ) 
  (h1 : volume = 144)
  (h2 : base_area = 18)
  (h3 : volume = base_area * height) :
  height = 8 := by
sorry

end NUMINAMATH_CALUDE_cuboid_height_l1161_116191


namespace NUMINAMATH_CALUDE_half_coverage_days_l1161_116143

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 48

/-- Represents the growth factor of the lily pad patch per day -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days required to cover half the lake
    is one day less than the number of days required to cover the full lake -/
theorem half_coverage_days : 
  full_coverage_days - 1 = full_coverage_days - (daily_growth_factor.log 2) := by
  sorry

end NUMINAMATH_CALUDE_half_coverage_days_l1161_116143


namespace NUMINAMATH_CALUDE_octal_minus_base9_equals_19559_l1161_116134

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem octal_minus_base9_equals_19559 : 
  let octal := [5, 4, 3, 2, 1]
  let base9 := [4, 3, 2, 1]
  base_to_decimal octal 8 - base_to_decimal base9 9 = 19559 := by
  sorry

end NUMINAMATH_CALUDE_octal_minus_base9_equals_19559_l1161_116134


namespace NUMINAMATH_CALUDE_square_side_increase_l1161_116129

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.21 → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l1161_116129


namespace NUMINAMATH_CALUDE_proposition_relation_l1161_116196

theorem proposition_relation (a b : ℝ) : 
  (∃ a b : ℝ, |a - b| < 3 ∧ (|a| ≥ 1 ∨ |b| ≥ 2)) ∧
  (∀ a b : ℝ, |a| < 1 ∧ |b| < 2 → |a - b| < 3) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_l1161_116196


namespace NUMINAMATH_CALUDE_inequality_proof_l1161_116121

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x * y + y * z + z * x = 6) : 
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z))) + 
  (1 / (2 * Real.sqrt 2 + y^2 * (x + z))) + 
  (1 / (2 * Real.sqrt 2 + z^2 * (x + y))) ≤ 
  1 / (x * y * z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1161_116121


namespace NUMINAMATH_CALUDE_triangle_side_length_l1161_116193

theorem triangle_side_length (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →  -- positive side lengths
  (|a - 7| + (b - 2)^2 = 0) →    -- given equation
  (∃ n : ℕ, c = 2*n + 1) →      -- c is odd
  (a + b > c) →                 -- triangle inequality
  (a + c > b) →                 -- triangle inequality
  (b + c > a) →                 -- triangle inequality
  (c = 7) :=                    -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1161_116193


namespace NUMINAMATH_CALUDE_tetrahedron_section_theorem_l1161_116126

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane defined by three points -/
structure Plane where
  P : Point3D
  Q : Point3D
  R : Point3D

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point3D) (A : Point3D) (D : Point3D) : Prop :=
  M.x = (A.x + D.x) / 2 ∧ M.y = (A.y + D.y) / 2 ∧ M.z = (A.z + D.z) / 2

/-- Check if a point is on the extension of a line segment -/
def isOnExtension (N : Point3D) (A : Point3D) (B : Point3D) : Prop :=
  ∃ t : ℝ, t > 1 ∧ N.x = A.x + t * (B.x - A.x) ∧
                 N.y = A.y + t * (B.y - A.y) ∧
                 N.z = A.z + t * (B.z - A.z)

/-- Calculate the ratio in which a plane divides a line segment -/
def divisionRatio (P : Plane) (A : Point3D) (B : Point3D) : ℝ × ℝ :=
  sorry

theorem tetrahedron_section_theorem (ABCD : Tetrahedron) (M N K : Point3D) :
  isMidpoint M ABCD.A ABCD.D →
  isOnExtension N ABCD.A ABCD.B →
  isOnExtension K ABCD.A ABCD.C →
  (N.x - ABCD.B.x)^2 + (N.y - ABCD.B.y)^2 + (N.z - ABCD.B.z)^2 =
    (ABCD.B.x - ABCD.A.x)^2 + (ABCD.B.y - ABCD.A.y)^2 + (ABCD.B.z - ABCD.A.z)^2 →
  (K.x - ABCD.C.x)^2 + (K.y - ABCD.C.y)^2 + (K.z - ABCD.C.z)^2 =
    4 * ((ABCD.C.x - ABCD.A.x)^2 + (ABCD.C.y - ABCD.A.y)^2 + (ABCD.C.z - ABCD.A.z)^2) →
  let P : Plane := {P := M, Q := N, R := K}
  divisionRatio P ABCD.D ABCD.B = (2, 1) ∧
  divisionRatio P ABCD.D ABCD.C = (3, 2) :=
by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_section_theorem_l1161_116126


namespace NUMINAMATH_CALUDE_daily_wage_calculation_l1161_116163

def days_in_week : ℕ := 7
def weeks : ℕ := 6
def total_earnings : ℕ := 2646

theorem daily_wage_calculation (days_worked : ℕ) (daily_wage : ℚ) 
  (h1 : days_worked = days_in_week * weeks)
  (h2 : daily_wage * days_worked = total_earnings) :
  daily_wage = 63 := by sorry

end NUMINAMATH_CALUDE_daily_wage_calculation_l1161_116163


namespace NUMINAMATH_CALUDE_remainder_proof_l1161_116171

theorem remainder_proof : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1161_116171


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1161_116115

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function for the general term of the expansion
def generalTerm (r : ℕ) : ℤ :=
  (-1)^r * 2^(4 - r) * binomial 4 r

-- Theorem statement
theorem constant_term_expansion :
  generalTerm 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1161_116115


namespace NUMINAMATH_CALUDE_exactly_one_integer_satisfies_condition_l1161_116136

theorem exactly_one_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n ≥ 15 := by sorry

end NUMINAMATH_CALUDE_exactly_one_integer_satisfies_condition_l1161_116136


namespace NUMINAMATH_CALUDE_unique_fifth_power_solution_l1161_116112

theorem unique_fifth_power_solution :
  ∀ x y : ℕ, x^5 = y^5 + 10*y^2 + 20*y + 1 → (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_fifth_power_solution_l1161_116112


namespace NUMINAMATH_CALUDE_equal_benefit_credit_debit_l1161_116181

/-- Represents the benefit of using a card for a purchase -/
structure CardBenefit where
  purchase_amount : ℝ
  cashback_rate : ℝ
  interest_rate : ℝ

/-- Calculates the net benefit of using a card after one month -/
def net_benefit (card : CardBenefit) : ℝ :=
  card.purchase_amount * card.cashback_rate + card.purchase_amount * card.interest_rate

/-- The purchase amount in rubles -/
def purchase_amount : ℝ := 10000

/-- Theorem stating that the net benefit is equal for both credit and debit cards -/
theorem equal_benefit_credit_debit :
  let credit_card := CardBenefit.mk purchase_amount 0.005 0.005
  let debit_card := CardBenefit.mk purchase_amount 0.01 0
  net_benefit credit_card = net_benefit debit_card :=
by sorry

end NUMINAMATH_CALUDE_equal_benefit_credit_debit_l1161_116181


namespace NUMINAMATH_CALUDE_quadratic_two_roots_range_l1161_116192

theorem quadratic_two_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1/4 = 0 ∧ y^2 + m*y + 1/4 = 0) ↔ 
  (m < -1 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_range_l1161_116192


namespace NUMINAMATH_CALUDE_equation_has_29_solutions_l1161_116131

/-- The number of real solutions to the equation x/50 = sin x -/
def num_solutions : ℕ := 29

/-- The equation we're considering -/
def equation (x : ℝ) : Prop := x / 50 = Real.sin x

theorem equation_has_29_solutions :
  ∃! (s : Set ℝ), (∀ x ∈ s, equation x) ∧ Finite s ∧ Nat.card s = num_solutions :=
sorry

end NUMINAMATH_CALUDE_equation_has_29_solutions_l1161_116131


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1161_116145

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0 → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1161_116145


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1161_116169

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ), 8*x^4 + 7*x^3 + 3*x^2 - 5*x - 8 = (x - 1) * (8*x^3 + 15*x^2 + 18*x + 13) + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1161_116169


namespace NUMINAMATH_CALUDE_bird_count_after_changes_l1161_116187

/-- Represents the number of birds of each type on the fence -/
structure BirdCount where
  sparrows : ℕ
  storks : ℕ
  pigeons : ℕ
  swallows : ℕ

/-- Calculates the total number of birds -/
def totalBirds (birds : BirdCount) : ℕ :=
  birds.sparrows + birds.storks + birds.pigeons + birds.swallows

/-- Represents the changes in bird population -/
structure BirdChanges where
  sparrowsJoined : ℕ
  swallowsJoined : ℕ
  pigeonsLeft : ℕ

/-- Applies changes to the bird population -/
def applyChanges (initial : BirdCount) (changes : BirdChanges) : BirdCount :=
  { sparrows := initial.sparrows + changes.sparrowsJoined,
    storks := initial.storks,
    pigeons := initial.pigeons - changes.pigeonsLeft,
    swallows := initial.swallows + changes.swallowsJoined }

theorem bird_count_after_changes 
  (initial : BirdCount)
  (changes : BirdChanges)
  (h_initial : initial = { sparrows := 3, storks := 2, pigeons := 4, swallows := 0 })
  (h_changes : changes = { sparrowsJoined := 3, swallowsJoined := 5, pigeonsLeft := 2 }) :
  totalBirds (applyChanges initial changes) = 15 := by
  sorry


end NUMINAMATH_CALUDE_bird_count_after_changes_l1161_116187


namespace NUMINAMATH_CALUDE_house_numbering_counts_l1161_116184

/-- Count of 9s in house numbers from 1 to n -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Total count of digits used in house numbers from 1 to n -/
def total_digits (n : ℕ) : ℕ := sorry

theorem house_numbering_counts :
  (count_nines 100 = 10) ∧ (total_digits 100 = 192) := by sorry

end NUMINAMATH_CALUDE_house_numbering_counts_l1161_116184


namespace NUMINAMATH_CALUDE_power_multiplication_l1161_116122

theorem power_multiplication (x : ℝ) : x^3 * x^4 = x^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1161_116122


namespace NUMINAMATH_CALUDE_sequence_properties_l1161_116103

def a (n : ℤ) : ℤ := 30 + n - n^2

theorem sequence_properties :
  (a 10 = -60) ∧
  (∀ n : ℤ, a n = 0 ↔ n = 6) ∧
  (∀ n : ℤ, a n > 0 ↔ n > 6) ∧
  (∀ n : ℤ, a n < 0 ↔ n < 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1161_116103


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3_l1161_116174

theorem unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3 :
  ∃! n : ℕ+, 20 ∣ n ∧ (8.2 : ℝ) < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 8.3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3_l1161_116174


namespace NUMINAMATH_CALUDE_domino_placement_theorem_l1161_116151

/-- Represents a chessboard with dimensions n x n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a domino with dimensions 1 x 2 -/
structure Domino where

/-- Represents a position on the chessboard -/
structure Position where
  x : ℝ
  y : ℝ

/-- Checks if a position is strictly within the chessboard boundaries -/
def Position.isWithinBoard (p : Position) (b : Chessboard n) : Prop :=
  0 < p.x ∧ p.x < n ∧ 0 < p.y ∧ p.y < n

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement (b : Chessboard n) where
  center : Position
  isValid : center.isWithinBoard b

/-- Represents a configuration of domino placements on the chessboard -/
def Configuration (b : Chessboard n) := List (DominoPlacement b)

/-- Counts the number of dominoes in a configuration -/
def countDominoes (config : Configuration b) : ℕ := config.length

theorem domino_placement_theorem (b : Chessboard 8) :
  (∃ config : Configuration b, countDominoes config ≥ 40) ∧
  (∃ config : Configuration b, countDominoes config ≥ 41) ∧
  (∃ config : Configuration b, countDominoes config > 41) := by
  sorry

end NUMINAMATH_CALUDE_domino_placement_theorem_l1161_116151


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1161_116180

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1161_116180


namespace NUMINAMATH_CALUDE_range_of_a_l1161_116189

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 5 → x^2 - 2*x + a ≥ 0) ↔ a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1161_116189


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1161_116152

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d > 0 →  -- positive common difference
  a 1 + a 7 = 10 →  -- sum of roots condition
  a 1 * a 7 = 16 →  -- product of roots condition
  a 2 + a 4 + a 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1161_116152


namespace NUMINAMATH_CALUDE_ab_value_l1161_116199

theorem ab_value (a b : ℝ) 
  (h1 : (a + b)^2 + |b + 5| = b + 5)
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1161_116199


namespace NUMINAMATH_CALUDE_coin_flip_problem_l1161_116178

theorem coin_flip_problem : ∃ (n : ℕ+) (a b : ℕ),
  a + b = n ∧
  4 + 8 * a - 3 * b = 1 + 3 * 2^(a - b) ∧
  (4 + 8 * a - 3 * b : ℤ) < 2012 ∧
  n = 137 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l1161_116178


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l1161_116120

theorem seed_germination_percentage
  (seeds_plot1 : ℕ)
  (seeds_plot2 : ℕ)
  (germination_rate_plot1 : ℚ)
  (germination_rate_plot2 : ℚ)
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 15 / 100)
  (h4 : germination_rate_plot2 = 35 / 100)
  : (((seeds_plot1 * germination_rate_plot1 + seeds_plot2 * germination_rate_plot2) / (seeds_plot1 + seeds_plot2)) : ℚ) = 23 / 100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l1161_116120


namespace NUMINAMATH_CALUDE_milk_savings_l1161_116141

-- Define the problem parameters
def gallons : ℕ := 8
def original_price : ℚ := 3.20
def discount_rate : ℚ := 0.25

-- Define the function to calculate savings
def calculate_savings (g : ℕ) (p : ℚ) (d : ℚ) : ℚ :=
  g * p * d

-- Theorem statement
theorem milk_savings :
  calculate_savings gallons original_price discount_rate = 6.40 := by
  sorry


end NUMINAMATH_CALUDE_milk_savings_l1161_116141


namespace NUMINAMATH_CALUDE_triangle_properties_l1161_116109

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (abc : Triangle) (R : Real) :
  2 * Real.sqrt 3 * (Real.sin (abc.A / 2))^2 + Real.sin abc.A - Real.sqrt 3 = 0 →
  (1/2) * abc.b * abc.c * Real.sin abc.A = Real.sqrt 3 →
  R = Real.sqrt 3 →
  abc.A = π/3 ∧ abc.a + abc.b + abc.c = 3 + Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1161_116109


namespace NUMINAMATH_CALUDE_box_long_side_length_l1161_116138

/-- The length of the long sides of a box, given its dimensions and total velvet needed. -/
theorem box_long_side_length (total_velvet : ℝ) (short_side_length short_side_width : ℝ) 
  (long_side_width : ℝ) (top_bottom_area : ℝ) :
  total_velvet = 236 ∧
  short_side_length = 5 ∧
  short_side_width = 6 ∧
  long_side_width = 6 ∧
  top_bottom_area = 40 →
  ∃ long_side_length : ℝ,
    long_side_length = 8 ∧
    total_velvet = 2 * (short_side_length * short_side_width) + 
                   2 * top_bottom_area + 
                   2 * (long_side_length * long_side_width) :=
by sorry

end NUMINAMATH_CALUDE_box_long_side_length_l1161_116138


namespace NUMINAMATH_CALUDE_darnel_workout_l1161_116124

/-- Darnel's sprinting distances in miles -/
def sprint_distances : List ℝ := [0.8932, 0.7773, 0.9539, 0.5417, 0.6843]

/-- Darnel's jogging distances in miles -/
def jog_distances : List ℝ := [0.7683, 0.4231, 0.5733, 0.625, 0.6549]

/-- The difference between Darnel's total sprinting distance and total jogging distance -/
def sprint_jog_difference : ℝ := sprint_distances.sum - jog_distances.sum

theorem darnel_workout :
  sprint_jog_difference = 0.8058 := by sorry

end NUMINAMATH_CALUDE_darnel_workout_l1161_116124


namespace NUMINAMATH_CALUDE_at_least_one_red_certain_l1161_116157

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of balls drawn
def drawn_balls : ℕ := 4

-- Theorem statement
theorem at_least_one_red_certain :
  ∀ (draw : Finset ℕ),
  draw.card = drawn_balls →
  draw ⊆ Finset.range total_balls →
  ∃ (x : ℕ), x ∈ draw ∧ x < red_balls :=
sorry

end NUMINAMATH_CALUDE_at_least_one_red_certain_l1161_116157


namespace NUMINAMATH_CALUDE_indeterminate_equation_solutions_l1161_116179

def solution_set : Set (ℤ × ℤ) := {(3, -1), (5, 1), (1, 5), (-1, 3)}

theorem indeterminate_equation_solutions :
  {(x, y) : ℤ × ℤ | 2 * (x + y) = x * y + 7} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_equation_solutions_l1161_116179


namespace NUMINAMATH_CALUDE_books_not_sold_percentage_l1161_116107

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem books_not_sold_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_not_sold - 80.57| < ε :=
sorry

end NUMINAMATH_CALUDE_books_not_sold_percentage_l1161_116107


namespace NUMINAMATH_CALUDE_largest_b_value_l1161_116130

/-- The polynomial function representing the equation -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^4 - a*x^3 - b*x^2 - c*x - 2007

/-- Predicate to check if a number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Predicate to check if the equation has exactly three distinct integer solutions -/
def hasThreeDistinctIntegerSolutions (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    isInteger x ∧ isInteger y ∧ isInteger z ∧
    f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0 ∧
    ∀ w : ℝ, f a b c w = 0 → w = x ∨ w = y ∨ w = z

/-- The main theorem -/
theorem largest_b_value :
  ∃ b_max : ℝ, (∀ a c b : ℝ, hasThreeDistinctIntegerSolutions a b c → b ≤ b_max) ∧
    (∃ a c : ℝ, hasThreeDistinctIntegerSolutions a b_max c) ∧
    b_max = 3343 := by sorry

end NUMINAMATH_CALUDE_largest_b_value_l1161_116130


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1161_116149

/-- The number of schools --/
def num_schools : ℕ := 4

/-- The number of members per school --/
def members_per_school : ℕ := 5

/-- The number of representatives from the host school --/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school --/
def non_host_representatives : ℕ := 1

/-- The total number of ways to arrange a presidency meeting --/
def meeting_arrangements : ℕ := num_schools * (Nat.choose members_per_school host_representatives) * (Nat.choose members_per_school non_host_representatives)^(num_schools - 1)

theorem presidency_meeting_arrangements :
  meeting_arrangements = 5000 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1161_116149


namespace NUMINAMATH_CALUDE_product_equation_sum_l1161_116147

theorem product_equation_sum (p q r s : ℤ) : 
  (∀ x, (x^2 + p*x + q) * (x^2 + r*x + s) = x^4 - x^3 + 3*x^2 - 4*x + 4) →
  p + q + r + s = -1 := by
sorry

end NUMINAMATH_CALUDE_product_equation_sum_l1161_116147


namespace NUMINAMATH_CALUDE_equation_solution_l1161_116190

theorem equation_solution : ∃ x : ℝ, x * 15 - x * (2/3) + 1.4 = 10 ∧ x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1161_116190


namespace NUMINAMATH_CALUDE_gabriel_jaxon_toy_ratio_l1161_116132

theorem gabriel_jaxon_toy_ratio :
  ∀ (g j x : ℕ),
  j = g + 8 →
  x = 15 →
  g + j + x = 83 →
  g = 2 * x :=
by
  sorry

end NUMINAMATH_CALUDE_gabriel_jaxon_toy_ratio_l1161_116132


namespace NUMINAMATH_CALUDE_least_five_digit_multiple_l1161_116111

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem least_five_digit_multiple : ∃ (n : ℕ),
  n = 21000 ∧
  n ≥ 10000 ∧ n < 100000 ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < n →
    ¬(is_divisible_by m 15 ∧
      is_divisible_by m 25 ∧
      is_divisible_by m 40 ∧
      is_divisible_by m 75 ∧
      is_divisible_by m 125 ∧
      is_divisible_by m 140)) ∧
  is_divisible_by n 15 ∧
  is_divisible_by n 25 ∧
  is_divisible_by n 40 ∧
  is_divisible_by n 75 ∧
  is_divisible_by n 125 ∧
  is_divisible_by n 140 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_multiple_l1161_116111


namespace NUMINAMATH_CALUDE_chip_notebook_packs_l1161_116123

/-- The number of packs of notebook paper Chip will use after 6 weeks -/
def notebook_packs_used (pages_per_class_per_day : ℕ) (num_classes : ℕ) 
  (days_per_week : ℕ) (sheets_per_pack : ℕ) (num_weeks : ℕ) : ℕ :=
  (pages_per_class_per_day * num_classes * days_per_week * num_weeks) / sheets_per_pack

/-- Theorem stating the number of packs Chip will use -/
theorem chip_notebook_packs : 
  notebook_packs_used 2 5 5 100 6 = 3 := by sorry

end NUMINAMATH_CALUDE_chip_notebook_packs_l1161_116123


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l1161_116137

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (Real.pi + α) = 2/3) : 
  Real.cos (2 * α) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l1161_116137


namespace NUMINAMATH_CALUDE_system_solution_l1161_116144

theorem system_solution :
  ∀ x y z : ℝ,
  (x + y - 2 + 4*x*y = 0 ∧
   y + z - 2 + 4*y*z = 0 ∧
   z + x - 2 + 4*z*x = 0) ↔
  ((x = -1 ∧ y = -1 ∧ z = -1) ∨
   (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1161_116144


namespace NUMINAMATH_CALUDE_three_digit_multiple_of_2_3_5_l1161_116108

theorem three_digit_multiple_of_2_3_5 (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ 
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n →
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m → 120 ≤ m) ∧
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m → m ≤ 990) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_multiple_of_2_3_5_l1161_116108


namespace NUMINAMATH_CALUDE_total_gumballs_l1161_116146

/-- Represents the number of gumballs of each color in a gumball machine. -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Defines the properties of the gumball machine as described in the problem. -/
def validGumballMachine (m : GumballMachine) : Prop :=
  m.blue = m.red / 2 ∧ m.green = 4 * m.blue ∧ m.red = 16

/-- Theorem stating that a valid gumball machine contains 56 gumballs in total. -/
theorem total_gumballs (m : GumballMachine) (h : validGumballMachine m) :
  m.red + m.blue + m.green = 56 := by
  sorry

#check total_gumballs

end NUMINAMATH_CALUDE_total_gumballs_l1161_116146


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1161_116160

def f (x : ℝ) := x * |x|

theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1161_116160


namespace NUMINAMATH_CALUDE_fraction_calculation_l1161_116186

theorem fraction_calculation : 
  (1/5 - 1/3) / ((3/7) / (2/9)) = -28/405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1161_116186


namespace NUMINAMATH_CALUDE_rain_difference_l1161_116159

/-- The amount of rain Greg experienced while camping, in millimeters. -/
def camping_rain : List ℝ := [3, 6, 5]

/-- The amount of rain at Greg's house during the same week, in millimeters. -/
def house_rain : ℝ := 26

/-- The difference in rainfall between Greg's house and his camping trip. -/
theorem rain_difference : house_rain - (camping_rain.sum) = 12 := by
  sorry

end NUMINAMATH_CALUDE_rain_difference_l1161_116159


namespace NUMINAMATH_CALUDE_podcast_ratio_l1161_116185

def total_drive_time : ℕ := 360 -- in minutes
def first_podcast : ℕ := 45 -- in minutes
def third_podcast : ℕ := 105 -- in minutes
def fourth_podcast : ℕ := 60 -- in minutes
def next_podcast : ℕ := 60 -- in minutes

theorem podcast_ratio : 
  let second_podcast := total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast)
  (second_podcast : ℚ) / first_podcast = 2 := by
sorry

end NUMINAMATH_CALUDE_podcast_ratio_l1161_116185


namespace NUMINAMATH_CALUDE_equation_solution_l1161_116101

theorem equation_solution (a b : ℝ) (h : a ≠ -1) :
  let x := (a^2 - b^2 + 2*a - 2*b) / (2*(a+1))
  x^2 + (b+1)^2 = (a+1 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1161_116101


namespace NUMINAMATH_CALUDE_total_dinners_sold_l1161_116100

def monday_sales : ℕ := 40

def tuesday_sales : ℕ := monday_sales + 40

def wednesday_sales : ℕ := tuesday_sales / 2

def thursday_sales : ℕ := wednesday_sales + 3

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales

theorem total_dinners_sold : total_sales = 203 := by
  sorry

end NUMINAMATH_CALUDE_total_dinners_sold_l1161_116100


namespace NUMINAMATH_CALUDE_F_opposite_A_l1161_116172

/-- Represents a face of a cube --/
inductive CubeFace
| A | B | C | D | E | F

/-- Represents the position of a face relative to face A in the net --/
inductive Position
| Left | Above | Right | Below | NotAttached

/-- Describes the layout of faces in the cube net --/
def net_layout : CubeFace → Position
| CubeFace.B => Position.Left
| CubeFace.C => Position.Above
| CubeFace.D => Position.Right
| CubeFace.E => Position.Below
| CubeFace.F => Position.NotAttached
| CubeFace.A => Position.NotAttached  -- A's position relative to itself is not relevant

/-- Determines if two faces are opposite in the folded cube --/
def are_opposite (f1 f2 : CubeFace) : Prop := sorry

/-- Theorem stating that face F is opposite to face A when the net is folded --/
theorem F_opposite_A : are_opposite CubeFace.F CubeFace.A := by
  sorry

end NUMINAMATH_CALUDE_F_opposite_A_l1161_116172


namespace NUMINAMATH_CALUDE_cookie_sugar_measurement_l1161_116175

def sugar_needed : ℚ := 15/4  -- 3¾ cups of sugar
def cup_capacity : ℚ := 1/3   -- ⅓ cup measuring cup

theorem cookie_sugar_measurement : ∃ n : ℕ, n * cup_capacity ≥ sugar_needed ∧ 
  ∀ m : ℕ, m * cup_capacity ≥ sugar_needed → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_sugar_measurement_l1161_116175


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1161_116198

theorem decimal_to_fraction : 
  (3.56 : ℚ) = 89 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1161_116198


namespace NUMINAMATH_CALUDE_inequality_range_l1161_116167

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x + 1 > 2*x + m) → m < -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1161_116167


namespace NUMINAMATH_CALUDE_semicircle_problem_l1161_116142

theorem semicircle_problem (M : ℕ) (r : ℝ) (h_positive : r > 0) : 
  (M * π * r^2 / 2) / (π * r^2 * (M^2 - M) / 2) = 1/4 → M = 5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_problem_l1161_116142


namespace NUMINAMATH_CALUDE_ralph_squares_count_l1161_116182

/-- The number of matchsticks in a box -/
def total_matchsticks : ℕ := 50

/-- The number of matchsticks Elvis uses for one square -/
def elvis_square_size : ℕ := 4

/-- The number of matchsticks Ralph uses for one square -/
def ralph_square_size : ℕ := 8

/-- The number of squares Elvis makes -/
def elvis_squares : ℕ := 5

/-- The number of matchsticks left in the box -/
def remaining_matchsticks : ℕ := 6

/-- The number of squares Ralph makes -/
def ralph_squares : ℕ := 3

theorem ralph_squares_count : 
  elvis_square_size * elvis_squares + ralph_square_size * ralph_squares + remaining_matchsticks = total_matchsticks :=
by sorry

end NUMINAMATH_CALUDE_ralph_squares_count_l1161_116182


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1161_116119

/-- Two vectors a and b in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular (x, 2) (1, -1) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1161_116119
