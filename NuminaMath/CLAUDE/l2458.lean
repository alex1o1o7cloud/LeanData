import Mathlib

namespace NUMINAMATH_CALUDE_store_shelves_proof_l2458_245837

/-- Calculates the number of shelves needed to store coloring books -/
def shelves_needed (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (initial_stock - books_sold) / books_per_shelf

/-- Proves that the number of shelves needed is 7 given the problem conditions -/
theorem store_shelves_proof :
  shelves_needed 86 37 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_store_shelves_proof_l2458_245837


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2458_245887

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x^2 + 1/x^2 = 5) : x^4 + 1/x^4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2458_245887


namespace NUMINAMATH_CALUDE_mary_promised_cards_l2458_245885

/-- The number of baseball cards Mary promised to give Fred -/
def promised_cards (initial : ℝ) (bought : ℝ) (left : ℝ) : ℝ :=
  initial + bought - left

theorem mary_promised_cards :
  promised_cards 18.0 40.0 32.0 = 26.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_promised_cards_l2458_245885


namespace NUMINAMATH_CALUDE_negative_calculation_l2458_245891

theorem negative_calculation : 
  ((-4) + (-5) < 0) ∧ 
  ((-4) - (-5) ≥ 0) ∧ 
  ((-4) * (-5) ≥ 0) ∧ 
  ((-4) / (-5) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_calculation_l2458_245891


namespace NUMINAMATH_CALUDE_function_equality_l2458_245827

open Real

theorem function_equality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 1) * x + 3) : 
  f 0 = f 4 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2458_245827


namespace NUMINAMATH_CALUDE_cylinder_equation_l2458_245849

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSurface c

theorem cylinder_equation (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSurface c) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_equation_l2458_245849


namespace NUMINAMATH_CALUDE_managers_salary_l2458_245855

/-- Proves that the manager's salary is 11500 given the conditions of the problem -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 24 →
  avg_salary = 1500 →
  salary_increase = 400 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase : ℕ) = 11500 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l2458_245855


namespace NUMINAMATH_CALUDE_largest_divisor_of_1615_l2458_245817

theorem largest_divisor_of_1615 (n : ℕ) : n ≤ 5 ↔ n * 1615 ≤ 8640 ∧ n * 1615 ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_1615_l2458_245817


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l2458_245853

/-- Represents the fruit stand problem --/
structure FruitStand where
  apple_price : ℝ
  banana_price : ℝ
  orange_price : ℝ
  apple_discount : ℝ
  min_fruit_qty : ℕ
  emmy_budget : ℝ
  gerry_budget : ℝ

/-- Calculates the maximum number of apples that can be bought --/
def max_apples (fs : FruitStand) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem fruit_stand_problem :
  let fs : FruitStand := {
    apple_price := 2,
    banana_price := 1,
    orange_price := 3,
    apple_discount := 0.2,
    min_fruit_qty := 5,
    emmy_budget := 200,
    gerry_budget := 100
  }
  max_apples fs = 160 :=
sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l2458_245853


namespace NUMINAMATH_CALUDE_smallest_n_below_threshold_l2458_245807

/-- The number of boxes in the warehouse -/
def num_boxes : ℕ := 2023

/-- The probability of drawing a green marble on the nth draw -/
def Q (n : ℕ) : ℚ := 1 / (n * (2 * n + 1))

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

/-- 32 is the smallest positive integer n such that Q(n) < 1/2023 -/
theorem smallest_n_below_threshold : 
  (∀ k < 32, Q k ≥ threshold) ∧ Q 32 < threshold :=
sorry

end NUMINAMATH_CALUDE_smallest_n_below_threshold_l2458_245807


namespace NUMINAMATH_CALUDE_total_sequences_value_l2458_245842

/-- The number of students in the first class -/
def students_class1 : ℕ := 12

/-- The number of students in the second class -/
def students_class2 : ℕ := 13

/-- The number of meetings per week for each class -/
def meetings_per_week : ℕ := 3

/-- The total number of different sequences of students solving problems for both classes in a week -/
def total_sequences : ℕ := (students_class1 * students_class2) ^ meetings_per_week

theorem total_sequences_value : total_sequences = 3796416 := by sorry

end NUMINAMATH_CALUDE_total_sequences_value_l2458_245842


namespace NUMINAMATH_CALUDE_min_xyz_value_l2458_245824

theorem min_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + z = (x + z) * (y + z)) :
  x * y * z ≥ (1 : ℝ) / 27 := by sorry

end NUMINAMATH_CALUDE_min_xyz_value_l2458_245824


namespace NUMINAMATH_CALUDE_percent_relation_l2458_245818

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : c = 0.50 * b) : b = 0.50 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2458_245818


namespace NUMINAMATH_CALUDE_impossible_equal_sum_distribution_l2458_245884

theorem impossible_equal_sum_distribution : ∀ n : ℕ, 2 ≤ n → n ≤ 14 →
  ¬ ∃ (partition : List (List ℕ)), 
    (∀ group ∈ partition, ∀ x ∈ group, 1 ≤ x ∧ x ≤ 14) ∧
    (partition.length = n) ∧
    (∀ group ∈ partition, group.sum = 105 / n) ∧
    (partition.join.toFinset = Finset.range 14) :=
by sorry

end NUMINAMATH_CALUDE_impossible_equal_sum_distribution_l2458_245884


namespace NUMINAMATH_CALUDE_right_triangles_problem_l2458_245896

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define a right triangle
def RightTriangle (A B C : ℝ × ℝ) := Triangle A B C ∧ True

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem right_triangles_problem 
  (A B C D : ℝ × ℝ) 
  (a : ℝ) 
  (h1 : RightTriangle A B C)
  (h2 : RightTriangle A B D)
  (h3 : Length B C = 3)
  (h4 : Length A C = a)
  (h5 : Length A D = 1) :
  Length B D = Real.sqrt (a^2 + 8) := by
  sorry


end NUMINAMATH_CALUDE_right_triangles_problem_l2458_245896


namespace NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_le_two_l2458_245890

theorem quadratic_inequality_iff_abs_a_le_two (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ |a| ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_le_two_l2458_245890


namespace NUMINAMATH_CALUDE_poster_cost_l2458_245816

theorem poster_cost (initial_amount : ℕ) (notebook_cost : ℕ) (bookmark_cost : ℕ)
  (poster_count : ℕ) (leftover : ℕ) :
  initial_amount = 40 →
  notebook_cost = 12 →
  bookmark_cost = 4 →
  poster_count = 2 →
  leftover = 14 →
  (initial_amount - notebook_cost - bookmark_cost - leftover) / poster_count = 13 := by
  sorry

end NUMINAMATH_CALUDE_poster_cost_l2458_245816


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l2458_245857

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_one
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = 2^x + 2*x + m)
  (m : ℝ) :
  f (-1) = -3 :=
sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l2458_245857


namespace NUMINAMATH_CALUDE_cos_240_degrees_l2458_245888

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l2458_245888


namespace NUMINAMATH_CALUDE_cube_root_sum_reciprocal_cube_l2458_245833

theorem cube_root_sum_reciprocal_cube (x : ℝ) : 
  x = Real.rpow 4 (1/3) + Real.rpow 2 (1/3) + 1 → (1 + 1/x)^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_reciprocal_cube_l2458_245833


namespace NUMINAMATH_CALUDE_marcy_cat_time_l2458_245871

def total_time (petting combing brushing playing feeding cleaning : ℚ) : ℚ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem marcy_cat_time : 
  let petting : ℚ := 12
  let combing : ℚ := (1/3) * petting
  let brushing : ℚ := (1/4) * combing
  let playing : ℚ := (1/2) * petting
  let feeding : ℚ := 5
  let cleaning : ℚ := (2/5) * feeding
  total_time petting combing brushing playing feeding cleaning = 30 := by
sorry

end NUMINAMATH_CALUDE_marcy_cat_time_l2458_245871


namespace NUMINAMATH_CALUDE_watson_class_composition_l2458_245883

/-- Represents the number of students in each grade level in Ms. Watson's class -/
structure ClassComposition where
  kindergartners : Nat
  first_graders : Nat
  second_graders : Nat

/-- The total number of students in Ms. Watson's class -/
def total_students (c : ClassComposition) : Nat :=
  c.kindergartners + c.first_graders + c.second_graders

/-- Theorem stating that given the conditions of Ms. Watson's class, 
    there are 4 second graders -/
theorem watson_class_composition :
  ∃ (c : ClassComposition),
    c.kindergartners = 14 ∧
    c.first_graders = 24 ∧
    total_students c = 42 ∧
    c.second_graders = 4 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_composition_l2458_245883


namespace NUMINAMATH_CALUDE_slope_of_line_l2458_245806

theorem slope_of_line (x y : ℝ) : y = x - 1 → (y - (x - 1)) / (x - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2458_245806


namespace NUMINAMATH_CALUDE_smallest_even_five_digit_number_tens_place_l2458_245856

def Digits : Finset ℕ := {1, 2, 3, 5, 8}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  (∀ d : ℕ, d ∈ Digits → (n.digits 10).count d = 1) ∧
  (∀ d : ℕ, d ∉ Digits → (n.digits 10).count d = 0)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem smallest_even_five_digit_number_tens_place :
  ∃ n : ℕ, is_valid_number n ∧ is_even n ∧
    (∀ m : ℕ, is_valid_number m ∧ is_even m → n ≤ m) ∧
    tens_digit n = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_five_digit_number_tens_place_l2458_245856


namespace NUMINAMATH_CALUDE_house_market_value_l2458_245813

/-- Proves that the market value of a house is $500,000 given the specified conditions --/
theorem house_market_value : 
  ∀ (market_value selling_price revenue_per_person : ℝ),
  selling_price = market_value * 1.2 →
  selling_price = 4 * revenue_per_person →
  revenue_per_person * 0.9 = 135000 →
  market_value = 500000 := by
  sorry

end NUMINAMATH_CALUDE_house_market_value_l2458_245813


namespace NUMINAMATH_CALUDE_payment_is_two_l2458_245845

/-- The amount Edmund needs to save -/
def saving_goal : ℕ := 75

/-- The number of chores Edmund normally does per week -/
def normal_chores_per_week : ℕ := 12

/-- The number of chores Edmund does per day during the saving period -/
def chores_per_day : ℕ := 4

/-- The number of days Edmund works during the saving period -/
def working_days : ℕ := 14

/-- The total amount Edmund earns for extra chores -/
def total_earned : ℕ := 64

/-- Calculates the number of extra chores Edmund does -/
def extra_chores : ℕ := chores_per_day * working_days - normal_chores_per_week * 2

/-- The payment per extra chore -/
def payment_per_extra_chore : ℚ := total_earned / extra_chores

theorem payment_is_two :
  payment_per_extra_chore = 2 := by sorry

end NUMINAMATH_CALUDE_payment_is_two_l2458_245845


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_l2458_245872

theorem parametric_to_ordinary :
  ∀ θ : ℝ,
  let x : ℝ := 2 + Real.sin θ ^ 2
  let y : ℝ := -1 + Real.cos (2 * θ)
  2 * x + y - 4 = 0 ∧ x ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_l2458_245872


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2458_245811

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement that a_3 and a_10 are roots of x^2 - 3x - 5 = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  a 3 ^ 2 - 3 * a 3 - 5 = 0 ∧ a 10 ^ 2 - 3 * a 10 - 5 = 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : roots_condition a) : 
  a 5 + a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2458_245811


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2458_245821

theorem quadratic_equation_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ + (m - 1) = 0 ∧ x₂^2 - m*x₂ + (m - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2458_245821


namespace NUMINAMATH_CALUDE_fib_sum_product_l2458_245822

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem: F_{m+n} = F_{m-1} * F_n + F_m * F_{n+1} for all non-negative integers m and n -/
theorem fib_sum_product (m n : ℕ) : fib (m + n) = fib (m - 1) * fib n + fib m * fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_product_l2458_245822


namespace NUMINAMATH_CALUDE_arithmetic_progression_nested_l2458_245852

/-- An arithmetic progression of distinct positive integers -/
def ArithmeticProgression (s : ℕ → ℕ) : Prop :=
  ∃ a b : ℤ, a ≠ 0 ∧ ∀ n : ℕ, s n = a * n + b

/-- The sequence is strictly increasing -/
def StrictlyIncreasing (s : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → s m < s n

/-- All elements in the sequence are positive -/
def AllPositive (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < s n

theorem arithmetic_progression_nested (s : ℕ → ℕ) :
  ArithmeticProgression s →
  StrictlyIncreasing s →
  AllPositive s →
  ArithmeticProgression (fun n ↦ s (s n)) ∧
  StrictlyIncreasing (fun n ↦ s (s n)) ∧
  AllPositive (fun n ↦ s (s n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_nested_l2458_245852


namespace NUMINAMATH_CALUDE_sphere_in_cylinder_ratio_l2458_245893

theorem sphere_in_cylinder_ratio : 
  ∀ (r h : ℝ),
  r > 0 →
  h > 0 →
  (4 / 3 * π * r^3) * 2 = π * r^2 * h →
  h / (2 * r) = 4 / 3 :=
λ r h hr hh vol_eq ↦ by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cylinder_ratio_l2458_245893


namespace NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2458_245801

-- Define a die as having 6 faces
def die : Finset ℕ := Finset.range 6

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := die.card ^ 3

-- Define the favorable outcomes (sum of 10)
def favorable_outcomes : ℕ := 27

-- Theorem statement
theorem probability_sum_10_three_dice :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2458_245801


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2458_245838

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2458_245838


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2458_245860

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (6, 8, 11) cannot form a right-angled triangle --/
theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 8 15 17 ∧
  is_right_triangle 7 24 25 ∧
  ¬(is_right_triangle 6 8 11) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2458_245860


namespace NUMINAMATH_CALUDE_sin_theta_value_l2458_245802

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : 
  Real.sin θ = 3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l2458_245802


namespace NUMINAMATH_CALUDE_cans_storage_l2458_245876

theorem cans_storage (cans_per_row : ℕ) (shelves_per_closet : ℕ) (cans_per_closet : ℕ) :
  cans_per_row = 12 →
  shelves_per_closet = 10 →
  cans_per_closet = 480 →
  (cans_per_closet / cans_per_row) / shelves_per_closet = 4 :=
by sorry

end NUMINAMATH_CALUDE_cans_storage_l2458_245876


namespace NUMINAMATH_CALUDE_james_budget_theorem_l2458_245864

/-- James's budget and expenses --/
def budget : ℝ := 1000
def food_percentage : ℝ := 0.22
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.18
def transportation_percentage : ℝ := 0.12
def clothes_percentage : ℝ := 0.08
def miscellaneous_percentage : ℝ := 0.05

/-- Theorem: James's savings percentage and combined expenses --/
theorem james_budget_theorem :
  let food := budget * food_percentage
  let accommodation := budget * accommodation_percentage
  let entertainment := budget * entertainment_percentage
  let transportation := budget * transportation_percentage
  let clothes := budget * clothes_percentage
  let miscellaneous := budget * miscellaneous_percentage
  let total_spent := food + accommodation + entertainment + transportation + clothes + miscellaneous
  let savings := budget - total_spent
  let savings_percentage := (savings / budget) * 100
  let combined_expenses := entertainment + transportation + miscellaneous
  savings_percentage = 20 ∧ combined_expenses = 350 := by
  sorry

end NUMINAMATH_CALUDE_james_budget_theorem_l2458_245864


namespace NUMINAMATH_CALUDE_divisible_by_64_l2458_245830

theorem divisible_by_64 (n : ℕ+) : ∃ k : ℤ, 3^(2*n.val + 2) - 8*n.val - 9 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l2458_245830


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_origin_l2458_245886

/-- The shortest distance between any point on the circle (x-2)^2+(y+m-4)^2=1 and the origin (0,0) is 1, where m is a real number. -/
theorem shortest_distance_circle_to_origin :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), (x - 2)^2 + (y + m - 4)^2 = 1) →
  (∃ (d : ℝ), d = 1 ∧ 
    ∀ (x y : ℝ), (x - 2)^2 + (y + m - 4)^2 = 1 → 
      Real.sqrt (x^2 + y^2) ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_origin_l2458_245886


namespace NUMINAMATH_CALUDE_parallelogram_on_circle_l2458_245882

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is inside a circle -/
def isInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Check if a point is on a circle -/
def isOn (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if four points form a parallelogram -/
def isParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2) ∧
  (c.1 - b.1 = a.1 - d.1) ∧ (c.2 - b.2 = a.2 - d.2)

theorem parallelogram_on_circle (ω : Circle) (A B : ℝ × ℝ) 
  (h_A : isInside ω A) (h_B : isOn ω B) :
  ∃ (C D : ℝ × ℝ), isOn ω C ∧ isOn ω D ∧ isParallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_parallelogram_on_circle_l2458_245882


namespace NUMINAMATH_CALUDE_ellipse_tangent_and_normal_l2458_245899

noncomputable def ellipse (t : ℝ) : ℝ × ℝ := (4 * Real.cos t, 3 * Real.sin t)

theorem ellipse_tangent_and_normal (t : ℝ) :
  let (x₀, y₀) := ellipse (π/3)
  let tangent_slope := -(3 * Real.cos (π/3)) / (4 * Real.sin (π/3))
  let normal_slope := -1 / tangent_slope
  (∀ x y, y - y₀ = tangent_slope * (x - x₀) ↔ y = -Real.sqrt 3 / 4 * x + 2 * Real.sqrt 3) ∧
  (∀ x y, y - y₀ = normal_slope * (x - x₀) ↔ y = 4 / Real.sqrt 3 * x - 7 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_and_normal_l2458_245899


namespace NUMINAMATH_CALUDE_problem_solution_l2458_245850

theorem problem_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = 100) (h4 : (3/10) * a = (1/5) * b) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2458_245850


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l2458_245848

theorem photo_arrangement_count :
  let total_people : ℕ := 7
  let adjacent_pair : ℕ := 1  -- A and B treated as one unit
  let separated_pair : ℕ := 2  -- C and D
  let other_people : ℕ := total_people - adjacent_pair - separated_pair
  
  let total_elements : ℕ := adjacent_pair + other_people + 1
  let adjacent_pair_arrangements : ℕ := 2  -- A and B can switch
  let spaces_for_separated : ℕ := total_elements + 1

  (total_elements.factorial * adjacent_pair_arrangements * 
   (spaces_for_separated * (spaces_for_separated - 1))) = 960 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l2458_245848


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l2458_245826

-- Define the opposite of a rational number
def opposite (x : ℚ) : ℚ := -x

-- Theorem statement
theorem opposite_of_negative_two_thirds : 
  opposite (-2/3) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l2458_245826


namespace NUMINAMATH_CALUDE_last_digit_divisibility_l2458_245881

theorem last_digit_divisibility (n : ℕ) (h : n > 3) :
  let a := (2^n) % 10
  let b := 2^n - a
  6 ∣ (a * b) := by sorry

end NUMINAMATH_CALUDE_last_digit_divisibility_l2458_245881


namespace NUMINAMATH_CALUDE_function_properties_l2458_245847

def f (x a : ℝ) : ℝ := (4*a + 2)*x^2 + (9 - 6*a)*x - 4*a + 4

theorem function_properties :
  (∀ a : ℝ, ∃ x : ℝ, f x a = 0) ∧
  (∃ a : ℤ, ∃ x : ℤ, f (x : ℝ) (a : ℝ) = 0) ∧
  ({a : ℤ | ∃ x : ℤ, f (x : ℝ) (a : ℝ) = 0} = {-2, -1, 0, 1}) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2458_245847


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l2458_245895

/-- Given vectors a and b, if a + 3b is parallel to b, then the first component of a is 6. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (m : ℝ) :
  a = (m, 2) →
  b = (3, 1) →
  ∃ k : ℝ, a + 3 • b = k • b →
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l2458_245895


namespace NUMINAMATH_CALUDE_solve_for_a_l2458_245869

theorem solve_for_a : ∃ a : ℝ, (1 : ℝ) - a * 2 = 3 ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2458_245869


namespace NUMINAMATH_CALUDE_total_material_needed_l2458_245812

-- Define the dimensions of the tablecloth
def tablecloth_length : ℕ := 102
def tablecloth_width : ℕ := 54

-- Define the dimensions of a napkin
def napkin_length : ℕ := 6
def napkin_width : ℕ := 7

-- Define the number of napkins
def num_napkins : ℕ := 8

-- Theorem to prove
theorem total_material_needed :
  tablecloth_length * tablecloth_width + num_napkins * napkin_length * napkin_width = 5844 :=
by sorry

end NUMINAMATH_CALUDE_total_material_needed_l2458_245812


namespace NUMINAMATH_CALUDE_descent_time_l2458_245805

/-- Prove that the time to descend a hill is 2 hours given the specified conditions -/
theorem descent_time (climb_time : ℝ) (climb_speed : ℝ) (total_avg_speed : ℝ) :
  climb_time = 4 →
  climb_speed = 2.625 →
  total_avg_speed = 3.5 →
  ∃ (descent_time : ℝ),
    descent_time = 2 ∧
    (2 * climb_time * climb_speed) = (total_avg_speed * (climb_time + descent_time)) :=
by sorry

end NUMINAMATH_CALUDE_descent_time_l2458_245805


namespace NUMINAMATH_CALUDE_area_transformation_l2458_245814

-- Define a function representing the area between a curve and the x-axis
noncomputable def area_between_curve_and_x_axis (f : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem area_transformation (g : ℝ → ℝ) 
  (h : area_between_curve_and_x_axis g = 8) : 
  area_between_curve_and_x_axis (λ x => 4 * g (x + 3)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_area_transformation_l2458_245814


namespace NUMINAMATH_CALUDE_distance_between_trees_l2458_245839

/-- Given a yard with trees planted at equal distances, calculate the distance between consecutive trees. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 255 ∧ num_trees = 18 → 
  (yard_length / (num_trees - 1 : ℝ)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2458_245839


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2458_245800

/-- Tom's age problem -/
theorem toms_age_ratio (T N : ℚ) : T > 0 → N > 0 →
  (∃ (a b c d : ℚ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ T = a + b + c + d) →
  (T - N = 3 * (T - 4 * N)) →
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2458_245800


namespace NUMINAMATH_CALUDE_percentage_problem_l2458_245877

theorem percentage_problem (P : ℝ) : 
  (0.2 * 580 = (P / 100) * 120 + 80) → P = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2458_245877


namespace NUMINAMATH_CALUDE_difference_61st_terms_l2458_245862

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def sequenceC (n : ℕ) : ℝ := arithmeticSequence 20 15 n

def sequenceD (n : ℕ) : ℝ := arithmeticSequence 20 (-15) n

theorem difference_61st_terms :
  |sequenceC 61 - sequenceD 61| = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_61st_terms_l2458_245862


namespace NUMINAMATH_CALUDE_quadratic_roots_l2458_245820

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  f : ℝ → ℝ
  passesThrough : f (-3) = 0 ∧ f (-2) = -3 ∧ f 0 = -3

/-- The roots of the quadratic function -/
def roots (qf : QuadraticFunction) : Set ℝ :=
  {x : ℝ | qf.f x = 0}

/-- Theorem stating the roots of the quadratic function -/
theorem quadratic_roots (qf : QuadraticFunction) : roots qf = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2458_245820


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2458_245878

/-- Represents a repeating decimal with a single digit repetend -/
def repeating_decimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repetend -/
def repeating_decimal_two_digits (n : ℕ) : ℚ := n / 99

theorem repeating_decimal_sum :
  repeating_decimal 6 + repeating_decimal_two_digits 12 - repeating_decimal 4 = 34 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2458_245878


namespace NUMINAMATH_CALUDE_monotonic_decreasing_intervals_of_neg_tan_l2458_245875

open Real

noncomputable def f (x : ℝ) := -tan x

theorem monotonic_decreasing_intervals_of_neg_tan :
  ∀ (k : ℤ) (x y : ℝ),
    x ∈ Set.Ioo (k * π - π / 2) (k * π + π / 2) →
    y ∈ Set.Ioo (k * π - π / 2) (k * π + π / 2) →
    x < y →
    f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_intervals_of_neg_tan_l2458_245875


namespace NUMINAMATH_CALUDE_equation_solution_l2458_245867

theorem equation_solution :
  ∃! x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2458_245867


namespace NUMINAMATH_CALUDE_line_and_volume_proof_l2458_245889

-- Define the line l
def line_l (x y : ℝ) := x + y - 4 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) := x + y - 1 = 0

-- Theorem statement
theorem line_and_volume_proof :
  -- Condition 1: Line l passes through (3,1)
  line_l 3 1 ∧
  -- Condition 2: Line l is parallel to x+y-1=0
  ∀ (x y : ℝ), line_l x y ↔ ∃ (k : ℝ), parallel_line (x + k) (y + k) →
  -- Conclusion 1: Equation of line l is x+y-4=0
  (∀ (x y : ℝ), line_l x y ↔ x + y - 4 = 0) ∧
  -- Conclusion 2: Volume of the geometric solid is (64/3)π
  (let volume := (64 / 3) * Real.pi
   volume = (1 / 3) * Real.pi * 4^2 * 4) :=
by sorry

end NUMINAMATH_CALUDE_line_and_volume_proof_l2458_245889


namespace NUMINAMATH_CALUDE_number_count_l2458_245851

theorem number_count (average_all : ℝ) (average_group1 : ℝ) (average_group2 : ℝ) (average_group3 : ℝ) 
  (h1 : average_all = 3.9)
  (h2 : average_group1 = 3.4)
  (h3 : average_group2 = 3.85)
  (h4 : average_group3 = 4.45) :
  ∃ (n : ℕ), n = 6 ∧ (n : ℝ) * average_all = 2 * (average_group1 + average_group2 + average_group3) := by
  sorry

end NUMINAMATH_CALUDE_number_count_l2458_245851


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l2458_245874

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l2458_245874


namespace NUMINAMATH_CALUDE_net_pay_rate_l2458_245846

-- Define the given conditions
def travel_time : ℝ := 3
def speed : ℝ := 60
def fuel_efficiency : ℝ := 30
def pay_rate : ℝ := 0.60
def gas_price : ℝ := 2.50

-- Define the theorem
theorem net_pay_rate : 
  let distance := travel_time * speed
  let gas_used := distance / fuel_efficiency
  let earnings := distance * pay_rate
  let gas_cost := gas_used * gas_price
  let net_earnings := earnings - gas_cost
  net_earnings / travel_time = 31 := by sorry

end NUMINAMATH_CALUDE_net_pay_rate_l2458_245846


namespace NUMINAMATH_CALUDE_D_72_l2458_245841

/-- D(n) is the number of ways to express n as a product of integers greater than 1, considering order as distinct -/
def D (n : ℕ) : ℕ := sorry

/-- The prime factorization of 72 -/
def prime_factorization_72 : List (ℕ × ℕ) := [(2, 3), (3, 2)]

theorem D_72 : D 72 = 22 := by sorry

end NUMINAMATH_CALUDE_D_72_l2458_245841


namespace NUMINAMATH_CALUDE_inequality_proof_l2458_245894

theorem inequality_proof (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x^2 + y^2 + z^2 = 3) : 
  (x^2009 - 2008*(x-1))/(y+z) + (y^2009 - 2008*(y-1))/(x+z) + (z^2009 - 2008*(z-1))/(x+y) ≥ (1/2)*(x+y+z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2458_245894


namespace NUMINAMATH_CALUDE_chip_credit_card_balance_l2458_245823

/-- Calculates the final balance on a credit card after two months with interest --/
def final_balance (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_month := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_month + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating the final balance on Chip's credit card --/
theorem chip_credit_card_balance :
  final_balance 50 0.2 20 = 96 :=
by sorry

end NUMINAMATH_CALUDE_chip_credit_card_balance_l2458_245823


namespace NUMINAMATH_CALUDE_max_value_3a_plus_b_l2458_245844

theorem max_value_3a_plus_b (a b : ℝ) :
  (∀ x ∈ Set.Icc 1 2, |a * x^2 + b * x + a| ≤ x) →
  (∃ a₀ b₀ : ℝ, (∀ x ∈ Set.Icc 1 2, |a₀ * x^2 + b₀ * x + a₀| ≤ x) ∧ 3 * a₀ + b₀ = 3) ∧
  (∀ a₁ b₁ : ℝ, (∀ x ∈ Set.Icc 1 2, |a₁ * x^2 + b₁ * x + a₁| ≤ x) → 3 * a₁ + b₁ ≤ 3) :=
by sorry

#check max_value_3a_plus_b

end NUMINAMATH_CALUDE_max_value_3a_plus_b_l2458_245844


namespace NUMINAMATH_CALUDE_mets_fans_count_l2458_245831

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The conditions of the problem -/
def fan_conditions (fc : FanCounts) : Prop :=
  -- Ratio of Yankees to Mets fans is 3:2
  3 * fc.mets = 2 * fc.yankees ∧
  -- Ratio of Mets to Red Sox fans is 4:5
  4 * fc.red_sox = 5 * fc.mets ∧
  -- Total number of fans is 330
  fc.yankees + fc.mets + fc.red_sox = 330

/-- The theorem to prove -/
theorem mets_fans_count (fc : FanCounts) : 
  fan_conditions fc → fc.mets = 88 := by
  sorry


end NUMINAMATH_CALUDE_mets_fans_count_l2458_245831


namespace NUMINAMATH_CALUDE_adams_initial_money_l2458_245898

theorem adams_initial_money (initial_amount : ℚ) : 
  (initial_amount - 21) / 21 = 10 / 3 → initial_amount = 91 :=
by sorry

end NUMINAMATH_CALUDE_adams_initial_money_l2458_245898


namespace NUMINAMATH_CALUDE_gcd_111_1850_l2458_245829

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_111_1850_l2458_245829


namespace NUMINAMATH_CALUDE_initial_amount_sufficient_l2458_245815

/-- Kanul's initial amount of money -/
def initial_amount : ℝ := 11058.82

/-- Raw materials cost -/
def raw_materials_cost : ℝ := 5000

/-- Machinery cost -/
def machinery_cost : ℝ := 200

/-- Employee wages -/
def employee_wages : ℝ := 1200

/-- Maintenance cost percentage -/
def maintenance_percentage : ℝ := 0.15

/-- Desired remaining balance -/
def desired_balance : ℝ := 3000

/-- Theorem: Given the expenses and conditions, the initial amount is sufficient -/
theorem initial_amount_sufficient :
  initial_amount - (raw_materials_cost + machinery_cost + employee_wages + maintenance_percentage * initial_amount) ≥ desired_balance := by
  sorry

#check initial_amount_sufficient

end NUMINAMATH_CALUDE_initial_amount_sufficient_l2458_245815


namespace NUMINAMATH_CALUDE_sqrt_3x_minus_1_defined_l2458_245873

theorem sqrt_3x_minus_1_defined (x : ℝ) : Real.sqrt (3 * x - 1) ≥ 0 ↔ x ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3x_minus_1_defined_l2458_245873


namespace NUMINAMATH_CALUDE_circle_condition_relationship_l2458_245834

theorem circle_condition_relationship :
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → (x - 1)^2 + y^2 ≤ 4) ∧
  (∃ x y : ℝ, (x - 1)^2 + y^2 ≤ 4 ∧ x^2 + y^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_relationship_l2458_245834


namespace NUMINAMATH_CALUDE_min_disks_for_problem_l2458_245836

/-- Represents a file with its size in MB -/
structure File where
  size : Float

/-- Represents a disk with its capacity in MB -/
structure Disk where
  capacity : Float

/-- Function to calculate the minimum number of disks needed -/
def min_disks_needed (files : List File) (disk_capacity : Float) : Nat :=
  sorry

/-- Theorem stating the minimum number of disks needed for the given problem -/
theorem min_disks_for_problem : 
  let files : List File := 
    (List.replicate 5 ⟨1.0⟩) ++ 
    (List.replicate 15 ⟨0.6⟩) ++ 
    (List.replicate 25 ⟨0.3⟩)
  let disk_capacity : Float := 1.44
  min_disks_needed files disk_capacity = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_for_problem_l2458_245836


namespace NUMINAMATH_CALUDE_spending_recording_l2458_245880

/-- Represents the recording of a financial transaction -/
def record (amount : ℤ) : ℤ := amount

/-- Axiom: Depositing is recorded as a positive amount -/
axiom deposit_positive (amount : ℕ) : record amount = amount

/-- The main theorem: If depositing 300 is recorded as +300, then spending 500 should be recorded as -500 -/
theorem spending_recording :
  record 300 = 300 → record (-500) = -500 := by
  sorry

end NUMINAMATH_CALUDE_spending_recording_l2458_245880


namespace NUMINAMATH_CALUDE_collinear_points_y_value_l2458_245810

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_y_value :
  let A : Point := { x := 4, y := 8 }
  let B : Point := { x := 2, y := 4 }
  let C : Point := { x := 3, y := y }
  collinear A B C → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_y_value_l2458_245810


namespace NUMINAMATH_CALUDE_floor_cube_difference_l2458_245858

theorem floor_cube_difference : 
  ⌊(2007^3 : ℝ) / (2005 * 2006) - (2008^3 : ℝ) / (2006 * 2007)⌋ = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_cube_difference_l2458_245858


namespace NUMINAMATH_CALUDE_shopkeeper_face_cards_l2458_245870

/-- The number of complete decks of playing cards the shopkeeper has -/
def num_decks : ℕ := 5

/-- The number of face cards in a standard deck of playing cards -/
def face_cards_per_deck : ℕ := 12

/-- The total number of face cards the shopkeeper has -/
def total_face_cards : ℕ := num_decks * face_cards_per_deck

theorem shopkeeper_face_cards : total_face_cards = 60 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_face_cards_l2458_245870


namespace NUMINAMATH_CALUDE_max_y_over_x_l2458_245828

theorem max_y_over_x (x y : ℝ) 
  (h1 : x - 1 ≥ 0) 
  (h2 : x - y ≥ 0) 
  (h3 : x + y - 4 ≤ 0) : 
  ∃ (max : ℝ), max = 3 ∧ ∀ (x' y' : ℝ), 
    x' - 1 ≥ 0 → x' - y' ≥ 0 → x' + y' - 4 ≤ 0 → y' / x' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_y_over_x_l2458_245828


namespace NUMINAMATH_CALUDE_x_in_interval_l2458_245819

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define x as in the problem
noncomputable def x : ℝ := 1 / log (1/2) (1/3) + 1 / log (1/5) (1/3)

-- State the theorem
theorem x_in_interval : 2 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_x_in_interval_l2458_245819


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2458_245897

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2458_245897


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2458_245879

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 3) * x^2 + x + m^2 - 9 = 0) ∧ 
  ((m - 3) * 0^2 + 0 + m^2 - 9 = 0) →
  m = -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2458_245879


namespace NUMINAMATH_CALUDE_sheila_picnic_probability_l2458_245825

-- Define the probabilities
def rain_probability : ℝ := 0.5
def picnic_if_rain : ℝ := 0.3
def picnic_if_sunny : ℝ := 0.7

-- Theorem statement
theorem sheila_picnic_probability :
  rain_probability * picnic_if_rain + (1 - rain_probability) * picnic_if_sunny = 0.5 := by
  sorry

#eval rain_probability * picnic_if_rain + (1 - rain_probability) * picnic_if_sunny

end NUMINAMATH_CALUDE_sheila_picnic_probability_l2458_245825


namespace NUMINAMATH_CALUDE_product_sign_l2458_245840

theorem product_sign (a b c d e : ℝ) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : 
  a * b^2 * c * d^4 * e < 0 := by
sorry

end NUMINAMATH_CALUDE_product_sign_l2458_245840


namespace NUMINAMATH_CALUDE_sin_sum_angles_l2458_245868

theorem sin_sum_angles (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1/4)
  (h2 : Real.cos α + Real.sin β = -8/5) : 
  Real.sin (α + β) = 249/800 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_angles_l2458_245868


namespace NUMINAMATH_CALUDE_fractional_factorial_max_experiments_l2458_245854

/-- The number of experimental points -/
def n : ℕ := 20

/-- The maximum number of experiments needed -/
def max_experiments : ℕ := 6

/-- Theorem stating that for 20 experimental points, 
    the maximum number of experiments needed is 6 
    when using the fractional factorial design method -/
theorem fractional_factorial_max_experiments :
  n = 2^max_experiments - 1 := by sorry

end NUMINAMATH_CALUDE_fractional_factorial_max_experiments_l2458_245854


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l2458_245859

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def divisible_by (a b : ℕ) : Prop := b ∣ a

theorem unique_three_digit_number_divisible_by_11 :
  ∃! n : ℕ, is_three_digit n ∧ 
            units_digit n = 3 ∧ 
            hundreds_digit n = 6 ∧ 
            divisible_by n 11 ∧
            n = 693 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l2458_245859


namespace NUMINAMATH_CALUDE_triangle_square_distance_l2458_245808

-- Define the triangle ABF
def Triangle (A B F : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), 
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = x^2 ∧
    (B.1 - F.1)^2 + (B.2 - F.2)^2 = y^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = z^2

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ),
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = s^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = s^2 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = s^2 ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = s^2

-- Define the circumcenter of a square
def Circumcenter (E A B C D : ℝ × ℝ) : Prop :=
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (E.1 - B.1)^2 + (E.2 - B.2)^2 ∧
  (E.1 - B.1)^2 + (E.2 - B.2)^2 = (E.1 - C.1)^2 + (E.2 - C.2)^2 ∧
  (E.1 - C.1)^2 + (E.2 - C.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2

theorem triangle_square_distance 
  (A B C D E F : ℝ × ℝ)
  (h1 : Triangle A B F)
  (h2 : Square A B C D)
  (h3 : Circumcenter E A B C D)
  (h4 : (A.1 - F.1)^2 + (A.2 - F.2)^2 = 36)
  (h5 : (B.1 - F.1)^2 + (B.2 - F.2)^2 = 64)
  (h6 : (A.1 - B.1) * (F.1 - B.1) + (A.2 - B.2) * (F.2 - B.2) = 0) :
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 98 :=
by sorry

end NUMINAMATH_CALUDE_triangle_square_distance_l2458_245808


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2458_245866

/-- An arithmetic sequence defined by the given recurrence relation. -/
def ArithmeticSequence (x : ℕ → ℚ) : Prop :=
  ∀ n ≥ 3, x (n - 1) = (x n + x (n - 1) + x (n - 2)) / 3

/-- The main theorem stating the ratio of differences in the sequence. -/
theorem arithmetic_sequence_ratio 
  (x : ℕ → ℚ) 
  (h : ArithmeticSequence x) : 
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2458_245866


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2458_245892

/-- Given a line with equation 5x - 2y = 10, the slope of a perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) →
  (slope_of_perpendicular_line : ℝ) = -2/5 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2458_245892


namespace NUMINAMATH_CALUDE_ratio_xyz_l2458_245865

theorem ratio_xyz (x y z : ℝ) 
  (h1 : 0.6 * x = 0.3 * y)
  (h2 : 0.8 * z = 0.4 * x)
  (h3 : z = 2 * y) :
  ∃ (k : ℝ), k > 0 ∧ x = 4 * k ∧ y = k ∧ z = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_ratio_xyz_l2458_245865


namespace NUMINAMATH_CALUDE_sam_has_46_balloons_l2458_245843

/-- Given the number of red balloons Fred and Dan have, and the total number of red balloons,
    calculate the number of red balloons Sam has. -/
def sams_balloons (fred_balloons dan_balloons total_balloons : ℕ) : ℕ :=
  total_balloons - (fred_balloons + dan_balloons)

/-- Theorem stating that given the specific numbers of balloons in the problem,
    Sam must have 46 red balloons. -/
theorem sam_has_46_balloons :
  sams_balloons 10 16 72 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_46_balloons_l2458_245843


namespace NUMINAMATH_CALUDE_correct_decision_probability_l2458_245863

theorem correct_decision_probability (p : ℝ) (h : p = 0.8) :
  let n := 3  -- number of consultants
  let prob_two_correct := Nat.choose n 2 * p^2 * (1 - p)
  let prob_three_correct := Nat.choose n 3 * p^3
  prob_two_correct + prob_three_correct = 0.896 :=
sorry

end NUMINAMATH_CALUDE_correct_decision_probability_l2458_245863


namespace NUMINAMATH_CALUDE_ball_probabilities_l2458_245809

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : ℕ :=
  counts.red + counts.yellow + counts.white

/-- Calculates the probability of picking a ball of a specific color -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  (favorable : ℚ) / (total : ℚ)

theorem ball_probabilities (initial : BallCounts)
    (h_initial : initial = ⟨10, 6, 4⟩) :
  let total := totalBalls initial
  (probability initial.white total = 1/5) ∧
  (probability (initial.red + initial.yellow) total = 4/5) ∧
  (probability (initial.white - 2) (total - 4) = 1/8) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2458_245809


namespace NUMINAMATH_CALUDE_multiples_of_three_is_closed_l2458_245804

def is_closed (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def multiples_of_three : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem multiples_of_three_is_closed :
  is_closed multiples_of_three :=
by
  sorry

end NUMINAMATH_CALUDE_multiples_of_three_is_closed_l2458_245804


namespace NUMINAMATH_CALUDE_neg_a_cubed_times_a_squared_l2458_245832

theorem neg_a_cubed_times_a_squared (a : ℝ) : (-a)^3 * a^2 = -a^5 := by
  sorry

end NUMINAMATH_CALUDE_neg_a_cubed_times_a_squared_l2458_245832


namespace NUMINAMATH_CALUDE_teacher_age_l2458_245835

/-- Given a class of students and their teacher, calculate the teacher's age based on how it affects the class average. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 22 →
  student_avg_age = 21 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 44 :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l2458_245835


namespace NUMINAMATH_CALUDE_smallest_n_remainder_l2458_245803

theorem smallest_n_remainder (N : ℕ) : 
  (N > 0) →
  (∃ k : ℕ, 2008 * N = k^2) →
  (∃ m : ℕ, 2007 * N = m^3) →
  (∀ M : ℕ, M < N → (¬∃ k : ℕ, 2008 * M = k^2) ∨ (¬∃ m : ℕ, 2007 * M = m^3)) →
  N % 25 = 17 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_remainder_l2458_245803


namespace NUMINAMATH_CALUDE_total_books_l2458_245861

theorem total_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 12)
  (h3 : picture_shelves = 9) :
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 168 :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_l2458_245861
