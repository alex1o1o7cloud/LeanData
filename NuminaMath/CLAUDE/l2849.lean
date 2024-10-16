import Mathlib

namespace NUMINAMATH_CALUDE_min_photos_theorem_l2849_284987

/-- Represents a photo of two children -/
structure Photo where
  child1 : Nat
  child2 : Nat
  deriving Repr

/-- The set of all possible photos -/
def AllPhotos : Set Photo := sorry

/-- The set of photos with two boys -/
def BoyBoyPhotos : Set Photo := sorry

/-- The set of photos with two girls -/
def GirlGirlPhotos : Set Photo := sorry

/-- Predicate to check if two photos are the same -/
def SamePhoto (p1 p2 : Photo) : Prop := sorry

theorem min_photos_theorem (n : Nat) (photos : Fin n → Photo) :
  (∀ i : Fin n, photos i ∈ AllPhotos) →
  (n ≥ 33) →
  (∃ i : Fin n, photos i ∈ BoyBoyPhotos) ∨
  (∃ i : Fin n, photos i ∈ GirlGirlPhotos) ∨
  (∃ i j : Fin n, i ≠ j ∧ SamePhoto (photos i) (photos j)) := by
  sorry

#check min_photos_theorem

end NUMINAMATH_CALUDE_min_photos_theorem_l2849_284987


namespace NUMINAMATH_CALUDE_probability_second_high_given_first_inferior_is_eight_ninths_l2849_284919

/-- Represents the total number of pencils -/
def total_pencils : ℕ := 10

/-- Represents the number of high-quality pencils -/
def high_quality : ℕ := 8

/-- Represents the number of inferior quality pencils -/
def inferior_quality : ℕ := 2

/-- Represents the probability of drawing a high-quality pencil on the second draw,
    given that the first draw was an inferior quality pencil -/
def probability_second_high_given_first_inferior : ℚ :=
  high_quality / (total_pencils - 1)

theorem probability_second_high_given_first_inferior_is_eight_ninths :
  probability_second_high_given_first_inferior = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_high_given_first_inferior_is_eight_ninths_l2849_284919


namespace NUMINAMATH_CALUDE_yellow_tiles_count_l2849_284932

theorem yellow_tiles_count (total : ℕ) (purple : ℕ) (white : ℕ) 
  (h1 : total = 20)
  (h2 : purple = 6)
  (h3 : white = 7)
  : ∃ (yellow : ℕ), 
    yellow + (yellow + 1) + purple + white = total ∧ 
    yellow = 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_tiles_count_l2849_284932


namespace NUMINAMATH_CALUDE_custom_op_example_l2849_284998

/-- Custom binary operation ⊗ defined as a ⊗ b = a - |b| -/
def custom_op (a b : ℝ) : ℝ := a - abs b

/-- Theorem stating that 2 ⊗ (-3) = -1 -/
theorem custom_op_example : custom_op 2 (-3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2849_284998


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2849_284929

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_with_complement :
  A ∩ (U \ B) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2849_284929


namespace NUMINAMATH_CALUDE_digit_theta_value_l2849_284965

theorem digit_theta_value : ∃! (Θ : ℕ), 
  Θ > 0 ∧ Θ < 10 ∧ (252 : ℚ) / Θ = 30 + 2 * Θ := by
  sorry

end NUMINAMATH_CALUDE_digit_theta_value_l2849_284965


namespace NUMINAMATH_CALUDE_train_length_l2849_284910

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 174.98560115190784 →
  man_speed = 5 →
  crossing_time = 10 →
  ∃ (length : ℝ), abs (length - 499.96) < 0.01 ∧
    length = (train_speed + man_speed) * (1000 / 3600) * crossing_time :=
sorry

end NUMINAMATH_CALUDE_train_length_l2849_284910


namespace NUMINAMATH_CALUDE_exists_m_for_all_n_no_even_digits_l2849_284939

-- Define a function to check if a natural number has no even digits
def has_no_even_digits (k : ℕ) : Prop := sorry

-- State the theorem
theorem exists_m_for_all_n_no_even_digits :
  ∃ m : ℕ+, ∀ n : ℕ+, has_no_even_digits ((5 : ℕ) ^ n.val * m.val) := by sorry

end NUMINAMATH_CALUDE_exists_m_for_all_n_no_even_digits_l2849_284939


namespace NUMINAMATH_CALUDE_ed_lost_no_marbles_l2849_284913

def marbles_lost (ed_initial : ℕ) (ed_now : ℕ) (doug : ℕ) : ℕ :=
  ed_initial - ed_now

theorem ed_lost_no_marbles 
  (h1 : ∃ ed_initial : ℕ, ed_initial = doug + 12)
  (h2 : ∃ ed_now : ℕ, ed_now = 17)
  (h3 : doug = 5) :
  marbles_lost (doug + 12) 17 doug = 0 := by
  sorry

end NUMINAMATH_CALUDE_ed_lost_no_marbles_l2849_284913


namespace NUMINAMATH_CALUDE_next_five_even_sum_l2849_284907

/-- Given a sum of 5 consecutive even positive integers with one divisible by 13,
    the sum of the next 5 even consecutive integers is 50 more than the original sum. -/
theorem next_five_even_sum (a : ℕ) (x : ℕ) : 
  (∃ k : ℕ, x = 26 * k) →
  a = (x - 4) + (x - 2) + x + (x + 2) + (x + 4) →
  (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = a + 50 := by
sorry

end NUMINAMATH_CALUDE_next_five_even_sum_l2849_284907


namespace NUMINAMATH_CALUDE_solid_is_cone_l2849_284970

/-- Represents a three-dimensional solid -/
structure Solid where
  -- Add necessary fields

/-- Represents a view of a solid -/
inductive View
  | Front
  | Side
  | Top

/-- Represents a shape -/
inductive Shape
  | Cone
  | Pyramid
  | Prism
  | Cylinder

/-- Returns true if the given view of the solid is an equilateral triangle -/
def isEquilateralTriangle (s : Solid) (v : View) : Prop :=
  sorry

/-- Returns true if the given view of the solid is a circle with its center -/
def isCircleWithCenter (s : Solid) (v : View) : Prop :=
  sorry

/-- Returns true if the front and side view triangles have equal sides -/
def hasFrontSideEqualSides (s : Solid) : Prop :=
  sorry

/-- Determines the shape of the solid based on its properties -/
def determineShape (s : Solid) : Shape :=
  sorry

theorem solid_is_cone (s : Solid) 
  (h1 : isEquilateralTriangle s View.Front)
  (h2 : isEquilateralTriangle s View.Side)
  (h3 : hasFrontSideEqualSides s)
  (h4 : isCircleWithCenter s View.Top) :
  determineShape s = Shape.Cone :=
sorry

end NUMINAMATH_CALUDE_solid_is_cone_l2849_284970


namespace NUMINAMATH_CALUDE_pole_length_theorem_l2849_284948

/-- The length of a pole after two cuts -/
def pole_length_after_cuts (initial_length : ℝ) (first_cut_percentage : ℝ) (second_cut_percentage : ℝ) : ℝ :=
  initial_length * (1 - first_cut_percentage) * (1 - second_cut_percentage)

/-- Theorem stating that a 20-meter pole, after cuts of 30% and 25%, will be 10.5 meters long -/
theorem pole_length_theorem :
  pole_length_after_cuts 20 0.3 0.25 = 10.5 := by
  sorry

#eval pole_length_after_cuts 20 0.3 0.25

end NUMINAMATH_CALUDE_pole_length_theorem_l2849_284948


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2849_284973

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2849_284973


namespace NUMINAMATH_CALUDE_inequality_proof_l2849_284972

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2849_284972


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_solution_l2849_284944

theorem cubic_equation_one_real_solution :
  ∀ (a : ℝ), ∃! (x : ℝ), x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_solution_l2849_284944


namespace NUMINAMATH_CALUDE_hall_mat_expenditure_l2849_284999

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := floor_area + wall_area
  total_area * cost_per_sqm

/-- Theorem stating that the total expenditure for the given hall dimensions and mat cost is 19500 -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 5 30 = 19500 := by
  sorry

#eval total_expenditure 20 15 5 30

end NUMINAMATH_CALUDE_hall_mat_expenditure_l2849_284999


namespace NUMINAMATH_CALUDE_math_question_probability_l2849_284979

/-- The probability of drawing a math question in a quiz -/
theorem math_question_probability :
  let chinese_questions : ℕ := 2
  let math_questions : ℕ := 3
  let comprehensive_questions : ℕ := 4
  let total_questions : ℕ := chinese_questions + math_questions + comprehensive_questions
  (math_questions : ℚ) / (total_questions : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_math_question_probability_l2849_284979


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l2849_284963

theorem fruit_basket_problem :
  let oranges : ℕ := 15
  let peaches : ℕ := 9
  let pears : ℕ := 18
  let bananas : ℕ := 12
  let apples : ℕ := 24
  Nat.gcd oranges (Nat.gcd peaches (Nat.gcd pears (Nat.gcd bananas apples))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l2849_284963


namespace NUMINAMATH_CALUDE_renovation_cost_calculation_l2849_284986

def renovation_cost (hourly_rates : List ℝ) (hours_per_day : ℝ) (days : ℕ) 
  (meal_cost : ℝ) (material_cost : ℝ) (unexpected_costs : List ℝ) : ℝ :=
  let daily_labor_cost := hourly_rates.sum * hours_per_day
  let total_labor_cost := daily_labor_cost * days
  let total_meal_cost := meal_cost * hourly_rates.length * days
  let total_unexpected_cost := unexpected_costs.sum
  total_labor_cost + total_meal_cost + material_cost + total_unexpected_cost

theorem renovation_cost_calculation : 
  renovation_cost [15, 20, 18, 22] 8 10 10 2500 [750, 500, 400] = 10550 := by
  sorry

end NUMINAMATH_CALUDE_renovation_cost_calculation_l2849_284986


namespace NUMINAMATH_CALUDE_largest_number_problem_l2849_284960

theorem largest_number_problem (a b c d e : ℝ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  a + b = 32 →
  a + c = 36 →
  b + c = 37 →
  c + e = 48 →
  d + e = 51 →
  e = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2849_284960


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l2849_284955

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallest_divisible_by_1_to_10 : ℕ := 2520

/-- Predicate to check if a number is divisible by all integers from 1 to 10 -/
def divisible_by_1_to_10 (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i → i ≤ 10 → n % i = 0

theorem smallest_divisible_by_1_to_10_is_correct :
  (divisible_by_1_to_10 smallest_divisible_by_1_to_10) ∧
  (∀ n : ℕ, n > 0 → divisible_by_1_to_10 n → n ≥ smallest_divisible_by_1_to_10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l2849_284955


namespace NUMINAMATH_CALUDE_fib_div_by_five_l2849_284991

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_div_by_five (n : ℕ) : 5 ∣ n → 5 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_fib_div_by_five_l2849_284991


namespace NUMINAMATH_CALUDE_sunday_school_three_year_olds_l2849_284940

/-- The number of 4-year-olds in the Sunday school -/
def four_year_olds : ℕ := 20

/-- The number of 5-year-olds in the Sunday school -/
def five_year_olds : ℕ := 15

/-- The number of 6-year-olds in the Sunday school -/
def six_year_olds : ℕ := 22

/-- The average class size -/
def average_class_size : ℕ := 35

/-- The number of classes -/
def num_classes : ℕ := 2

theorem sunday_school_three_year_olds :
  ∃ (three_year_olds : ℕ),
    (three_year_olds + four_year_olds + five_year_olds + six_year_olds) / num_classes = average_class_size ∧
    three_year_olds = 13 := by
  sorry

end NUMINAMATH_CALUDE_sunday_school_three_year_olds_l2849_284940


namespace NUMINAMATH_CALUDE_absolute_prime_at_most_three_digits_l2849_284921

/-- A function that returns true if a positive integer is prime -/
def IsPrime (n : ℕ) : Prop := sorry

/-- A function that returns the set of distinct digits in a positive integer's decimal representation -/
def DistinctDigits (n : ℕ) : Finset ℕ := sorry

/-- A function that returns true if all permutations of a positive integer's digits are prime -/
def AllDigitPermutationsPrime (n : ℕ) : Prop := sorry

/-- Definition of an absolute prime -/
def IsAbsolutePrime (n : ℕ) : Prop :=
  n > 0 ∧ IsPrime n ∧ AllDigitPermutationsPrime n

theorem absolute_prime_at_most_three_digits (n : ℕ) :
  IsAbsolutePrime n → Finset.card (DistinctDigits n) ≤ 3 := by sorry

end NUMINAMATH_CALUDE_absolute_prime_at_most_three_digits_l2849_284921


namespace NUMINAMATH_CALUDE_arbor_day_planting_l2849_284967

theorem arbor_day_planting (class_average : ℝ) (girls_trees : ℝ) (boys_trees : ℝ) :
  class_average = 6 →
  girls_trees = 15 →
  (1 / boys_trees + 1 / girls_trees = 1 / class_average) →
  boys_trees = 10 := by
  sorry

end NUMINAMATH_CALUDE_arbor_day_planting_l2849_284967


namespace NUMINAMATH_CALUDE_students_above_120_l2849_284931

/-- Represents the probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- Represents the cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The math scores follow a normal distribution with mean 110 and some standard deviation σ -/
axiom score_distribution (σ : ℝ) (x : ℝ) : 
  normal_pdf 110 σ x = normal_pdf 110 σ x

/-- The probability of scoring between 100 and 110 is 0.2 -/
axiom prob_100_to_110 (σ : ℝ) : 
  normal_cdf 110 σ 110 - normal_cdf 110 σ 100 = 0.2

/-- The total number of students is 800 -/
def total_students : ℕ := 800

/-- Theorem: Given the conditions, 240 students will score above 120 -/
theorem students_above_120 (σ : ℝ) : 
  (1 - normal_cdf 110 σ 120) * total_students = 240 := by sorry

end NUMINAMATH_CALUDE_students_above_120_l2849_284931


namespace NUMINAMATH_CALUDE_bruce_payment_l2849_284969

/-- The amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1125 to the shopkeeper -/
theorem bruce_payment :
  total_amount 9 70 9 55 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l2849_284969


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2849_284964

/-- Given a geometric sequence with common ratio q > 0, prove that if S_2 = 3a_2 + 2 and S_4 = 3a_4 + 2, then a_1 = -1 -/
theorem geometric_sequence_problem (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_q_pos : q > 0)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (h_sum : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h_S2 : S 2 = 3 * a 2 + 2)
  (h_S4 : S 4 = 3 * a 4 + 2) :
  a 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2849_284964


namespace NUMINAMATH_CALUDE_prob_three_odd_in_six_rolls_prob_three_odd_in_six_rolls_correct_l2849_284909

/-- The probability of getting exactly 3 odd numbers when rolling a fair 6-sided die 6 times -/
theorem prob_three_odd_in_six_rolls : ℚ :=
  5/16

/-- Proves that the probability of getting exactly 3 odd numbers when rolling a fair 6-sided die 6 times is 5/16 -/
theorem prob_three_odd_in_six_rolls_correct : prob_three_odd_in_six_rolls = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_in_six_rolls_prob_three_odd_in_six_rolls_correct_l2849_284909


namespace NUMINAMATH_CALUDE_largest_angle_is_right_l2849_284917

/-- Given a triangle ABC with side lengths a, b, and c, where c = 5 and 
    sqrt(a-4) + (b-3)^2 = 0, the largest interior angle of the triangle is 90°. -/
theorem largest_angle_is_right (a b c : ℝ) 
  (h1 : c = 5)
  (h2 : Real.sqrt (a - 4) + (b - 3)^2 = 0) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ 
    θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))
            (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
                 (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_right_l2849_284917


namespace NUMINAMATH_CALUDE_linda_furniture_fraction_l2849_284983

/-- Proves that the fraction of Linda's savings spent on furniture is 3/5 -/
theorem linda_furniture_fraction (original_savings : ℚ) (tv_cost : ℚ) : 
  original_savings = 1000 →
  tv_cost = 400 →
  (original_savings - tv_cost) / original_savings = 3/5 := by
sorry

end NUMINAMATH_CALUDE_linda_furniture_fraction_l2849_284983


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2849_284971

theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  -- The triangle is isosceles with two 60° angles
  α = 60 ∧ β = 60 ∧ 
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 60°
  max α (max β γ) = 60 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2849_284971


namespace NUMINAMATH_CALUDE_local_value_in_product_l2849_284930

/-- The face value of a digit is the digit itself -/
def faceValue (d : ℕ) : ℕ := d

/-- The local value of a digit in a number is the digit multiplied by its place value -/
def localValue (d : ℕ) (placeValue : ℕ) : ℕ := d * placeValue

/-- The product of two numbers -/
def product (a b : ℕ) : ℕ := a * b

/-- Theorem: In the product of the face value of 7 and the local value of 6 in 7098060,
    the local value of 6 is 6000 -/
theorem local_value_in_product :
  let number : ℕ := 7098060
  let fv7 : ℕ := faceValue 7
  let lv6 : ℕ := localValue 6 1000
  let prod : ℕ := product fv7 lv6
  localValue 6 1000 = 6000 := by sorry

end NUMINAMATH_CALUDE_local_value_in_product_l2849_284930


namespace NUMINAMATH_CALUDE_min_value_parabola_vectors_l2849_284933

/-- Given a parabola y² = 2px where p > 0, prove that the minimum value of 
    |⃗OA + ⃗OB|² - |⃗AB|² for any two distinct points A and B on the parabola is -4p² -/
theorem min_value_parabola_vectors (p : ℝ) (hp : p > 0) :
  ∃ (min : ℝ), min = -4 * p^2 ∧
  ∀ (A B : ℝ × ℝ), A ≠ B →
  (A.2)^2 = 2 * p * A.1 →
  (B.2)^2 = 2 * p * B.1 →
  (A.1 + B.1)^2 + (A.2 + B.2)^2 - ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ min :=
by sorry


end NUMINAMATH_CALUDE_min_value_parabola_vectors_l2849_284933


namespace NUMINAMATH_CALUDE_bill_share_proof_l2849_284904

def total_bill : ℝ := 139.00
def num_people : ℕ := 9
def tip_percentage : ℝ := 0.10

theorem bill_share_proof :
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let share_per_person := total_with_tip / num_people
  ∃ ε > 0, |share_per_person - 16.99| < ε :=
by sorry

end NUMINAMATH_CALUDE_bill_share_proof_l2849_284904


namespace NUMINAMATH_CALUDE_exists_silver_division_l2849_284984

/-- Represents the relationship between the number of people and the amount of silver in the problem. -/
def silver_division (x y : ℕ) : Prop :=
  (6 * x - 6 = y) ∧ (5 * x + 5 = y)

/-- The theorem states that the silver_division relationship holds for some positive integers x and y. -/
theorem exists_silver_division : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ silver_division x y := by
  sorry

end NUMINAMATH_CALUDE_exists_silver_division_l2849_284984


namespace NUMINAMATH_CALUDE_consecutive_four_product_ending_l2849_284902

theorem consecutive_four_product_ending (n : ℕ) :
  ∃ (k : ℕ), (n * (n + 1) * (n + 2) * (n + 3) % 1000 = 24 ∧ k = n * (n + 1) * (n + 2) * (n + 3) / 1000) ∨
              (n * (n + 1) * (n + 2) * (n + 3) % 10 = 0 ∧ (k = n * (n + 1) * (n + 2) * (n + 3) / 10) ∧ k % 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_four_product_ending_l2849_284902


namespace NUMINAMATH_CALUDE_dance_step_time_ratio_l2849_284988

/-- Proves that the ratio of time spent on the third dance step to the combined time
    spent on the first and second steps is 1:1, given the specified conditions. -/
theorem dance_step_time_ratio :
  ∀ (time_step1 time_step2 time_step3 total_time : ℕ),
  time_step1 = 30 →
  time_step2 = time_step1 / 2 →
  total_time = 90 →
  total_time = time_step1 + time_step2 + time_step3 →
  time_step3 = time_step1 + time_step2 :=
by
  sorry

#check dance_step_time_ratio

end NUMINAMATH_CALUDE_dance_step_time_ratio_l2849_284988


namespace NUMINAMATH_CALUDE_expression_range_l2849_284954

theorem expression_range (a b c x : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3),
    y = (a * Real.cos x + b * Real.sin x + c) / Real.sqrt (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_range_l2849_284954


namespace NUMINAMATH_CALUDE_purely_imaginary_trajectory_l2849_284934

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def trajectory (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≠ y

theorem purely_imaginary_trajectory (x y : ℝ) :
  is_purely_imaginary ((x^2 + y^2 - 4 : ℝ) + (x - y) * I) ↔ trajectory x y :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_trajectory_l2849_284934


namespace NUMINAMATH_CALUDE_david_fewer_crunches_l2849_284949

/-- Represents the number of exercises done by a person -/
structure ExerciseCount where
  pushups : ℕ
  crunches : ℕ

/-- Given the exercise counts for David and Zachary, proves that David did 17 fewer crunches than Zachary -/
theorem david_fewer_crunches (david zachary : ExerciseCount) 
  (h1 : david.pushups = zachary.pushups + 40)
  (h2 : david.crunches < zachary.crunches)
  (h3 : zachary.pushups = 34)
  (h4 : zachary.crunches = 62)
  (h5 : david.crunches = 45) :
  zachary.crunches - david.crunches = 17 := by
  sorry


end NUMINAMATH_CALUDE_david_fewer_crunches_l2849_284949


namespace NUMINAMATH_CALUDE_solid_shapes_count_l2849_284912

-- Define the set of geometric shapes
inductive GeometricShape
  | Square
  | Cuboid
  | Circle
  | Sphere
  | Cone

-- Define a function to determine if a shape is solid
def isSolid (shape : GeometricShape) : Bool :=
  match shape with
  | GeometricShape.Square => false
  | GeometricShape.Cuboid => true
  | GeometricShape.Circle => false
  | GeometricShape.Sphere => true
  | GeometricShape.Cone => true

-- Define the list of given shapes
def givenShapes : List GeometricShape :=
  [GeometricShape.Square, GeometricShape.Cuboid, GeometricShape.Circle, GeometricShape.Sphere, GeometricShape.Cone]

-- Theorem statement
theorem solid_shapes_count :
  (givenShapes.filter isSolid).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_solid_shapes_count_l2849_284912


namespace NUMINAMATH_CALUDE_watermelon_count_l2849_284905

theorem watermelon_count (seeds_per_watermelon : ℕ) (total_seeds : ℕ) (h1 : seeds_per_watermelon = 100) (h2 : total_seeds = 400) :
  total_seeds / seeds_per_watermelon = 4 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_count_l2849_284905


namespace NUMINAMATH_CALUDE_triangular_array_sum_l2849_284920

theorem triangular_array_sum (N : ℕ) : 
  (N * (N + 1)) / 2 = 3003 → (N / 10 + N % 10) = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_l2849_284920


namespace NUMINAMATH_CALUDE_existence_of_c_l2849_284966

theorem existence_of_c (p r a b : ℤ) : 
  Prime p → 
  p ∣ (r^7 - 1) → 
  p ∣ (r + 1 - a^2) → 
  p ∣ (r^2 + 1 - b^2) → 
  ∃ c : ℤ, p ∣ (r^3 + 1 - c^2) := by
sorry

end NUMINAMATH_CALUDE_existence_of_c_l2849_284966


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2849_284977

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

theorem arithmetic_sequence_condition (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, a n = 2 * (n : ℝ) + 1) → is_arithmetic_sequence a ∧
  ∃ b : ℕ+ → ℝ, is_arithmetic_sequence b ∧ ∃ m : ℕ+, b m ≠ 2 * (m : ℝ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2849_284977


namespace NUMINAMATH_CALUDE_line_bisects_circle_l2849_284923

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*y - 8 = 0

/-- The equation of the line -/
def line_equation (x y b : ℝ) : Prop :=
  y = x + b

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-2, 2)

/-- The line bisects the circumference of the circle if it passes through the center -/
def bisects_circle (b : ℝ) : Prop :=
  let (cx, cy) := circle_center
  line_equation cx cy b

theorem line_bisects_circle (b : ℝ) :
  bisects_circle b → b = 4 := by sorry

end NUMINAMATH_CALUDE_line_bisects_circle_l2849_284923


namespace NUMINAMATH_CALUDE_min_ratio_of_circles_l2849_284924

/-- The locus M of point A -/
def locus_M (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ y ≠ 0

/-- Point B -/
def B : ℝ × ℝ := (-1, 0)

/-- Point C -/
def C : ℝ × ℝ := (1, 0)

/-- The area of the inscribed circle of triangle PBC -/
noncomputable def S₁ (P : ℝ × ℝ) : ℝ := sorry

/-- The area of the circumscribed circle of triangle PBC -/
noncomputable def S₂ (P : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem min_ratio_of_circles :
  ∀ P : ℝ × ℝ, locus_M P.1 P.2 → S₂ P / S₁ P ≥ 4 ∧ ∃ Q : ℝ × ℝ, locus_M Q.1 Q.2 ∧ S₂ Q / S₁ Q = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_of_circles_l2849_284924


namespace NUMINAMATH_CALUDE_third_person_gets_max_median_l2849_284918

/-- Represents the money distribution among three people -/
structure MoneyDistribution where
  person1 : ℕ
  person2 : ℕ
  person3 : ℕ

/-- The initial distribution of money -/
def initial_distribution : MoneyDistribution :=
  { person1 := 28, person2 := 72, person3 := 98 }

/-- The total amount of money -/
def total_money (d : MoneyDistribution) : ℕ :=
  d.person1 + d.person2 + d.person3

/-- Checks if a distribution is valid (sum equals total money) -/
def is_valid_distribution (d : MoneyDistribution) : Prop :=
  total_money d = total_money initial_distribution

/-- Checks if a number is the median of three numbers -/
def is_median (a b c m : ℕ) : Prop :=
  (a ≤ m ∧ m ≤ c) ∨ (c ≤ m ∧ m ≤ a)

/-- The maximum possible median after redistribution -/
def max_median : ℕ := 99

/-- Theorem: After redistribution to maximize the median, the third person ends up with $99 -/
theorem third_person_gets_max_median :
  ∃ (d : MoneyDistribution),
    is_valid_distribution d ∧
    is_median d.person1 d.person2 d.person3 max_median ∧
    d.person3 = max_median :=
  sorry

end NUMINAMATH_CALUDE_third_person_gets_max_median_l2849_284918


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l2849_284925

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with length 8 cm, breadth 6 cm, and surface area 432 cm² has a height of 12 cm -/
theorem cuboid_height_calculation (h : ℝ) : 
  cuboidSurfaceArea 8 6 h = 432 → h = 12 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_calculation_l2849_284925


namespace NUMINAMATH_CALUDE_circle_parameter_range_l2849_284981

theorem circle_parameter_range (a : ℝ) : 
  (∃ (h : ℝ) (k : ℝ) (r : ℝ), ∀ (x y : ℝ), 
    x^2 + y^2 + 2*x - 4*y + a + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) → 
  a < 4 := by
sorry

end NUMINAMATH_CALUDE_circle_parameter_range_l2849_284981


namespace NUMINAMATH_CALUDE_prob_even_sum_is_8_15_l2849_284989

def wheel1 : Finset ℕ := {1, 2, 3, 4, 5}
def wheel2 : Finset ℕ := {1, 2, 3}

def isEven (n : ℕ) : Bool := n % 2 = 0

def probEvenSum : ℚ :=
  (Finset.filter (fun (pair : ℕ × ℕ) => isEven (pair.1 + pair.2)) (wheel1.product wheel2)).card /
  (wheel1.card * wheel2.card : ℚ)

theorem prob_even_sum_is_8_15 : probEvenSum = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_8_15_l2849_284989


namespace NUMINAMATH_CALUDE_total_tissues_used_l2849_284911

/-- The number of tissues Carol had initially -/
def initial_tissues : ℕ := 97

/-- The number of tissues Carol had after use -/
def remaining_tissues : ℕ := 58

/-- The total number of tissues used by Carol and her friends -/
def tissues_used : ℕ := initial_tissues - remaining_tissues

theorem total_tissues_used :
  tissues_used = 39 :=
by sorry

end NUMINAMATH_CALUDE_total_tissues_used_l2849_284911


namespace NUMINAMATH_CALUDE_sum_of_products_l2849_284926

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 20) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 30 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2849_284926


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l2849_284976

theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_ratio : ℝ)
  (fair_hair_ratio : ℝ)
  (h1 : women_fair_hair_ratio = 0.1)
  (h2 : fair_hair_ratio = 0.25) :
  women_fair_hair_ratio / fair_hair_ratio = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l2849_284976


namespace NUMINAMATH_CALUDE_lines_no_common_points_implies_a_equals_negative_one_l2849_284943

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ → ℝ := λ x => (a^2 - a) * x + 1 - a
  line2 : ℝ → ℝ := λ x => 2 * x - 1

/-- The property that two lines have no points in common -/
def NoCommonPoints (l : TwoLines) : Prop :=
  ∀ x : ℝ, l.line1 x ≠ l.line2 x

/-- The theorem statement -/
theorem lines_no_common_points_implies_a_equals_negative_one (l : TwoLines) :
  NoCommonPoints l → l.a = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_no_common_points_implies_a_equals_negative_one_l2849_284943


namespace NUMINAMATH_CALUDE_largest_number_with_13_matchsticks_has_digit_sum_9_l2849_284961

/-- Represents the number of matchsticks needed to form each digit --/
def matchsticks_per_digit : Fin 10 → ℕ
| 0 => 6
| 1 => 2
| 2 => 5
| 3 => 5
| 4 => 4
| 5 => 5
| 6 => 6
| 7 => 3
| 8 => 7
| 9 => 6

/-- Represents a number as a list of digits --/
def Number := List (Fin 10)

/-- Calculates the sum of digits in a number --/
def sum_of_digits (n : Number) : ℕ :=
  n.foldl (fun acc d => acc + d.val) 0

/-- Calculates the total number of matchsticks used to form a number --/
def matchsticks_used (n : Number) : ℕ :=
  n.foldl (fun acc d => acc + matchsticks_per_digit d) 0

/-- Checks if a number is valid (uses exactly 13 matchsticks) --/
def is_valid_number (n : Number) : Prop :=
  matchsticks_used n = 13

/-- Compares two numbers lexicographically --/
def number_gt (a b : Number) : Prop :=
  match a, b with
  | [], [] => False
  | _ :: _, [] => True
  | [], _ :: _ => False
  | x :: xs, y :: ys => x > y ∨ (x = y ∧ number_gt xs ys)

/-- The main theorem to be proved --/
theorem largest_number_with_13_matchsticks_has_digit_sum_9 :
  ∃ (n : Number), is_valid_number n ∧
    (∀ (m : Number), is_valid_number m → number_gt n m ∨ n = m) ∧
    sum_of_digits n = 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_13_matchsticks_has_digit_sum_9_l2849_284961


namespace NUMINAMATH_CALUDE_solutions_satisfy_system_l2849_284951

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧
  10 * x^2 + 18 * x * y + 8 * y^2 = 6

/-- The solutions to the system of equations -/
def solutions : List (ℝ × ℝ) :=
  [(-1, 2), (-11, 14), (11, -14), (1, -2)]

/-- Theorem stating that the given points are solutions to the system -/
theorem solutions_satisfy_system :
  ∀ (p : ℝ × ℝ), p ∈ solutions → system p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_solutions_satisfy_system_l2849_284951


namespace NUMINAMATH_CALUDE_pet_store_problem_l2849_284914

/-- The number of ways to distribute pets among Alice, Bob, and Charlie -/
def pet_distribution_ways (num_puppies num_kittens num_hamsters : ℕ) : ℕ :=
  num_kittens * num_hamsters + num_hamsters * num_kittens

/-- Theorem stating the number of ways Alice, Bob, and Charlie can buy pets -/
theorem pet_store_problem :
  let num_puppies : ℕ := 20
  let num_kittens : ℕ := 4
  let num_hamsters : ℕ := 8
  pet_distribution_ways num_puppies num_kittens num_hamsters = 64 :=
by
  sorry

#eval pet_distribution_ways 20 4 8

end NUMINAMATH_CALUDE_pet_store_problem_l2849_284914


namespace NUMINAMATH_CALUDE_function_minimum_and_inequality_l2849_284982

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem function_minimum_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) 
  (hequal : ∃ x, f a b x = 1) : 
  (2*a + b = 2) ∧ 
  (∀ t : ℝ, a + 2*b ≥ t*a*b → t ≤ 9/2) ∧
  (∃ t : ℝ, t = 9/2 ∧ a + 2*b = t*a*b) :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_and_inequality_l2849_284982


namespace NUMINAMATH_CALUDE_spurs_rockets_basketballs_l2849_284992

/-- The number of basketballs for two teams given their player counts and basketballs per player -/
def combined_basketballs (x y z : ℕ) : ℕ := x * z + y * z

/-- Theorem: The combined number of basketballs for the Spurs and Rockets is 440 -/
theorem spurs_rockets_basketballs :
  let x : ℕ := 22  -- number of Spurs players
  let y : ℕ := 18  -- number of Rockets players
  let z : ℕ := 11  -- number of basketballs per player
  combined_basketballs x y z = 440 := by
  sorry

end NUMINAMATH_CALUDE_spurs_rockets_basketballs_l2849_284992


namespace NUMINAMATH_CALUDE_range_of_a_l2849_284936

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem range_of_a (h : A ⊆ B a) : a ∈ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2849_284936


namespace NUMINAMATH_CALUDE_right_triangle_arctans_l2849_284938

theorem right_triangle_arctans (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_angle : a^2 = b^2 + c^2) : 
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctans_l2849_284938


namespace NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l2849_284927

-- Define the types for 2D and 3D figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Rectangle
| Parallelogram

inductive SpaceFigure
| Parallelepiped

-- Define the property of being formed by translation
def FormedByTranslation (plane : PlaneFigure) (space : SpaceFigure) : Prop :=
  match plane, space with
  | PlaneFigure.Parallelogram, SpaceFigure.Parallelepiped => True
  | _, _ => False

-- Define the concept of being analogous
def Analogous (plane : PlaneFigure) (space : SpaceFigure) : Prop :=
  FormedByTranslation plane space

-- Theorem statement
theorem parallelogram_most_analogous_to_parallelepiped :
  ∀ (plane : PlaneFigure),
    Analogous plane SpaceFigure.Parallelepiped →
    plane = PlaneFigure.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l2849_284927


namespace NUMINAMATH_CALUDE_solve_equation_l2849_284946

theorem solve_equation (x : ℚ) : x / 4 * 5 + 10 - 12 = 48 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2849_284946


namespace NUMINAMATH_CALUDE_square_addition_l2849_284978

theorem square_addition (b : ℝ) : b^2 + b^2 = 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_addition_l2849_284978


namespace NUMINAMATH_CALUDE_solution_set_equality_l2849_284928

theorem solution_set_equality : 
  {x : ℝ | (x - 1) * (x - 2) ≤ 0} = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2849_284928


namespace NUMINAMATH_CALUDE_odd_divisors_of_180_l2849_284945

/-- The number of positive divisors of 180 that are not divisible by 2 -/
def count_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ ¬ 2 ∣ d) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 180 not divisible by 2 is 6 -/
theorem odd_divisors_of_180 : count_odd_divisors 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisors_of_180_l2849_284945


namespace NUMINAMATH_CALUDE_root_two_implies_a_and_other_root_always_real_roots_l2849_284996

-- Define the equation
def equation (x a : ℝ) : Prop := x^2 + a*x + a - 1 = 0

-- Theorem 1: If 2 is a root, then a = -1 and the other root is -1
theorem root_two_implies_a_and_other_root (a : ℝ) :
  equation 2 a → a = -1 ∧ equation (-1) a := by sorry

-- Theorem 2: The equation always has real roots
theorem always_real_roots (a : ℝ) :
  ∃ x : ℝ, equation x a := by sorry

end NUMINAMATH_CALUDE_root_two_implies_a_and_other_root_always_real_roots_l2849_284996


namespace NUMINAMATH_CALUDE_yards_mowed_l2849_284993

/-- The problem of calculating how many yards Christian mowed --/
theorem yards_mowed (perfume_price : ℕ) (christian_savings sue_savings : ℕ)
  (yard_price : ℕ) (dogs_walked dog_price : ℕ) (remaining : ℕ) :
  perfume_price = 50 →
  christian_savings = 5 →
  sue_savings = 7 →
  yard_price = 5 →
  dogs_walked = 6 →
  dog_price = 2 →
  remaining = 6 →
  (perfume_price - (christian_savings + sue_savings + dogs_walked * dog_price + remaining)) / yard_price = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_yards_mowed_l2849_284993


namespace NUMINAMATH_CALUDE_midpoint_chain_l2849_284903

/-- Given points A, B, C, D, E, F on a line segment, where:
    C is the midpoint of AB,
    D is the midpoint of AC,
    E is the midpoint of AD,
    F is the midpoint of AE,
    and AB = 64,
    prove that AF = 4. -/
theorem midpoint_chain (A B C D E F : ℝ) : 
  C = (A + B) / 2 →
  D = (A + C) / 2 →
  E = (A + D) / 2 →
  F = (A + E) / 2 →
  B - A = 64 →
  F - A = 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_chain_l2849_284903


namespace NUMINAMATH_CALUDE_rotten_oranges_percentage_l2849_284974

theorem rotten_oranges_percentage 
  (total_oranges : ℕ) 
  (total_bananas : ℕ) 
  (rotten_bananas_percentage : ℚ) 
  (good_fruits_percentage : ℚ) :
  total_oranges = 600 →
  total_bananas = 400 →
  rotten_bananas_percentage = 8 / 100 →
  good_fruits_percentage = 878 / 1000 →
  (total_oranges - (good_fruits_percentage * (total_oranges + total_bananas : ℚ) - rotten_bananas_percentage * total_bananas)) / total_oranges = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_rotten_oranges_percentage_l2849_284974


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2849_284906

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ n) ∧ 
  17 ∣ n ∧ n ≤ 99 ∧ n ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2849_284906


namespace NUMINAMATH_CALUDE_time_addition_theorem_l2849_284975

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time (8:00:00 a.m.) -/
def initialTime : Time :=
  { hours := 8, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 7500

/-- The expected final time (10:05:00 a.m.) -/
def expectedFinalTime : Time :=
  { hours := 10, minutes := 5, seconds := 0 }

theorem time_addition_theorem :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l2849_284975


namespace NUMINAMATH_CALUDE_wine_pouring_equivalence_l2849_284953

/-- Represents the state of the four glasses --/
structure GlassState :=
  (glass1 : ℕ)
  (glass2 : ℕ)
  (glass3 : ℕ)
  (glass4 : ℕ)

/-- Represents a single pouring operation --/
inductive PourOperation
  | pour1to2
  | pour1to3
  | pour1to4
  | pour2to1
  | pour2to3
  | pour2to4
  | pour3to1
  | pour3to2
  | pour3to4
  | pour4to1
  | pour4to2
  | pour4to3

/-- Applies a single pouring operation to a glass state --/
def applyOperation (state : GlassState) (op : PourOperation) (m n k : ℕ) : GlassState :=
  sorry

/-- Checks if a specific amount can be achieved in any glass --/
def canAchieveAmount (m n k s : ℕ) : Prop :=
  ∃ (operations : List PourOperation),
    let finalState := operations.foldl (λ state op => applyOperation state op m n k)
                        (GlassState.mk 0 0 0 (m + n + k))
    finalState.glass1 = s ∨ finalState.glass2 = s ∨ finalState.glass3 = s ∨ finalState.glass4 = s

/-- The main theorem stating the equivalence --/
theorem wine_pouring_equivalence (m n k : ℕ) :
  (∀ s : ℕ, s < m + n + k → canAchieveAmount m n k s) ↔ Nat.gcd m (Nat.gcd n k) = 1 :=
sorry

end NUMINAMATH_CALUDE_wine_pouring_equivalence_l2849_284953


namespace NUMINAMATH_CALUDE_expression_simplification_l2849_284935

theorem expression_simplification (x y z w : ℝ) :
  (x - (y - (z - w))) - ((x - y) - (z - w)) = 2*z - 2*w := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2849_284935


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l2849_284942

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (k : ℝ), 29 * 39 * x^4 + 4 = k * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l2849_284942


namespace NUMINAMATH_CALUDE_dance_attendance_problem_l2849_284962

/-- Represents the number of different dance pairs at a school dance. -/
def total_pairs : ℕ := 430

/-- Represents the number of boys the first girl danced with. -/
def first_girl_partners : ℕ := 12

/-- Calculates the number of boys a girl danced with based on her position. -/
def partners_for_girl (girl_position : ℕ) : ℕ :=
  first_girl_partners + girl_position - 1

/-- Calculates the total number of dance pairs for a given number of girls. -/
def sum_of_pairs (num_girls : ℕ) : ℕ :=
  (num_girls * (2 * first_girl_partners + num_girls - 1)) / 2

/-- Represents the problem of finding the number of girls and boys at the dance. -/
theorem dance_attendance_problem :
  ∃ (num_girls num_boys : ℕ),
    num_girls > 0 ∧
    num_boys = partners_for_girl num_girls ∧
    sum_of_pairs num_girls = total_pairs ∧
    num_girls = 20 ∧
    num_boys = 31 := by
  sorry

end NUMINAMATH_CALUDE_dance_attendance_problem_l2849_284962


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_l2849_284900

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The fiscal year starts on Tuesday, February 1 -/
def fiscalYearStart : Date := { day := 1, month := 2 }

/-- The day of the week for the fiscal year start -/
def fiscalYearStartDay : DayOfWeek := DayOfWeek.Tuesday

/-- Function to determine if a given date is a Terrific Tuesday -/
def isTerrificTuesday (d : Date) : Prop := sorry

/-- The first Terrific Tuesday after the fiscal year starts -/
def firstTerrificTuesday : Date := { day := 29, month := 3 }

/-- Theorem stating that the first Terrific Tuesday after the fiscal year starts is March 29 -/
theorem first_terrific_tuesday :
  isTerrificTuesday firstTerrificTuesday ∧
  (∀ d : Date, d.month < firstTerrificTuesday.month ∨ 
    (d.month = firstTerrificTuesday.month ∧ d.day < firstTerrificTuesday.day) → 
    ¬isTerrificTuesday d) :=
by sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_l2849_284900


namespace NUMINAMATH_CALUDE_tennis_players_count_l2849_284958

theorem tennis_players_count (total_members : ℕ) (badminton_players : ℕ) (both_players : ℕ) (neither_players : ℕ) :
  total_members = 30 →
  badminton_players = 16 →
  both_players = 7 →
  neither_players = 2 →
  ∃ (tennis_players : ℕ), tennis_players = 19 ∧
    tennis_players = total_members - neither_players - (badminton_players - both_players) := by
  sorry


end NUMINAMATH_CALUDE_tennis_players_count_l2849_284958


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2849_284997

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  (s + 2) * (s + 2) * (s - 2) = s^3 - 10 →
  s^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2849_284997


namespace NUMINAMATH_CALUDE_probability_nine_correct_zero_l2849_284901

/-- Represents a matching problem with n pairs -/
structure MatchingProblem (n : ℕ) where
  /-- The number of pairs to match -/
  pairs : ℕ
  /-- Assumption that the number of pairs is positive -/
  positive : 0 < pairs
  /-- Assumption that the number of pairs is equal to n -/
  eq_n : pairs = n

/-- The probability of randomly matching exactly k pairs correctly in a matching problem with n pairs -/
def probability_exact_match (n k : ℕ) (prob : MatchingProblem n) : ℚ :=
  sorry

/-- Theorem stating that the probability of randomly matching exactly 9 pairs correctly in a matching problem with 10 pairs is 0 -/
theorem probability_nine_correct_zero : 
  ∀ (prob : MatchingProblem 10), probability_exact_match 10 9 prob = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_nine_correct_zero_l2849_284901


namespace NUMINAMATH_CALUDE_max_sector_area_l2849_284950

/-- Sector represents a circular sector with radius and central angle -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The perimeter of a sector -/
def sectorPerimeter (s : Sector) : ℝ := s.radius * s.angle + 2 * s.radius

/-- The area of a sector -/
def sectorArea (s : Sector) : ℝ := 0.5 * s.radius^2 * s.angle

/-- Theorem: Maximum area of a sector with perimeter 40 -/
theorem max_sector_area (s : Sector) (h : sectorPerimeter s = 40) :
  sectorArea s ≤ 100 ∧ (sectorArea s = 100 ↔ s.angle = 2) := by sorry

end NUMINAMATH_CALUDE_max_sector_area_l2849_284950


namespace NUMINAMATH_CALUDE_green_peaches_count_l2849_284952

/-- The number of green peaches in a basket, given the number of red, yellow, and total green and yellow peaches. -/
def num_green_peaches (red : ℕ) (yellow : ℕ) (green_and_yellow : ℕ) : ℕ :=
  green_and_yellow - yellow

/-- Theorem stating that there are 6 green peaches in the basket. -/
theorem green_peaches_count :
  let red : ℕ := 5
  let yellow : ℕ := 14
  let green_and_yellow : ℕ := 20
  num_green_peaches red yellow green_and_yellow = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2849_284952


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2849_284980

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_sequence : arithmetic_sequence 3 5 20 = 98 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2849_284980


namespace NUMINAMATH_CALUDE_gcf_of_180_150_210_l2849_284941

theorem gcf_of_180_150_210 : Nat.gcd 180 (Nat.gcd 150 210) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_150_210_l2849_284941


namespace NUMINAMATH_CALUDE_proposition_relation_l2849_284916

theorem proposition_relation :
  (∀ x y : ℝ, x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≤ 4 ∧ x^2 + y^2 > 2*x) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_l2849_284916


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2849_284985

/-- Given a hyperbola with equation x²/9 - y² = 1, its asymptotes are y = x/3 and y = -x/3 -/
theorem hyperbola_asymptotes :
  let hyperbola := fun (x y : ℝ) => x^2 / 9 - y^2 = 1
  let asymptote1 := fun (x y : ℝ) => y = x / 3
  let asymptote2 := fun (x y : ℝ) => y = -x / 3
  (∀ x y, hyperbola x y → (asymptote1 x y ∨ asymptote2 x y)) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2849_284985


namespace NUMINAMATH_CALUDE_tarun_worked_days_l2849_284956

/-- Represents the number of days it takes for Arun and Tarun to complete the work together -/
def combined_days : ℝ := 10

/-- Represents the number of days it takes for Arun to complete the work alone -/
def arun_alone_days : ℝ := 60

/-- Represents the number of days Arun worked alone after Tarun left -/
def arun_remaining_days : ℝ := 36

/-- Represents the total amount of work to be done -/
def total_work : ℝ := 1

/-- Theorem stating that Tarun worked for 4 days before leaving -/
theorem tarun_worked_days : 
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < combined_days ∧ 
    (t / combined_days + arun_remaining_days / arun_alone_days = total_work) ∧ 
    t = 4 := by
  sorry


end NUMINAMATH_CALUDE_tarun_worked_days_l2849_284956


namespace NUMINAMATH_CALUDE_total_camp_attendance_l2849_284959

/-- The number of kids from Lawrence county who went to camp -/
def lawrence_camp : ℕ := 34044

/-- The number of kids from outside the county who attended the camp -/
def outside_camp : ℕ := 424944

/-- The total number of kids who attended the camp -/
def total_camp : ℕ := lawrence_camp + outside_camp

/-- Theorem stating that the total number of kids who attended the camp is 458988 -/
theorem total_camp_attendance : total_camp = 458988 := by
  sorry

end NUMINAMATH_CALUDE_total_camp_attendance_l2849_284959


namespace NUMINAMATH_CALUDE_prove_weights_l2849_284990

/-- Represents a weighing device that signals when the total weight is 46 kg -/
def WeighingDevice (weights : List Nat) : Bool :=
  weights.sum = 46

/-- Represents the set of ingots with weights from 1 to 13 kg -/
def Ingots : List Nat := List.range 13 |>.map (· + 1)

/-- Checks if a given list of weights is a subset of the Ingots -/
def IsValidSelection (selection : List Nat) : Bool :=
  selection.all (· ∈ Ingots) ∧ selection.length ≤ Ingots.length

theorem prove_weights :
  ∃ (selection1 selection2 : List Nat),
    IsValidSelection selection1 ∧
    IsValidSelection selection2 ∧
    WeighingDevice selection1 ∧
    WeighingDevice selection2 ∧
    (9 ∈ selection1 ∨ 9 ∈ selection2) ∧
    (10 ∈ selection1 ∨ 10 ∈ selection2) :=
  sorry

end NUMINAMATH_CALUDE_prove_weights_l2849_284990


namespace NUMINAMATH_CALUDE_strawberry_price_difference_l2849_284908

theorem strawberry_price_difference (sale_price regular_price : ℚ) : 
  (54 * sale_price = 216) →
  (54 * regular_price = 216 + 108) →
  regular_price - sale_price = 2 := by
sorry

end NUMINAMATH_CALUDE_strawberry_price_difference_l2849_284908


namespace NUMINAMATH_CALUDE_incorrect_proposition_statement_l2849_284995

theorem incorrect_proposition_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_statement_l2849_284995


namespace NUMINAMATH_CALUDE_sand_gravel_transport_l2849_284947

theorem sand_gravel_transport :
  ∃ (x y : ℕ), 3 * x + 5 * y = 20 ∧ ((x = 5 ∧ y = 1) ∨ (x = 0 ∧ y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_sand_gravel_transport_l2849_284947


namespace NUMINAMATH_CALUDE_pen_distribution_l2849_284957

theorem pen_distribution (num_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 828 →
  num_students = 4 →
  num_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = 4 * k :=
by sorry

end NUMINAMATH_CALUDE_pen_distribution_l2849_284957


namespace NUMINAMATH_CALUDE_simplify_logarithmic_expression_l2849_284968

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem simplify_logarithmic_expression :
  lg 5 * lg 20 - lg 2 * lg 50 - lg 25 = -lg 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_logarithmic_expression_l2849_284968


namespace NUMINAMATH_CALUDE_inequality_proof_l2849_284937

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_ineq : x + y + z ≥ x*y + y*z + z*x) : 
  x/(y*z) + y/(z*x) + z/(x*y) ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2849_284937


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2849_284922

theorem pentagon_angle_measure (Q R S T U : ℝ) :
  R = 120 ∧ S = 94 ∧ T = 115 ∧ U = 101 →
  Q + R + S + T + U = 540 →
  Q = 110 := by
sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l2849_284922


namespace NUMINAMATH_CALUDE_evaluate_expression_l2849_284994

theorem evaluate_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2849_284994


namespace NUMINAMATH_CALUDE_sin_225_degrees_l2849_284915

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l2849_284915
