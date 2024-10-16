import Mathlib

namespace NUMINAMATH_CALUDE_apothem_lateral_face_angle_l519_51984

/-- Given a regular triangular pyramid where the lateral edge forms an angle of 60° with the base plane,
    the sine of the angle between the apothem and the plane of the adjacent lateral face
    is equal to (3√3) / 13. -/
theorem apothem_lateral_face_angle (a : ℝ) (h : a > 0) :
  let β : ℝ := 60 * π / 180  -- Convert 60° to radians
  let lateral_edge_angle : ℝ := β
  let apothem : ℝ := a * Real.sqrt 13 / (2 * Real.sqrt 3)
  let perpendicular_distance : ℝ := a * Real.sqrt 3 / 8
  let sin_φ : ℝ := perpendicular_distance / apothem
  sin_φ = 3 * Real.sqrt 3 / 13 := by
  sorry


end NUMINAMATH_CALUDE_apothem_lateral_face_angle_l519_51984


namespace NUMINAMATH_CALUDE_infinite_composite_generators_l519_51941

theorem infinite_composite_generators :
  ∃ A : Set ℕ, Set.Infinite A ∧
    ∀ a ∈ A, ∀ n : ℕ, ∃ m1 m2 : ℕ,
      m1 > 1 ∧ m2 > 1 ∧ n^4 + a = m1 * m2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_composite_generators_l519_51941


namespace NUMINAMATH_CALUDE_prob_6_or_less_l519_51974

/-- The probability of an archer hitting 9 rings or more in one shot. -/
def p_9_or_more : ℝ := 0.5

/-- The probability of an archer hitting exactly 8 rings in one shot. -/
def p_8 : ℝ := 0.2

/-- The probability of an archer hitting exactly 7 rings in one shot. -/
def p_7 : ℝ := 0.1

/-- Theorem: The probability of an archer hitting 6 rings or less in one shot is 0.2. -/
theorem prob_6_or_less : 1 - (p_9_or_more + p_8 + p_7) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_prob_6_or_less_l519_51974


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt3_l519_51961

theorem sqrt_difference_equals_2sqrt3 : 
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt3_l519_51961


namespace NUMINAMATH_CALUDE_carol_initial_peanuts_l519_51999

/-- The number of peanuts Carol initially collected -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Carol's father gave her -/
def fathers_peanuts : ℕ := 5

/-- The total number of peanuts Carol has after receiving peanuts from her father -/
def total_peanuts : ℕ := 7

/-- Theorem: Carol initially collected 2 peanuts -/
theorem carol_initial_peanuts : 
  initial_peanuts + fathers_peanuts = total_peanuts ∧ initial_peanuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_carol_initial_peanuts_l519_51999


namespace NUMINAMATH_CALUDE_constant_term_expansion_l519_51934

theorem constant_term_expansion (a : ℝ) : 
  a > 0 → (∃ c : ℝ, c = 80 ∧ c = (5 : ℕ).choose 4 * a^4) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l519_51934


namespace NUMINAMATH_CALUDE_income_relationship_l519_51997

-- Define the incomes as real numbers
variable (juan_income tim_income mary_income : ℝ)

-- State the theorem
theorem income_relationship :
  tim_income = 0.6 * juan_income →
  mary_income = 1.5 * tim_income →
  mary_income = 0.9 * juan_income :=
by
  sorry

#check income_relationship

end NUMINAMATH_CALUDE_income_relationship_l519_51997


namespace NUMINAMATH_CALUDE_sequence_inequality_l519_51955

theorem sequence_inequality (a : ℕ → ℕ) (h1 : a 1 < a 2) 
  (h2 : ∀ k ≥ 3, a k = 4 * a (k - 1) - 3 * a (k - 2)) : 
  a 45 > 3^43 := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l519_51955


namespace NUMINAMATH_CALUDE_f_neg_nine_equals_neg_three_l519_51939

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_neg_nine_equals_neg_three
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = Real.sqrt x) :
  f (-9) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_nine_equals_neg_three_l519_51939


namespace NUMINAMATH_CALUDE_complement_A_in_U_l519_51942

def U : Set ℝ := {x | x ≤ 1}
def A : Set ℝ := {x | x < 0}

theorem complement_A_in_U : {x : ℝ | x ∈ U ∧ x ∉ A} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l519_51942


namespace NUMINAMATH_CALUDE_casper_candy_problem_l519_51978

theorem casper_candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_after_eating := day1_remaining - (day1_remaining / 4)
  let day2_remaining := day2_after_eating + 5 - 5
  day2_remaining = 10 → initial_candies = 58 := by
  sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l519_51978


namespace NUMINAMATH_CALUDE_steak_weight_l519_51957

/-- Given 15 pounds of beef cut into 20 equal steaks, prove that each steak weighs 12 ounces. -/
theorem steak_weight (total_pounds : ℕ) (num_steaks : ℕ) (ounces_per_pound : ℕ) : 
  total_pounds = 15 → 
  num_steaks = 20 → 
  ounces_per_pound = 16 → 
  (total_pounds * ounces_per_pound) / num_steaks = 12 := by
  sorry

end NUMINAMATH_CALUDE_steak_weight_l519_51957


namespace NUMINAMATH_CALUDE_min_value_of_function_l519_51936

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 ∧
  ∃ y > 0, y + 2 / (2 * y + 1) - 3 / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l519_51936


namespace NUMINAMATH_CALUDE_leo_commute_cost_l519_51963

theorem leo_commute_cost (total_cost : ℕ) (working_days : ℕ) (trips_per_day : ℕ) 
  (h1 : total_cost = 960)
  (h2 : working_days = 20)
  (h3 : trips_per_day = 2) :
  total_cost / (working_days * trips_per_day) = 24 := by
sorry

end NUMINAMATH_CALUDE_leo_commute_cost_l519_51963


namespace NUMINAMATH_CALUDE_hospital_staff_count_l519_51937

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) (h1 : total = 456) (h2 : doctor_ratio = 8) (h3 : nurse_ratio = 11) : 
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 264 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l519_51937


namespace NUMINAMATH_CALUDE_digit_subtraction_problem_l519_51998

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem digit_subtraction_problem :
  ∃ (F G D E H I : ℕ),
    is_digit F ∧ is_digit G ∧ is_digit D ∧ is_digit E ∧ is_digit H ∧ is_digit I ∧
    F ≠ G ∧ F ≠ D ∧ F ≠ E ∧ F ≠ H ∧ F ≠ I ∧
    G ≠ D ∧ G ≠ E ∧ G ≠ H ∧ G ≠ I ∧
    D ≠ E ∧ D ≠ H ∧ D ≠ I ∧
    E ≠ H ∧ E ≠ I ∧
    H ≠ I ∧
    F * 10 + G = 93 ∧
    D * 10 + E = 68 ∧
    H * 10 + I = 25 ∧
    (F * 10 + G) - (D * 10 + E) = H * 10 + I :=
by
  sorry

end NUMINAMATH_CALUDE_digit_subtraction_problem_l519_51998


namespace NUMINAMATH_CALUDE_vector_equality_l519_51945

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, 2)

theorem vector_equality : c = (1/2 : ℝ) • a - (3/2 : ℝ) • b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l519_51945


namespace NUMINAMATH_CALUDE_total_cost_of_purchase_leas_purchase_l519_51911

/-- The total cost of Léa's purchases is $28 given the prices and quantities of items she bought. -/
theorem total_cost_of_purchase : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun book_price binder_price notebook_price book_quantity binder_quantity notebook_quantity =>
    book_price * book_quantity +
    binder_price * binder_quantity +
    notebook_price * notebook_quantity = 28

/-- Léa's actual purchase -/
theorem leas_purchase : total_cost_of_purchase 16 2 1 1 3 6 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_purchase_leas_purchase_l519_51911


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element7_l519_51975

theorem pascal_triangle_row20_element7 : Nat.choose 20 6 = 38760 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element7_l519_51975


namespace NUMINAMATH_CALUDE_siblings_age_sum_l519_51981

/-- The age difference between each sibling -/
def age_gap : ℕ := 5

/-- The current age of the eldest sibling -/
def eldest_age : ℕ := 20

/-- The number of years into the future we're calculating -/
def years_ahead : ℕ := 10

/-- The total age of three siblings born 'age_gap' years apart, 
    where the eldest is currently 'eldest_age' years old, 
    after 'years_ahead' years -/
def total_age (age_gap eldest_age years_ahead : ℕ) : ℕ :=
  (eldest_age + years_ahead) + 
  (eldest_age - age_gap + years_ahead) + 
  (eldest_age - 2 * age_gap + years_ahead)

theorem siblings_age_sum : 
  total_age age_gap eldest_age years_ahead = 75 := by
  sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l519_51981


namespace NUMINAMATH_CALUDE_true_discount_calculation_l519_51952

/-- Calculates the true discount given the banker's gain, interest rate, and time period. -/
def true_discount (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  (bankers_gain * 100) / (interest_rate * time)

/-- Theorem stating that under the given conditions, the true discount is 55. -/
theorem true_discount_calculation :
  let bankers_gain : ℚ := 6.6
  let interest_rate : ℚ := 12
  let time : ℚ := 1
  true_discount bankers_gain interest_rate time = 55 := by
sorry

end NUMINAMATH_CALUDE_true_discount_calculation_l519_51952


namespace NUMINAMATH_CALUDE_total_boys_in_circle_l519_51910

/-- Given a circular arrangement of boys, this function checks if two positions are opposite --/
def areOpposite (n : ℕ) (pos1 pos2 : ℕ) : Prop :=
  pos2 - pos1 = n / 2 ∨ pos1 - pos2 = n / 2

/-- Theorem stating the total number of boys in the circular arrangement --/
theorem total_boys_in_circle (n : ℕ) : 
  areOpposite n 7 27 ∧ 
  areOpposite n 11 36 ∧ 
  areOpposite n 15 42 → 
  n = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_boys_in_circle_l519_51910


namespace NUMINAMATH_CALUDE_length_BC_l519_51962

-- Define the centers and radii of the circles
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def radius_A : ℝ := 7
def radius_B : ℝ := 4

-- Define the distance between centers A and B
def AB : ℝ := radius_A + radius_B

-- Define point C
def C : ℝ × ℝ := sorry

-- Define the distance AC
def AC : ℝ := AB + 2

-- Theorem to prove
theorem length_BC : ∃ (BC : ℝ), BC = 52 / 7 := by
  sorry

end NUMINAMATH_CALUDE_length_BC_l519_51962


namespace NUMINAMATH_CALUDE_no_integer_roots_l519_51924

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0)
  (h0 : Odd (c))
  (h1 : Odd (a + b + c)) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l519_51924


namespace NUMINAMATH_CALUDE_min_faces_to_paint_correct_faces_to_paint_less_than_total_l519_51902

/-- The minimum number of cube faces Vasya needs to paint to prevent Petya from assembling
    an nxnxn cube that is completely white on the outside, given n^3 white 1x1x1 cubes. -/
def min_faces_to_paint (n : ℕ) : ℕ :=
  match n with
  | 2 => 2
  | 3 => 12
  | _ => 0  -- undefined for other values of n

/-- Theorem stating the correct minimum number of faces to paint for n=2 and n=3 -/
theorem min_faces_to_paint_correct :
  (min_faces_to_paint 2 = 2) ∧ (min_faces_to_paint 3 = 12) :=
by sorry

/-- Helper function to calculate the total number of small cubes -/
def total_small_cubes (n : ℕ) : ℕ := n^3

/-- Theorem stating that the number of faces to paint is less than the total number of cube faces -/
theorem faces_to_paint_less_than_total (n : ℕ) :
  n = 2 ∨ n = 3 → min_faces_to_paint n < 6 * total_small_cubes n :=
by sorry

end NUMINAMATH_CALUDE_min_faces_to_paint_correct_faces_to_paint_less_than_total_l519_51902


namespace NUMINAMATH_CALUDE_set_operations_and_subset_condition_l519_51903

def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_subset_condition (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x < 4}) ∧
  (A ∪ (C a ∪ B) = {x | x < 4}) ∧
  (A ⊆ C a → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_condition_l519_51903


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l519_51983

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + 2 * x + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- State the theorem
theorem quadratic_roots_condition (m : ℝ) :
  has_two_distinct_real_roots m ↔ m < 2 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l519_51983


namespace NUMINAMATH_CALUDE_product_parity_two_numbers_product_even_three_numbers_l519_51954

-- Definition for two numbers
def sum_is_even (a b : ℤ) : Prop := ∃ k : ℤ, a + b = 2 * k

-- Theorem for two numbers
theorem product_parity_two_numbers (a b : ℤ) (h : sum_is_even a b) :
  (∃ m : ℤ, a * b = 2 * m) ∨ (∃ n : ℤ, a * b = 2 * n + 1) :=
sorry

-- Theorem for three numbers
theorem product_even_three_numbers (a b c : ℤ) :
  ∃ k : ℤ, a * b * c = 2 * k :=
sorry

end NUMINAMATH_CALUDE_product_parity_two_numbers_product_even_three_numbers_l519_51954


namespace NUMINAMATH_CALUDE_unique_lowest_degree_polynomial_l519_51935

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + n + 3

theorem unique_lowest_degree_polynomial :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ∧
  (∀ g : ℕ → ℕ, (g 0 = 3 ∧ g 1 = 7 ∧ g 2 = 21 ∧ g 3 = 51) →
    (∃ a b c d : ℕ, ∀ n, g n = a*n^3 + b*n^2 + c*n + d) →
    (∀ n, f n = g n)) :=
by sorry

end NUMINAMATH_CALUDE_unique_lowest_degree_polynomial_l519_51935


namespace NUMINAMATH_CALUDE_polygon_sides_l519_51953

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 3420 → ∃ n : ℕ, n = 21 ∧ sum_interior_angles = 180 * (n - 2) := by
  sorry

#check polygon_sides

end NUMINAMATH_CALUDE_polygon_sides_l519_51953


namespace NUMINAMATH_CALUDE_mango_selling_price_l519_51921

theorem mango_selling_price 
  (loss_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (profit_price : ℝ) :
  loss_percentage = 20 →
  profit_percentage = 5 →
  profit_price = 6.5625 →
  ∃ (actual_price : ℝ), 
    actual_price = (1 - loss_percentage / 100) * (profit_price / (1 + profit_percentage / 100)) ∧
    actual_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_mango_selling_price_l519_51921


namespace NUMINAMATH_CALUDE_expression_evaluation_l519_51916

theorem expression_evaluation : (3^2 - 5) / (0.08 * 7 + 2) = 1.5625 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l519_51916


namespace NUMINAMATH_CALUDE_min_value_expression_l519_51928

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 18) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 22 ∧
  ∃ x₀ > 4, (x₀ + 18) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 22 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l519_51928


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l519_51943

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 31/9 ∧ B = 5/9 ∧
  ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -2 →
  (4*x^2 + 7*x + 3) / (x^2 - 5*x - 14) = A / (x - 7) + B / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l519_51943


namespace NUMINAMATH_CALUDE_xy_reciprocal_l519_51951

theorem xy_reciprocal (x y : ℝ) 
  (h1 : x * y > 0) 
  (h2 : 1 / x + 1 / y = 15) 
  (h3 : (x + y) / 5 = 0.6) : 
  1 / (x * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_l519_51951


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_greater_than_one_l519_51929

/-- Given functions f and g, prove that if for any x₁ in [-1, 2], 
    there exists an x₂ in [0, 2] such that f(x₁) > g(x₂), then a > 1 -/
theorem function_inequality_implies_a_greater_than_one (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (0 : ℝ) 2, x₁^2 > 2^x₂ - a) → 
  a > 1 := by
  sorry

#check function_inequality_implies_a_greater_than_one

end NUMINAMATH_CALUDE_function_inequality_implies_a_greater_than_one_l519_51929


namespace NUMINAMATH_CALUDE_train_crossing_time_l519_51907

theorem train_crossing_time (speed1 speed2 length1 length2 : ℝ) 
  (h1 : speed1 = 110)
  (h2 : speed2 = 90)
  (h3 : length1 = 1.10)
  (h4 : length2 = 0.9)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0)
  (h7 : length1 > 0)
  (h8 : length2 > 0) :
  (length1 + length2) / (speed1 + speed2) * 60 = 0.6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l519_51907


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_37_l519_51964

theorem inverse_of_3_mod_37 : ∃ x : ℕ, x ≤ 36 ∧ (3 * x) % 37 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_37_l519_51964


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l519_51991

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 0 }

-- Theorem statement
theorem max_min_values_of_f :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 3) ∧
  (∃ x ∈ interval, f x = -17) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l519_51991


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l519_51979

/-- Calculates the total revenue from concert ticket sales --/
theorem concert_ticket_revenue :
  let full_price : ℕ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_discount_percent : ℕ := 40
  let second_discount_percent : ℕ := 15
  let total_attendees : ℕ := 56

  let first_group_revenue := first_group_size * (full_price * (100 - first_discount_percent) / 100)
  let second_group_revenue := second_group_size * (full_price * (100 - second_discount_percent) / 100)
  let remaining_attendees := total_attendees - first_group_size - second_group_size
  let remaining_revenue := remaining_attendees * full_price

  let total_revenue := first_group_revenue + second_group_revenue + remaining_revenue

  total_revenue = 980 := by
    sorry

end NUMINAMATH_CALUDE_concert_ticket_revenue_l519_51979


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l519_51965

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l519_51965


namespace NUMINAMATH_CALUDE_median_inequality_exists_l519_51980

/-- A dataset is represented as a list of real numbers -/
def Dataset := List ℝ

/-- The median of a dataset -/
def median (d : Dataset) : ℝ := sorry

/-- Count of values in a dataset less than a given value -/
def count_less_than (d : Dataset) (x : ℝ) : ℕ := sorry

/-- Count of values in a dataset greater than a given value -/
def count_greater_than (d : Dataset) (x : ℝ) : ℕ := sorry

/-- Theorem: There exists a dataset where the number of values greater than 
    the median is not equal to the number of values less than the median -/
theorem median_inequality_exists : 
  ∃ (d : Dataset), count_greater_than d (median d) ≠ count_less_than d (median d) := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_exists_l519_51980


namespace NUMINAMATH_CALUDE_florist_roses_l519_51908

theorem florist_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 37 → sold = 16 → picked = 19 → initial - sold + picked = 40 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l519_51908


namespace NUMINAMATH_CALUDE_peach_pie_customers_l519_51948

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of customers who ordered apple pie slices -/
def apple_customers : ℕ := 56

/-- Represents the total number of pies sold -/
def total_pies : ℕ := 15

/-- Theorem stating that the number of customers who ordered peach pie slices is 48 -/
theorem peach_pie_customers : 
  (total_pies * peach_slices) - (apple_customers / apple_slices * peach_slices) = 48 := by
  sorry

end NUMINAMATH_CALUDE_peach_pie_customers_l519_51948


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l519_51958

theorem sum_of_special_integers (x y : ℕ) 
  (h1 : x > y) 
  (h2 : x - y = 8) 
  (h3 : x * y = 168) : 
  x + y = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l519_51958


namespace NUMINAMATH_CALUDE_stating_sandy_comic_books_l519_51901

/-- 
Given a person with an initial number of comic books, who sells half of them and then buys more,
this function calculates the final number of comic books.
-/
def final_comic_books (initial : ℕ) (bought : ℕ) : ℕ :=
  initial / 2 + bought

/-- 
Theorem stating that if Sandy starts with 14 comic books, sells half, and buys 6 more,
she will end up with 13 comic books.
-/
theorem sandy_comic_books : final_comic_books 14 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_stating_sandy_comic_books_l519_51901


namespace NUMINAMATH_CALUDE_simplify_expression_1_l519_51927

theorem simplify_expression_1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l519_51927


namespace NUMINAMATH_CALUDE_line_plane_relationship_l519_51925

-- Define the necessary structures
structure Line :=
  (id : String)

structure Plane :=
  (id : String)

-- Define the relationships
def parallel (l : Line) (p : Plane) : Prop :=
  sorry

def incident (l : Line) (p : Plane) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line) : Prop :=
  sorry

def skew_lines (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) 
  (h1 : parallel a α) (h2 : incident b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l519_51925


namespace NUMINAMATH_CALUDE_russian_tennis_pairing_probability_l519_51947

theorem russian_tennis_pairing_probability :
  let total_players : ℕ := 10
  let russian_players : ℕ := 4
  let total_pairs : ℕ := total_players / 2
  let russian_pairs : ℕ := russian_players / 2
  let favorable_outcomes : ℕ := Nat.choose total_pairs russian_pairs
  let total_outcomes : ℕ := Nat.choose total_players russian_players
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 21 :=
by sorry

end NUMINAMATH_CALUDE_russian_tennis_pairing_probability_l519_51947


namespace NUMINAMATH_CALUDE_congruence_solution_l519_51918

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 38574 ≡ n [ZMOD 17] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l519_51918


namespace NUMINAMATH_CALUDE_adjacent_triangle_number_l519_51933

/-- Given a triangular arrangement of natural numbers where the k-th row 
    contains numbers from (k-1)^2 + 1 to k^2, if 267 is in one triangle, 
    then 301 is in the adjacent triangle that shares a horizontal side. -/
theorem adjacent_triangle_number : ∀ (k : ℕ),
  (k - 1)^2 + 1 ≤ 267 ∧ 267 ≤ k^2 →
  ∃ (n : ℕ), n ≤ k^2 - ((k - 1)^2 + 1) + 1 ∧
  301 = (k + 1)^2 - (n + k - 1) :=
by sorry

end NUMINAMATH_CALUDE_adjacent_triangle_number_l519_51933


namespace NUMINAMATH_CALUDE_boys_playing_basketball_l519_51949

/-- Given a class with the following properties:
  * There are 30 students in total
  * One-third of the students are girls
  * Three-quarters of the boys play basketball
  Prove that the number of boys who play basketball is 15 -/
theorem boys_playing_basketball (total_students : ℕ) (girls : ℕ) (boys : ℕ) (boys_playing : ℕ) : 
  total_students = 30 →
  girls = total_students / 3 →
  boys = total_students - girls →
  boys_playing = (3 * boys) / 4 →
  boys_playing = 15 := by
sorry

end NUMINAMATH_CALUDE_boys_playing_basketball_l519_51949


namespace NUMINAMATH_CALUDE_parabola_translation_leftward_shift_by_2_l519_51950

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- Theorem stating the translation
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) :=
by
  sorry

-- Theorem stating the leftward shift by 2 units
theorem leftward_shift_by_2 :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_leftward_shift_by_2_l519_51950


namespace NUMINAMATH_CALUDE_next_red_probability_l519_51970

/-- Represents the count of balls of each color -/
structure BallCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Represents the result of pulling out balls -/
structure PullResult where
  pulled : ℕ
  redBlueDifference : ℤ

/-- Calculates the probability of pulling a red ball next -/
def probabilityNextRed (initial : BallCount) (result : PullResult) : ℚ :=
  9/26

theorem next_red_probability 
  (initial : BallCount)
  (result : PullResult)
  (h1 : initial.red = 50)
  (h2 : initial.blue = 50)
  (h3 : initial.yellow = 30)
  (h4 : result.pulled = 65)
  (h5 : result.redBlueDifference = 5) :
  probabilityNextRed initial result = 9/26 := by
  sorry

end NUMINAMATH_CALUDE_next_red_probability_l519_51970


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l519_51959

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (6 * x + 12) = 0 ↔ x = 5 ∧ 6 * x + 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l519_51959


namespace NUMINAMATH_CALUDE_segment_lengths_l519_51938

-- Define the points on the number line
def A : ℝ := 2
def B : ℝ := -5
def C : ℝ := -2
def D : ℝ := 4

-- Define the length of a segment
def segmentLength (x y : ℝ) : ℝ := |y - x|

-- Theorem statement
theorem segment_lengths :
  segmentLength A B = 7 ∧ segmentLength C D = 6 := by
  sorry

end NUMINAMATH_CALUDE_segment_lengths_l519_51938


namespace NUMINAMATH_CALUDE_movie_outing_cost_is_36_l519_51912

/-- Represents the cost of a movie outing for a family -/
def MovieOutingCost (ticket_price : ℚ) (popcorn_ratio : ℚ) (soda_ratio : ℚ) 
  (num_tickets : ℕ) (num_popcorn : ℕ) (num_soda : ℕ) : ℚ :=
  let popcorn_price := ticket_price * popcorn_ratio
  let soda_price := popcorn_price * soda_ratio
  (ticket_price * num_tickets) + (popcorn_price * num_popcorn) + (soda_price * num_soda)

/-- Theorem stating that the total cost for the family's movie outing is $36 -/
theorem movie_outing_cost_is_36 : 
  MovieOutingCost 5 (80/100) (50/100) 4 2 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_movie_outing_cost_is_36_l519_51912


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l519_51914

theorem circles_tangent_internally (r₁ r₂ d : ℝ) :
  r₁ = 4 → r₂ = 7 → d = 3 → d = r₂ - r₁ := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l519_51914


namespace NUMINAMATH_CALUDE_percentage_invalid_votes_l519_51946

/-- The percentage of invalid votes in an election --/
theorem percentage_invalid_votes 
  (total_votes : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_a_valid_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 75 / 100)
  (h3 : candidate_a_valid_votes = 357000) :
  (1 - (candidate_a_valid_votes : ℚ) / (candidate_a_percentage * total_votes)) * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_invalid_votes_l519_51946


namespace NUMINAMATH_CALUDE_min_distance_line_ellipse_l519_51966

noncomputable def minDistance : ℝ := (24 - 2 * Real.sqrt 41) / 5

/-- The minimum distance between a point on the line 4x + 3y = 24
    and a point on the ellipse (x²/8) + (y²/4) = 1 is (24 - 2√41) / 5 -/
theorem min_distance_line_ellipse :
  let line := {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 = 24}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  ∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ),
    p₁ ∈ line ∧ p₂ ∈ ellipse ∧
    ∀ (q₁ : ℝ × ℝ) (q₂ : ℝ × ℝ),
      q₁ ∈ line → q₂ ∈ ellipse →
      Real.sqrt ((q₁.1 - q₂.1)^2 + (q₁.2 - q₂.2)^2) ≥ minDistance :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_ellipse_l519_51966


namespace NUMINAMATH_CALUDE_fruit_selling_results_l519_51913

/-- Represents the farmer's fruit selling scenario -/
structure FruitSelling where
  investment : ℝ
  total_yield : ℝ
  orchard_price : ℝ
  market_price : ℝ
  daily_market_sales : ℝ
  orchard_sales : ℝ

/-- The main theorem about the fruit selling scenario -/
theorem fruit_selling_results (s : FruitSelling)
  (h1 : s.investment = 13500)
  (h2 : s.total_yield = 19000)
  (h3 : s.orchard_price = 4)
  (h4 : s.market_price > 4)
  (h5 : s.daily_market_sales = 1000)
  (h6 : s.orchard_sales = 6000) :
  (s.total_yield / s.daily_market_sales = 19) ∧
  (s.total_yield * s.market_price - s.total_yield * s.orchard_price = 19000 * s.market_price - 76000) ∧
  (s.orchard_sales * s.orchard_price + (s.total_yield - s.orchard_sales) * s.market_price - s.investment = 13000 * s.market_price + 10500) := by
  sorry


end NUMINAMATH_CALUDE_fruit_selling_results_l519_51913


namespace NUMINAMATH_CALUDE_parabola_vertex_l519_51994

/-- The parabola defined by y = -x^2 + cx + d -/
noncomputable def parabola (c d : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + d

/-- The set of x values satisfying the inequality -x^2 + cx + d ≤ 0 -/
def inequality_solution (c d : ℝ) : Set ℝ := {x | x ∈ Set.Icc (-7) 3 ∪ Set.Ici 9}

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

theorem parabola_vertex (c d : ℝ) :
  (inequality_solution c d = {x | x ∈ Set.Icc (-7) 3 ∪ Set.Ici 9}) →
  (∃ (vertex : Vertex), vertex.x = 1 ∧ vertex.y = -62 ∧
    ∀ (x : ℝ), parabola c d x ≤ parabola c d vertex.x) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l519_51994


namespace NUMINAMATH_CALUDE_problem_solution_l519_51915

theorem problem_solution (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : 
  x^2 + y^2 = 697 ∧ x + y = Real.sqrt 769 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l519_51915


namespace NUMINAMATH_CALUDE_function_inequality_l519_51976

theorem function_inequality (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 8/5) :
  ∀ x : ℝ, a ≤ x ∧ x ≤ 2*a - 1 → |x + a| + |2*x - 3| ≤ |x + 3| := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l519_51976


namespace NUMINAMATH_CALUDE_root_shift_theorem_l519_51956

theorem root_shift_theorem (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 - 15*x^2 + 74*x - 120 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_theorem_l519_51956


namespace NUMINAMATH_CALUDE_product_without_x_cube_term_l519_51990

theorem product_without_x_cube_term (m : ℚ) : 
  (∀ a b c d : ℚ, (m * X^4 + a * X^3 + b * X^2 + c * X + d) = 
    (m * X^2 - 3 * X) * (X^2 - 2 * X - 1) → a = 0) → 
  m = -3/2 := by sorry

end NUMINAMATH_CALUDE_product_without_x_cube_term_l519_51990


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l519_51993

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (I)
theorem solution_set_part_i :
  let a : ℝ := -2
  let S := {x : ℝ | f a x + f a (2 * x) > 2}
  S = {x : ℝ | x < -2 ∨ x > -2/3} :=
sorry

-- Theorem for part (II)
theorem range_of_a_part_ii :
  ∀ a : ℝ, a < 0 →
  (∃ x : ℝ, f a x + f a (2 * x) < 1/2) →
  -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l519_51993


namespace NUMINAMATH_CALUDE_certain_number_existence_l519_51968

theorem certain_number_existence : ∃ N : ℕ, 
  N % 127 = 10 ∧ 2045 % 127 = 13 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_existence_l519_51968


namespace NUMINAMATH_CALUDE_brian_stones_l519_51900

theorem brian_stones (total : ℕ) (white black : ℕ) (h1 : total = 100) 
  (h2 : white + black = total) 
  (h3 : white * 60 = black * 40) 
  (h4 : white > black) : white = 40 := by
  sorry

end NUMINAMATH_CALUDE_brian_stones_l519_51900


namespace NUMINAMATH_CALUDE_toy_car_speed_l519_51926

theorem toy_car_speed (t s : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3) : s = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_speed_l519_51926


namespace NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l519_51960

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours -/
def hours : ℚ := 3.5

/-- Theorem: The number of minutes in 3.5 hours is 210 -/
theorem minutes_in_three_and_half_hours : 
  (hours * minutes_per_hour : ℚ) = 210 := by sorry

end NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l519_51960


namespace NUMINAMATH_CALUDE_no_non_divisor_exists_l519_51922

theorem no_non_divisor_exists (a : ℕ+) : ∃ (b n : ℕ+), a.val ∣ (b.val ^ n.val - n.val) := by
  sorry

end NUMINAMATH_CALUDE_no_non_divisor_exists_l519_51922


namespace NUMINAMATH_CALUDE_cube_side_ratio_l519_51906

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 49 / 1 → a / b = 7 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l519_51906


namespace NUMINAMATH_CALUDE_triangle_side_length_l519_51972

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l519_51972


namespace NUMINAMATH_CALUDE_tangent_angle_sum_l519_51909

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the problem setup
def circumcircle (t : Triangle) : Circle := sorry

def is_acute_angled (t : Triangle) : Prop := sorry

def is_tangent_to_side (c : Circle) (p1 p2 : Point) : Prop := sorry

def angle_between_tangents (c : Circle) (p : Point) : ℝ := sorry

-- Theorem statement
theorem tangent_angle_sum 
  (t : Triangle)
  (O : Point)
  (S_A S_B S_C : Circle) :
  is_acute_angled t →
  O = (circumcircle t).center →
  S_A.center = O ∧ S_B.center = O ∧ S_C.center = O →
  is_tangent_to_side S_A t.B t.C →
  is_tangent_to_side S_B t.C t.A →
  is_tangent_to_side S_C t.A t.B →
  angle_between_tangents S_A t.A + 
  angle_between_tangents S_B t.B + 
  angle_between_tangents S_C t.C = 180 :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_sum_l519_51909


namespace NUMINAMATH_CALUDE_spring_decrease_percentage_l519_51920

theorem spring_decrease_percentage 
  (fall_increase : Real) 
  (total_change : Real) 
  (h1 : fall_increase = 0.08) 
  (h2 : total_change = -0.1252) : 
  let initial := 100
  let after_fall := initial * (1 + fall_increase)
  let after_spring := initial * (1 + total_change)
  (after_fall - after_spring) / after_fall = 0.19 := by
sorry

end NUMINAMATH_CALUDE_spring_decrease_percentage_l519_51920


namespace NUMINAMATH_CALUDE_parents_average_age_l519_51977

theorem parents_average_age
  (num_grandparents num_parents num_grandchildren : ℕ)
  (avg_age_grandparents avg_age_grandchildren avg_age_family : ℚ)
  (h1 : num_grandparents = 2)
  (h2 : num_parents = 2)
  (h3 : num_grandchildren = 3)
  (h4 : avg_age_grandparents = 64)
  (h5 : avg_age_grandchildren = 6)
  (h6 : avg_age_family = 32)
  (h7 : (num_grandparents + num_parents + num_grandchildren : ℚ) * avg_age_family =
        num_grandparents * avg_age_grandparents +
        num_parents * (num_grandparents * avg_age_grandparents + num_parents * avg_age_family + num_grandchildren * avg_age_grandchildren - (num_grandparents + num_parents + num_grandchildren) * avg_age_family) / num_parents +
        num_grandchildren * avg_age_grandchildren) :
  (num_grandparents * avg_age_grandparents + num_parents * avg_age_family + num_grandchildren * avg_age_grandchildren - (num_grandparents + num_parents + num_grandchildren) * avg_age_family) / num_parents = 39 :=
sorry

end NUMINAMATH_CALUDE_parents_average_age_l519_51977


namespace NUMINAMATH_CALUDE_exam_percentage_l519_51919

theorem exam_percentage (total_students : ℕ) (assigned_avg makeup_avg overall_avg : ℚ) 
  (h1 : total_students = 100)
  (h2 : assigned_avg = 55 / 100)
  (h3 : makeup_avg = 95 / 100)
  (h4 : overall_avg = 67 / 100) :
  ∃ (x : ℚ), 
    0 ≤ x ∧ x ≤ 1 ∧
    x * assigned_avg + (1 - x) * makeup_avg = overall_avg ∧
    x = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_percentage_l519_51919


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l519_51940

theorem pet_store_bird_count :
  let num_cages : ℕ := 6
  let parrots_per_cage : ℕ := 6
  let parakeets_per_cage : ℕ := 2
  let total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)
  total_birds = 48 := by
sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l519_51940


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l519_51992

theorem exponential_function_sum_of_extrema (a : ℝ) : 
  (a > 0) → 
  (∀ x ∈ Set.Icc 0 1, ∃ y, y = a^x) →
  (a^1 + a^0 = 3) →
  (a = 2) := by
sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l519_51992


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l519_51969

/-- A parallelogram in 2D space --/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Check if a given point is a valid fourth vertex of a parallelogram --/
def isValidFourthVertex (p : Parallelogram) (point : ℝ × ℝ) : Prop :=
  point = (11, 4) ∨ point = (-1, 12) ∨ point = (3, -12)

/-- The main theorem --/
theorem parallelogram_fourth_vertex 
  (p : Parallelogram) 
  (h1 : p.v1 = (1, 0)) 
  (h2 : p.v2 = (5, 8)) 
  (h3 : p.v3 = (7, -4)) : 
  isValidFourthVertex p p.v4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l519_51969


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l519_51930

theorem sqrt_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * Real.sqrt (50 * x) = 60 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l519_51930


namespace NUMINAMATH_CALUDE_simplify_expression_l519_51996

theorem simplify_expression (x : ℝ) : 3 - (2 - (1 + (2 * (1 - (3 - 2*x))))) = 8 - 4*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l519_51996


namespace NUMINAMATH_CALUDE_tony_fever_threshold_l519_51973

/-- Calculates how many degrees above the fever threshold a person's temperature is -/
def degrees_above_fever_threshold (normal_temp fever_threshold temp_increase : ℝ) : ℝ :=
  (normal_temp + temp_increase) - fever_threshold

/-- Proves that Tony's temperature is 5 degrees above the fever threshold -/
theorem tony_fever_threshold :
  degrees_above_fever_threshold 95 100 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tony_fever_threshold_l519_51973


namespace NUMINAMATH_CALUDE_first_candidate_vote_percentage_l519_51988

/-- Proves that the first candidate received 80% of the votes in an election with two candidates -/
theorem first_candidate_vote_percentage
  (total_votes : ℕ)
  (second_candidate_votes : ℕ)
  (h_total : total_votes = 2400)
  (h_second : second_candidate_votes = 480) :
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_vote_percentage_l519_51988


namespace NUMINAMATH_CALUDE_additional_money_needed_l519_51985

def lee_money : ℚ := 10
def friend_money : ℚ := 8
def chicken_wings_cost : ℚ := 6
def chicken_salad_cost : ℚ := 4
def cheeseburger_cost : ℚ := 3.5
def fries_cost : ℚ := 2
def soda_cost : ℚ := 1
def coupon_discount : ℚ := 0.15
def tax_rate : ℚ := 0.08

def total_order_cost : ℚ := chicken_wings_cost + chicken_salad_cost + 2 * cheeseburger_cost + fries_cost + 2 * soda_cost

def discounted_cost : ℚ := total_order_cost * (1 - coupon_discount)

def final_cost : ℚ := discounted_cost * (1 + tax_rate)

def total_money : ℚ := lee_money + friend_money

theorem additional_money_needed :
  final_cost - total_money = 1.28 := by sorry

end NUMINAMATH_CALUDE_additional_money_needed_l519_51985


namespace NUMINAMATH_CALUDE_lineup_combinations_l519_51989

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of players in a triplet
def triplet_size : ℕ := 3

-- Define the number of triplet sets
def triplet_sets : ℕ := 2

-- Define the number of players to choose for the lineup
def lineup_size : ℕ := 7

-- Define the maximum number of players that can be chosen from a triplet set
def max_from_triplet : ℕ := 2

-- Define the function to calculate the number of ways to choose the lineup
def choose_lineup : ℕ := sorry

-- Theorem stating that the number of ways to choose the lineup is 21582
theorem lineup_combinations : choose_lineup = 21582 := by sorry

end NUMINAMATH_CALUDE_lineup_combinations_l519_51989


namespace NUMINAMATH_CALUDE_gcd_83_power_plus_one_l519_51904

theorem gcd_83_power_plus_one (h : Prime 83) : 
  Nat.gcd (83^9 + 1) (83^9 + 83^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_83_power_plus_one_l519_51904


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l519_51905

theorem smallest_number_with_given_remainders :
  ∃! x : ℕ,
    x > 0 ∧
    x % 5 = 2 ∧
    x % 4 = 2 ∧
    x % 6 = 3 ∧
    ∀ y : ℕ, y > 0 → y % 5 = 2 → y % 4 = 2 → y % 6 = 3 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l519_51905


namespace NUMINAMATH_CALUDE_vacation_pictures_l519_51971

theorem vacation_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ)
  (h1 : zoo_pics = 41)
  (h2 : museum_pics = 29)
  (h3 : deleted_pics = 15) :
  zoo_pics + museum_pics - deleted_pics = 55 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l519_51971


namespace NUMINAMATH_CALUDE_right_triangle_area_l519_51995

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≤ b) (h5 : b < c) (h6 : a + b = 13) (h7 : a = 5) (h8 : c^2 = a^2 + b^2) :
  (1/2) * a * b = 20 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l519_51995


namespace NUMINAMATH_CALUDE_loss_percentage_is_twenty_percent_l519_51982

-- Define the given conditions
def articles_sold_gain : ℕ := 20
def selling_price_gain : ℚ := 60
def gain_percentage : ℚ := 20 / 100

def articles_sold_loss : ℚ := 24.999996875000388
def selling_price_loss : ℚ := 50

-- Theorem to prove
theorem loss_percentage_is_twenty_percent :
  let cost_price := selling_price_gain / (1 + gain_percentage)
  let cost_per_article := cost_price / articles_sold_gain
  let cost_price_loss := cost_per_article * articles_sold_loss
  let loss := cost_price_loss - selling_price_loss
  loss / cost_price_loss = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_is_twenty_percent_l519_51982


namespace NUMINAMATH_CALUDE_example_quadratic_equation_l519_51986

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The equation x² + 2x - 1 = 0 is a quadratic equation -/
theorem example_quadratic_equation :
  is_quadratic_equation (λ x => x^2 + 2*x - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_example_quadratic_equation_l519_51986


namespace NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_negative_correlation_implies_negative_slope_positive_slope_implies_positive_correlation_negative_slope_implies_negative_correlation_l519_51932

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Determines if two variables are positively correlated based on the slope of their linear regression equation -/
def positively_correlated (eq : LinearRegression) : Prop :=
  eq.slope > 0

/-- Determines if two variables are negatively correlated based on the slope of their linear regression equation -/
def negatively_correlated (eq : LinearRegression) : Prop :=
  eq.slope < 0

/-- States that a linear regression equation with positive slope implies positive correlation -/
theorem positive_correlation_implies_positive_slope (eq : LinearRegression) :
  positively_correlated eq → eq.slope > 0 := by sorry

/-- States that a linear regression equation with negative slope implies negative correlation -/
theorem negative_correlation_implies_negative_slope (eq : LinearRegression) :
  negatively_correlated eq → eq.slope < 0 := by sorry

/-- States that positive slope implies positive correlation -/
theorem positive_slope_implies_positive_correlation (eq : LinearRegression) :
  eq.slope > 0 → positively_correlated eq := by sorry

/-- States that negative slope implies negative correlation -/
theorem negative_slope_implies_negative_correlation (eq : LinearRegression) :
  eq.slope < 0 → negatively_correlated eq := by sorry

end NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_negative_correlation_implies_negative_slope_positive_slope_implies_positive_correlation_negative_slope_implies_negative_correlation_l519_51932


namespace NUMINAMATH_CALUDE_total_acorns_formula_l519_51944

/-- The total number of acorns for Shawna, Sheila, Danny, and Ella -/
def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := sheila + y
  let ella := 2 * (danny - shawna)
  shawna + sheila + danny + ella

/-- Theorem stating the total number of acorns -/
theorem total_acorns_formula (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y := by
  sorry

end NUMINAMATH_CALUDE_total_acorns_formula_l519_51944


namespace NUMINAMATH_CALUDE_farm_field_theorem_l519_51923

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_hectares_per_day : ℕ
  actual_hectares_per_day : ℕ
  technical_delay_days : ℕ
  weather_delay_days : ℕ
  remaining_hectares : ℕ

/-- The solution to the farm field problem -/
def farm_field_solution (f : FarmField) : ℕ × ℕ :=
  let total_area := 1560
  let planned_days := 13
  (total_area, planned_days)

/-- Theorem stating the correctness of the farm field solution -/
theorem farm_field_theorem (f : FarmField)
    (h1 : f.planned_hectares_per_day = 120)
    (h2 : f.actual_hectares_per_day = 85)
    (h3 : f.technical_delay_days = 3)
    (h4 : f.weather_delay_days = 2)
    (h5 : f.remaining_hectares = 40) :
    farm_field_solution f = (1560, 13) := by
  sorry

#check farm_field_theorem

end NUMINAMATH_CALUDE_farm_field_theorem_l519_51923


namespace NUMINAMATH_CALUDE_tangent_line_at_2_l519_51987

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

-- Theorem statement
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), 
    (∀ x y, y = m*x + b ↔ x - y - 4 = 0) ∧ 
    (m = f' 2) ∧
    (f 2 = m*2 + b) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_l519_51987


namespace NUMINAMATH_CALUDE_innovative_numbers_l519_51917

def is_innovative (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a^2 - b^2

theorem innovative_numbers :
  is_innovative 31 ∧ is_innovative 41 ∧ is_innovative 16 ∧ ¬is_innovative 54 :=
by sorry

end NUMINAMATH_CALUDE_innovative_numbers_l519_51917


namespace NUMINAMATH_CALUDE_total_lives_calculation_video_game_lives_proof_l519_51931

theorem total_lives_calculation (initial_players : Nat) (additional_players : Nat) (lives_per_player : Nat) : Nat :=
  by
  -- Define the total number of players
  let total_players := initial_players + additional_players
  
  -- Calculate the total number of lives
  let total_lives := total_players * lives_per_player
  
  -- Prove that the total number of lives is 24
  have h : total_lives = 24 := by
    -- Replace with actual proof
    sorry
  
  -- Return the result
  exact total_lives

-- Define the specific values from the problem
def initial_friends : Nat := 2
def new_players : Nat := 2
def lives_per_player : Nat := 6

-- Theorem to prove the specific case
theorem video_game_lives_proof : 
  total_lives_calculation initial_friends new_players lives_per_player = 24 :=
by
  -- Replace with actual proof
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_video_game_lives_proof_l519_51931


namespace NUMINAMATH_CALUDE_candy_difference_l519_51967

/-- Represents the number of boxes -/
def num_boxes : ℕ := 10

/-- Represents the total number of candies in all boxes -/
def total_candies : ℕ := 320

/-- Represents the number of candies in the second box -/
def second_box_candies : ℕ := 11

/-- Calculates the sum of an arithmetic progression -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℕ) * d) / 2

/-- Theorem stating the common difference between consecutive boxes -/
theorem candy_difference : 
  ∃ (d : ℕ), 
    arithmetic_sum (second_box_candies - d : ℚ) d num_boxes = total_candies ∧ 
    d = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l519_51967
