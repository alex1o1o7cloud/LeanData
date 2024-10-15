import Mathlib

namespace NUMINAMATH_CALUDE_circle_radius_c_value_l1507_150741

theorem circle_radius_c_value :
  ∀ (c : ℝ),
  (∀ (x y : ℝ), x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 36) →
  c = -19 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_c_value_l1507_150741


namespace NUMINAMATH_CALUDE_function_property_l1507_150739

theorem function_property (f : ℝ → ℝ) (h : ¬(∀ x > 0, f x > 0)) : ∃ x > 0, f x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1507_150739


namespace NUMINAMATH_CALUDE_find_n_l1507_150704

theorem find_n (x y n : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1507_150704


namespace NUMINAMATH_CALUDE_total_cost_of_hats_l1507_150730

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- Theorem: The total cost of John's hats is $700 -/
theorem total_cost_of_hats : 
  weeks_of_different_hats * days_per_week * cost_per_hat = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_hats_l1507_150730


namespace NUMINAMATH_CALUDE_food_cost_theorem_l1507_150794

def sandwich_cost : ℝ := 4

def juice_cost (sandwich_cost : ℝ) : ℝ := 2 * sandwich_cost

def milk_cost (sandwich_cost juice_cost : ℝ) : ℝ :=
  0.75 * (sandwich_cost + juice_cost)

def total_cost (sandwich_cost juice_cost milk_cost : ℝ) : ℝ :=
  sandwich_cost + juice_cost + milk_cost

theorem food_cost_theorem :
  total_cost sandwich_cost (juice_cost sandwich_cost) (milk_cost sandwich_cost (juice_cost sandwich_cost)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_food_cost_theorem_l1507_150794


namespace NUMINAMATH_CALUDE_prob_different_fruits_l1507_150736

/-- The number of fruit types Joe can choose from -/
def num_fruit_types : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 4

/-- The probability of Joe eating the same fruit for all meals -/
def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

/-- The probability of Joe eating at least two different kinds of fruit in a day -/
theorem prob_different_fruits : (1 : ℚ) - prob_same_fruit = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_fruits_l1507_150736


namespace NUMINAMATH_CALUDE_arrangements_with_A_not_first_is_48_l1507_150717

/-- The number of ways to arrange 3 people from 5, including A and B, with A not at the head -/
def arrangements_with_A_not_first (total_people : ℕ) (selected_people : ℕ) : ℕ :=
  (total_people * (total_people - 1) * (total_people - 2)) -
  ((total_people - 1) * (total_people - 2))

/-- Theorem stating that the number of arrangements with A not at the head is 48 -/
theorem arrangements_with_A_not_first_is_48 :
  arrangements_with_A_not_first 5 3 = 48 := by
  sorry

#eval arrangements_with_A_not_first 5 3

end NUMINAMATH_CALUDE_arrangements_with_A_not_first_is_48_l1507_150717


namespace NUMINAMATH_CALUDE_millie_bracelets_l1507_150765

/-- The number of bracelets Millie had initially -/
def initial_bracelets : ℕ := 9

/-- The number of bracelets Millie lost -/
def lost_bracelets : ℕ := 2

/-- The number of bracelets Millie has left -/
def remaining_bracelets : ℕ := initial_bracelets - lost_bracelets

theorem millie_bracelets : remaining_bracelets = 7 := by
  sorry

end NUMINAMATH_CALUDE_millie_bracelets_l1507_150765


namespace NUMINAMATH_CALUDE_triangle_special_area_implies_angle_l1507_150711

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S of the triangle is (√3/4)(a² + b² - c²), then angle C measures π/3 --/
theorem triangle_special_area_implies_angle (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_area : (a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) / 2 = 
            (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_area_implies_angle_l1507_150711


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l1507_150712

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))
  (vertex_count : vertices.card = 20)
  (edge_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 5)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (I : Icosahedron) : ℚ :=
  (I.edges.card : ℚ) / (I.vertices.card.choose 2 : ℚ)

/-- The main theorem stating the probability is 10/19 -/
theorem icosahedron_edge_probability (I : Icosahedron) :
  edge_probability I = 10 / 19 := by
  sorry

end NUMINAMATH_CALUDE_icosahedron_edge_probability_l1507_150712


namespace NUMINAMATH_CALUDE_jelly_bean_multiple_l1507_150703

/-- The number of vanilla jelly beans -/
def vanilla_beans : ℕ := 120

/-- The total number of jelly beans -/
def total_beans : ℕ := 770

/-- The number of grape jelly beans as a function of the multiple -/
def grape_beans (x : ℕ) : ℕ := 50 + x * vanilla_beans

/-- The theorem stating that the multiple of vanilla jelly beans taken as grape jelly beans is 5 -/
theorem jelly_bean_multiple :
  ∃ x : ℕ, x = 5 ∧ vanilla_beans + grape_beans x = total_beans :=
sorry

end NUMINAMATH_CALUDE_jelly_bean_multiple_l1507_150703


namespace NUMINAMATH_CALUDE_marks_trees_l1507_150767

theorem marks_trees (current_trees planted_trees : ℕ) :
  current_trees = 13 → planted_trees = 12 →
  current_trees + planted_trees = 25 := by
  sorry

end NUMINAMATH_CALUDE_marks_trees_l1507_150767


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_15120_l1507_150746

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_15120 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ digit_product n = 15120 →
    n ≤ 98754 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_15120_l1507_150746


namespace NUMINAMATH_CALUDE_no_prime_power_solution_l1507_150784

theorem no_prime_power_solution : 
  ¬ ∃ (p : ℕ) (x : ℕ) (k : ℕ), 
    Nat.Prime p ∧ x^5 + 2*x + 3 = p^k :=
sorry

end NUMINAMATH_CALUDE_no_prime_power_solution_l1507_150784


namespace NUMINAMATH_CALUDE_sequence_floor_representation_l1507_150701

theorem sequence_floor_representation (a : Fin 1999 → ℕ) 
  (h : ∀ i j : Fin 1999, i + j < 1999 → a i + a 1 ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1) :
  ∃ x : ℝ, ∀ n : Fin 1999, a n = ⌊n * x⌋ := by sorry

end NUMINAMATH_CALUDE_sequence_floor_representation_l1507_150701


namespace NUMINAMATH_CALUDE_unique_solution_is_zero_l1507_150735

theorem unique_solution_is_zero :
  ∃! y : ℝ, y = 3 * (1 / y * (-y)) + 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_zero_l1507_150735


namespace NUMINAMATH_CALUDE_students_per_fourth_grade_class_l1507_150734

/-- Proves that the number of students in each fourth-grade class is 30 --/
theorem students_per_fourth_grade_class
  (total_cupcakes : ℕ)
  (pe_class_students : ℕ)
  (fourth_grade_classes : ℕ)
  (h1 : total_cupcakes = 140)
  (h2 : pe_class_students = 50)
  (h3 : fourth_grade_classes = 3)
  : (total_cupcakes - pe_class_students) / fourth_grade_classes = 30 := by
  sorry

#check students_per_fourth_grade_class

end NUMINAMATH_CALUDE_students_per_fourth_grade_class_l1507_150734


namespace NUMINAMATH_CALUDE_jude_matchbox_vehicles_l1507_150758

/-- Calculates the total number of matchbox vehicles Jude buys given the specified conditions -/
theorem jude_matchbox_vehicles :
  let car_cost : ℕ := 10
  let truck_cost : ℕ := 15
  let helicopter_cost : ℕ := 20
  let total_caps : ℕ := 250
  let trucks_bought : ℕ := 5
  let caps_spent_on_trucks : ℕ := trucks_bought * truck_cost
  let remaining_caps : ℕ := total_caps - caps_spent_on_trucks
  let caps_for_cars : ℕ := (remaining_caps * 60) / 100
  let cars_bought : ℕ := caps_for_cars / car_cost
  let caps_left : ℕ := remaining_caps - (cars_bought * car_cost)
  let helicopters_bought : ℕ := caps_left / helicopter_cost
  trucks_bought + cars_bought + helicopters_bought = 18 :=
by sorry

end NUMINAMATH_CALUDE_jude_matchbox_vehicles_l1507_150758


namespace NUMINAMATH_CALUDE_four_balls_three_boxes_l1507_150731

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 8 ways to put 4 distinguishable balls into 3 indistinguishable boxes -/
theorem four_balls_three_boxes : ways_to_put_balls_in_boxes 4 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_three_boxes_l1507_150731


namespace NUMINAMATH_CALUDE_equation_solution_l1507_150757

theorem equation_solution : ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 :=
  by
    use -30
    constructor
    · -- Prove that x = -30 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

end NUMINAMATH_CALUDE_equation_solution_l1507_150757


namespace NUMINAMATH_CALUDE_otimes_calculation_l1507_150797

-- Define the new operation ⊗
def otimes (a b : ℚ) : ℚ := a^2 - a*b

-- State the theorem
theorem otimes_calculation :
  otimes (-5) (otimes 3 (-2)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_otimes_calculation_l1507_150797


namespace NUMINAMATH_CALUDE_austin_robot_purchase_l1507_150799

/-- Proves that Austin bought robots for 7 friends given the problem conditions --/
theorem austin_robot_purchase (robot_cost : ℚ) (tax : ℚ) (change : ℚ) (initial_amount : ℚ) 
  (h1 : robot_cost = 8.75)
  (h2 : tax = 7.22)
  (h3 : change = 11.53)
  (h4 : initial_amount = 80) :
  (initial_amount - (change + tax)) / robot_cost = 7 := by
  sorry

#eval (80 : ℚ) - (11.53 + 7.22)
#eval ((80 : ℚ) - (11.53 + 7.22)) / 8.75

end NUMINAMATH_CALUDE_austin_robot_purchase_l1507_150799


namespace NUMINAMATH_CALUDE_quadratic_solution_average_l1507_150716

theorem quadratic_solution_average (c : ℝ) :
  c < 3 →  -- Condition for real and distinct solutions
  let equation := fun x : ℝ => 3 * x^2 - 6 * x + c
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ (x₁ + x₂) / 2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_solution_average_l1507_150716


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1507_150733

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1507_150733


namespace NUMINAMATH_CALUDE_largest_divisor_of_odd_product_l1507_150756

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ (m : ℕ), m ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → m ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_odd_product_l1507_150756


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l1507_150738

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧
  (∀ (x : ℝ), x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l1507_150738


namespace NUMINAMATH_CALUDE_f_2011_value_l1507_150718

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2011_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : ∀ x, f (x + 2) = -f x) 
  (h_def : ∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) : 
  f 2011 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2011_value_l1507_150718


namespace NUMINAMATH_CALUDE_deanna_speed_l1507_150777

/-- Proves that given the conditions of Deanna's trip, her speed in the first 30 minutes was 90 km/h -/
theorem deanna_speed (v : ℝ) : 
  (v * (1/2) + (v + 20) * (1/2) = 100) → 
  v = 90 := by
  sorry

end NUMINAMATH_CALUDE_deanna_speed_l1507_150777


namespace NUMINAMATH_CALUDE_sector_area_l1507_150745

theorem sector_area (centralAngle : Real) (radius : Real) : 
  centralAngle = 72 → radius = 20 → 
  (centralAngle / 360) * Real.pi * radius^2 = 80 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1507_150745


namespace NUMINAMATH_CALUDE_weight_difference_l1507_150775

theorem weight_difference (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 84 →
  (w_a + w_b + w_c + w_d) / 4 = 80 →
  (w_b + w_c + w_d + w_e) / 4 = 79 →
  w_a = 80 →
  w_e > w_d →
  w_e - w_d = 8 := by
sorry


end NUMINAMATH_CALUDE_weight_difference_l1507_150775


namespace NUMINAMATH_CALUDE_M₄_is_mutually_orthogonal_l1507_150792

/-- A set M is a mutually orthogonal point set if for all (x₁, y₁) in M,
    there exists (x₂, y₂) in M such that x₁x₂ + y₁y₂ = 0 -/
def MutuallyOrthogonalPointSet (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

/-- The set M₄ defined as {(x, y) | y = sin(x) + 1} -/
def M₄ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sin p.1 + 1}

/-- Theorem stating that M₄ is a mutually orthogonal point set -/
theorem M₄_is_mutually_orthogonal : MutuallyOrthogonalPointSet M₄ := by
  sorry

end NUMINAMATH_CALUDE_M₄_is_mutually_orthogonal_l1507_150792


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1507_150776

theorem polynomial_factorization (a : ℝ) : a^2 - 5*a - 6 = (a - 6) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1507_150776


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_zero_l1507_150762

theorem quadratic_solution_difference_squared_zero :
  ∀ a b : ℝ,
  (5 * a^2 - 30 * a + 45 = 0) →
  (5 * b^2 - 30 * b + 45 = 0) →
  (a - b)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_zero_l1507_150762


namespace NUMINAMATH_CALUDE_log_simplification_l1507_150700

theorem log_simplification (p q r s t z : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log ((p * t) / (s * z)) = Real.log (z / t) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l1507_150700


namespace NUMINAMATH_CALUDE_cayley_competition_certificates_l1507_150761

theorem cayley_competition_certificates (boys girls : ℕ) 
  (boys_percent girls_percent : ℚ) (h1 : boys = 30) (h2 : girls = 20) 
  (h3 : boys_percent = 1/10) (h4 : girls_percent = 1/5) : 
  (boys_percent * boys + girls_percent * girls) / (boys + girls) = 7/50 := by
  sorry

end NUMINAMATH_CALUDE_cayley_competition_certificates_l1507_150761


namespace NUMINAMATH_CALUDE_range_of_x_l1507_150743

theorem range_of_x (a b c x : ℝ) : 
  a^2 + 2*b^2 + 3*c^2 = 6 →
  a + 2*b + 3*c > |x + 1| →
  -7 < x ∧ x < 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l1507_150743


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l1507_150722

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) :
  diagonal = 10 →
  offset1 = 3 →
  area = 50 →
  area = (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * offset2 →
  offset2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l1507_150722


namespace NUMINAMATH_CALUDE_number_of_factors_of_60_l1507_150702

theorem number_of_factors_of_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_60_l1507_150702


namespace NUMINAMATH_CALUDE_fifth_minus_fourth_rectangles_l1507_150728

def rectangle_tiles (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem fifth_minus_fourth_rectangles : rectangle_tiles 5 - rectangle_tiles 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_fifth_minus_fourth_rectangles_l1507_150728


namespace NUMINAMATH_CALUDE_lcm_of_given_numbers_l1507_150778

theorem lcm_of_given_numbers : 
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60))) = 600 := by sorry

end NUMINAMATH_CALUDE_lcm_of_given_numbers_l1507_150778


namespace NUMINAMATH_CALUDE_function_symmetry_range_l1507_150783

open Real

theorem function_symmetry_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (1/ℯ) ℯ, a + 8 * log x = x^2 + 2) ↔ 
  a ∈ Set.Icc (6 - 8 * log 2) (10 + 1 / ℯ^2) :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_range_l1507_150783


namespace NUMINAMATH_CALUDE_bikers_meeting_time_l1507_150789

def biker1_time : ℕ := 12
def biker2_time : ℕ := 18
def biker3_time : ℕ := 24

theorem bikers_meeting_time :
  Nat.lcm (Nat.lcm biker1_time biker2_time) biker3_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_bikers_meeting_time_l1507_150789


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1507_150705

/-- Two vectors in R² -/
def Vector2 := ℝ × ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two vectors in R² -/
def perpendicular (v w : Vector2) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors_m_value :
  ∀ m : ℝ,
  let a : Vector2 := (1, 2)
  let b : Vector2 := (m, 1)
  perpendicular a b → m = -2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1507_150705


namespace NUMINAMATH_CALUDE_long_furred_dogs_count_l1507_150785

/-- Represents the characteristics of dogs in a kennel --/
structure KennelData where
  total_dogs : ℕ
  brown_dogs : ℕ
  neither_long_furred_nor_brown : ℕ
  long_furred_brown : ℕ

/-- Calculates the number of dogs with long fur in the kennel --/
def long_furred_dogs (data : KennelData) : ℕ :=
  data.long_furred_brown + (data.total_dogs - data.brown_dogs - data.neither_long_furred_nor_brown)

/-- Theorem stating the number of dogs with long fur in the specific kennel scenario --/
theorem long_furred_dogs_count (data : KennelData) 
  (h1 : data.total_dogs = 45)
  (h2 : data.brown_dogs = 30)
  (h3 : data.neither_long_furred_nor_brown = 8)
  (h4 : data.long_furred_brown = 19) :
  long_furred_dogs data = 26 := by
  sorry

#eval long_furred_dogs { total_dogs := 45, brown_dogs := 30, neither_long_furred_nor_brown := 8, long_furred_brown := 19 }

end NUMINAMATH_CALUDE_long_furred_dogs_count_l1507_150785


namespace NUMINAMATH_CALUDE_no_regular_lattice_polygon_except_square_l1507_150788

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A regular n-gon with vertices at lattice points -/
structure RegularLatticePolygon where
  n : ℕ
  vertices : Fin n → LatticePoint

/-- Predicate to check if a set of points forms a regular n-gon -/
def IsRegularPolygon (poly : RegularLatticePolygon) : Prop :=
  ∀ i j : Fin poly.n,
    (poly.vertices i).x ^ 2 + (poly.vertices i).y ^ 2 =
    (poly.vertices j).x ^ 2 + (poly.vertices j).y ^ 2

/-- Main theorem: No regular n-gon with vertices at lattice points exists for n ≠ 4 -/
theorem no_regular_lattice_polygon_except_square :
  ∀ n : ℕ, n ≠ 4 → ¬∃ (poly : RegularLatticePolygon), poly.n = n ∧ IsRegularPolygon poly :=
sorry

end NUMINAMATH_CALUDE_no_regular_lattice_polygon_except_square_l1507_150788


namespace NUMINAMATH_CALUDE_odd_function_positive_range_l1507_150740

open Set

def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_positive_range
  (f : ℝ → ℝ)
  (hf_odd : isOdd f)
  (hf_neg_one : f (-1) = 0)
  (hf_deriv : ∀ x > 0, x * (deriv^[2] f x) - deriv f x > 0) :
  {x : ℝ | f x > 0} = Ioo (-1) 0 ∪ Ioi 1 := by sorry

end NUMINAMATH_CALUDE_odd_function_positive_range_l1507_150740


namespace NUMINAMATH_CALUDE_cubic_equation_coefficient_l1507_150753

theorem cubic_equation_coefficient (a b : ℝ) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + 1 = (a * x - 1) * (x^2 - x - 1)) → 
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_coefficient_l1507_150753


namespace NUMINAMATH_CALUDE_point_A_final_position_l1507_150732

-- Define the initial position of point A
def initial_position : Set ℤ := {-5, 5}

-- Define the movement function
def move (start : ℤ) (left : ℤ) (right : ℤ) : ℤ := start - left + right

-- Theorem statement
theorem point_A_final_position :
  ∀ start ∈ initial_position,
  move start 2 6 = -1 ∨ move start 2 6 = 9 := by
sorry

end NUMINAMATH_CALUDE_point_A_final_position_l1507_150732


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1507_150798

theorem unique_integer_solution : ∃! (x y : ℤ), 10*x + 18*y = 28 ∧ 18*x + 10*y = 56 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1507_150798


namespace NUMINAMATH_CALUDE_meet_once_l1507_150780

/-- Represents the meeting of Michael and the garbage truck -/
structure MeetingProblem where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ
  initialDistance : ℝ

/-- Calculates the number of meetings between Michael and the truck -/
def numberOfMeetings (p : MeetingProblem) : ℕ :=
  sorry

/-- The specific problem instance -/
def problemInstance : MeetingProblem :=
  { michaelSpeed := 4
  , truckSpeed := 12
  , pailDistance := 300
  , truckStopTime := 40
  , initialDistance := 300 }

/-- Theorem stating that Michael and the truck meet exactly once -/
theorem meet_once : numberOfMeetings problemInstance = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l1507_150780


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1507_150737

noncomputable def f (x : ℝ) : ℝ := Real.exp x / (x + 2)

theorem f_derivative_at_zero : 
  deriv f 0 = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1507_150737


namespace NUMINAMATH_CALUDE_girls_boys_seating_arrangements_l1507_150723

theorem girls_boys_seating_arrangements (n : ℕ) (h : n = 5) : 
  (n.factorial * n.factorial : ℕ) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_girls_boys_seating_arrangements_l1507_150723


namespace NUMINAMATH_CALUDE_divisible_difference_exists_l1507_150719

theorem divisible_difference_exists (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end NUMINAMATH_CALUDE_divisible_difference_exists_l1507_150719


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1507_150721

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 30 players, where each player plays every other player exactly once,
    the total number of games played is 435. --/
theorem chess_tournament_games :
  num_games 30 = 435 := by
  sorry

#eval num_games 30  -- This will evaluate to 435

end NUMINAMATH_CALUDE_chess_tournament_games_l1507_150721


namespace NUMINAMATH_CALUDE_three_number_problem_l1507_150796

theorem three_number_problem :
  ∃ (x y z : ℝ),
    x = 45 ∧ y = 37.5 ∧ z = 22.5 ∧
    x - y = (1/3) * z ∧
    y - z = (1/3) * x ∧
    z - 10 = (1/3) * y :=
by
  sorry

end NUMINAMATH_CALUDE_three_number_problem_l1507_150796


namespace NUMINAMATH_CALUDE_company_fund_problem_l1507_150750

/-- Proves that the initial amount in the company fund was $950 given the problem conditions --/
theorem company_fund_problem (n : ℕ) : 
  (60 * n - 10 = 50 * n + 150) → 
  (60 * n - 10 = 950) :=
by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l1507_150750


namespace NUMINAMATH_CALUDE_julia_played_with_33_kids_l1507_150793

/-- The number of kids Julia played with on Monday and Tuesday combined -/
def total_kids_monday_tuesday (monday : ℕ) (tuesday : ℕ) : ℕ :=
  monday + tuesday

/-- Proof that Julia played with 33 kids on Monday and Tuesday combined -/
theorem julia_played_with_33_kids : 
  total_kids_monday_tuesday 15 18 = 33 := by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_33_kids_l1507_150793


namespace NUMINAMATH_CALUDE_largest_value_proof_l1507_150726

theorem largest_value_proof (a b c d e : ℚ) :
  a = 0.9387 →
  b = 0.9381 →
  c = 9385 / 10000 →
  d = 0.9379 →
  e = 0.9389 →
  max a (max b (max c (max d e))) = e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_proof_l1507_150726


namespace NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l1507_150763

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 60) 
  (h2 : math = 36) 
  (h3 : physics = 27) 
  (h4 : both = 20) : 
  total - (math + physics - both) = 17 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l1507_150763


namespace NUMINAMATH_CALUDE_train_speed_problem_l1507_150754

/-- Proves that the speed of the faster train is 31.25 km/hr given the problem conditions. -/
theorem train_speed_problem (v : ℝ) (h1 : v > 25) : 
  v = 31.25 ∧ 
  ∃ (t : ℝ), t > 0 ∧ 
    v * t + 25 * t = 630 ∧ 
    v * t = 25 * t + 70 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1507_150754


namespace NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l1507_150782

theorem sqrt_pi_squared_minus_6pi_plus_9 : 
  Real.sqrt (π^2 - 6*π + 9) = π - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l1507_150782


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_2083_l1507_150751

theorem units_digit_of_7_to_2083 : 7^2083 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_2083_l1507_150751


namespace NUMINAMATH_CALUDE_largest_prime_common_factor_l1507_150774

def is_largest_prime_common_factor (n : ℕ) : Prop :=
  n.Prime ∧
  n ∣ 462 ∧
  n ∣ 385 ∧
  ∀ m : ℕ, m.Prime → m ∣ 462 → m ∣ 385 → m ≤ n

theorem largest_prime_common_factor :
  is_largest_prime_common_factor 7 := by sorry

end NUMINAMATH_CALUDE_largest_prime_common_factor_l1507_150774


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l1507_150760

theorem tax_percentage_calculation (initial_bars : ℕ) (remaining_bars : ℕ) : 
  initial_bars = 60 →
  remaining_bars = 27 →
  ∃ (tax_percentage : ℚ),
    tax_percentage = 10 ∧
    remaining_bars = (initial_bars * (1 - tax_percentage / 100) / 2).floor :=
by sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l1507_150760


namespace NUMINAMATH_CALUDE_solve_proportion_l1507_150742

theorem solve_proportion (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_proportion_l1507_150742


namespace NUMINAMATH_CALUDE_triangle_shape_l1507_150749

theorem triangle_shape (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 + 2*b^2 = 2*b*(a+c) - c^2) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1507_150749


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1507_150770

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1507_150770


namespace NUMINAMATH_CALUDE_probability_score_3_points_l1507_150748

/-- The probability of hitting target A -/
def prob_hit_A : ℚ := 3/4

/-- The probability of hitting target B -/
def prob_hit_B : ℚ := 2/3

/-- The score for hitting target A -/
def score_hit_A : ℤ := 1

/-- The score for missing target A -/
def score_miss_A : ℤ := -1

/-- The score for hitting target B -/
def score_hit_B : ℤ := 2

/-- The score for missing target B -/
def score_miss_B : ℤ := 0

/-- The number of shots at target B -/
def shots_B : ℕ := 2

theorem probability_score_3_points : 
  (prob_hit_A * shots_B * prob_hit_B * (1 - prob_hit_B) + 
   (1 - prob_hit_A) * prob_hit_B^shots_B) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_score_3_points_l1507_150748


namespace NUMINAMATH_CALUDE_highest_score_is_174_l1507_150715

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  scoreDifference : ℕ
  averageExcludingExtremes : ℚ

/-- Calculates the highest score of a batsman given their statistics -/
def highestScore (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.overallAverage * stats.totalInnings
  let runsExcludingExtremes := stats.averageExcludingExtremes * (stats.totalInnings - 2)
  let sumExtremes := totalRuns - runsExcludingExtremes
  (sumExtremes + stats.scoreDifference) / 2

/-- Theorem stating that the highest score is 174 for the given statistics -/
theorem highest_score_is_174 (stats : BatsmanStats)
  (h1 : stats.totalInnings = 46)
  (h2 : stats.overallAverage = 60)
  (h3 : stats.scoreDifference = 140)
  (h4 : stats.averageExcludingExtremes = 58) :
  highestScore stats = 174 := by
  sorry

#eval highestScore {
  totalInnings := 46,
  overallAverage := 60,
  scoreDifference := 140,
  averageExcludingExtremes := 58
}

end NUMINAMATH_CALUDE_highest_score_is_174_l1507_150715


namespace NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l1507_150779

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by sorry

end NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l1507_150779


namespace NUMINAMATH_CALUDE_inequality_solution_l1507_150781

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / ((x - 3)^2) < 0 ↔ -3 < x ∧ x < 3 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1507_150781


namespace NUMINAMATH_CALUDE_cornelia_triple_kilee_age_l1507_150708

/-- The number of years in the future when Cornelia will be three times as old as Kilee -/
def future_years : ℕ := 10

/-- Kilee's current age -/
def kilee_age : ℕ := 20

/-- Cornelia's current age -/
def cornelia_age : ℕ := 80

/-- Theorem stating that in 'future_years' years, Cornelia will be three times as old as Kilee -/
theorem cornelia_triple_kilee_age :
  cornelia_age + future_years = 3 * (kilee_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_cornelia_triple_kilee_age_l1507_150708


namespace NUMINAMATH_CALUDE_u_2002_equals_2_l1507_150710

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 1
  | 3 => 3
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Default case for completeness

def u : ℕ → ℕ
  | 0 => 4
  | n + 1 => f (u n)

theorem u_2002_equals_2 : u 2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_u_2002_equals_2_l1507_150710


namespace NUMINAMATH_CALUDE_no_solution_when_p_is_seven_l1507_150772

theorem no_solution_when_p_is_seven (p : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - p) / (x - 8)) ↔ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_when_p_is_seven_l1507_150772


namespace NUMINAMATH_CALUDE_circle_equation_with_given_conditions_l1507_150744

/-- A circle with center (h, k) and radius r has the standard equation (x - h)² + (y - k)² = r² -/
def is_standard_circle_equation (h k r : ℝ) (f : ℝ → ℝ → Prop) :=
  ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- A point (x, y) lies on the line 2x - y = 3 -/
def lies_on_line (x y : ℝ) : Prop := 2*x - y = 3

/-- A circle is tangent to the x-axis if its distance to the x-axis equals its radius -/
def tangent_to_x_axis (h k r : ℝ) : Prop := |k| = r

/-- A circle is tangent to the y-axis if its distance to the y-axis equals its radius -/
def tangent_to_y_axis (h k r : ℝ) : Prop := |h| = r

theorem circle_equation_with_given_conditions :
  ∃ f : ℝ → ℝ → Prop,
    (∃ h k r : ℝ, 
      is_standard_circle_equation h k r f ∧
      lies_on_line h k ∧
      tangent_to_x_axis h k r ∧
      tangent_to_y_axis h k r) →
    (∀ x y, f x y ↔ ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_conditions_l1507_150744


namespace NUMINAMATH_CALUDE_sum_of_lengths_l1507_150724

-- Define the conversion factors
def meters_to_cm : ℝ := 100
def meters_to_mm : ℝ := 1000

-- Define the values in their original units
def length_m : ℝ := 2
def length_cm : ℝ := 3
def length_mm : ℝ := 5

-- State the theorem
theorem sum_of_lengths :
  length_m + length_cm / meters_to_cm + length_mm / meters_to_mm = 2.035 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_lengths_l1507_150724


namespace NUMINAMATH_CALUDE_max_x_minus_y_value_l1507_150720

theorem max_x_minus_y_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  ∃ (max : ℝ), max = 2 / Real.sqrt 3 ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → a - b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_value_l1507_150720


namespace NUMINAMATH_CALUDE_no_g_sequence_to_nine_l1507_150725

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n^3 + 9 else n / 2

def iterateG (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k + 1 => g (iterateG n k)

theorem no_g_sequence_to_nine :
  ∀ n : ℤ, -100 ≤ n ∧ n ≤ 100 → ¬∃ k : ℕ, iterateG n k = 9 :=
by sorry

end NUMINAMATH_CALUDE_no_g_sequence_to_nine_l1507_150725


namespace NUMINAMATH_CALUDE_value_of_x_l1507_150773

theorem value_of_x : ∀ w y z x : ℤ,
  w = 90 →
  z = w + 15 →
  y = z - 3 →
  x = y + 7 →
  x = 109 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l1507_150773


namespace NUMINAMATH_CALUDE_problem_solution_l1507_150787

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∩ B a = {2} → a = -1 ∨ a = -3) ∧
  (∀ a : ℝ, A ∪ B a = A → a ≤ -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1507_150787


namespace NUMINAMATH_CALUDE_H_div_G_equals_two_l1507_150766

-- Define the equation as a function
def equation (G H x : ℝ) : Prop :=
  G / (x + 5) + H / (x^2 - 4*x) = (x^2 - 2*x + 10) / (x^3 + x^2 - 20*x)

-- Define the theorem
theorem H_div_G_equals_two :
  ∀ G H : ℤ,
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4 → equation G H x) →
  (H : ℝ) / (G : ℝ) = 2 := by
  sorry


end NUMINAMATH_CALUDE_H_div_G_equals_two_l1507_150766


namespace NUMINAMATH_CALUDE_stephanie_store_visits_l1507_150769

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 16 / 2

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

theorem stephanie_store_visits : store_visits = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_stephanie_store_visits_l1507_150769


namespace NUMINAMATH_CALUDE_roots_quadratic_sum_l1507_150771

theorem roots_quadratic_sum (a b : ℝ) : 
  (a^2 + 3*a - 4 = 0) → 
  (b^2 + 3*b - 4 = 0) → 
  (a^2 + 4*a + b - 3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_sum_l1507_150771


namespace NUMINAMATH_CALUDE_tess_distance_graph_l1507_150759

-- Define the triangular block
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define Tess's position as a function of time
def tessPosition (t : ℝ) (tri : Triangle) : ℝ × ℝ :=
  sorry

-- Define the straight-line distance from A to Tess's position
def distanceFromA (t : ℝ) (tri : Triangle) : ℝ :=
  sorry

-- Define the properties of the distance function
def isRisingThenFalling (f : ℝ → ℝ) : Prop :=
  sorry

def peaksAtB (f : ℝ → ℝ) (tri : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem tess_distance_graph (tri : Triangle) :
  isRisingThenFalling (fun t => distanceFromA t tri) ∧
  peaksAtB (fun t => distanceFromA t tri) tri :=
sorry

end NUMINAMATH_CALUDE_tess_distance_graph_l1507_150759


namespace NUMINAMATH_CALUDE_parabola_circle_separation_l1507_150713

/-- The range of 'a' for a parabola y^2 = 4ax with directrix separate from the circle x^2 + y^2 - 2y = 0 -/
theorem parabola_circle_separation (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*a*x → x^2 + y^2 - 2*y ≠ 0) →
  (∀ x y : ℝ, x = a → x^2 + y^2 - 2*y ≠ 0) →
  a > 1 ∨ a < -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_separation_l1507_150713


namespace NUMINAMATH_CALUDE_max_leftover_candy_l1507_150752

theorem max_leftover_candy (x : ℕ) : ∃ (q r : ℕ), x = 10 * q + r ∧ r < 10 ∧ r ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_leftover_candy_l1507_150752


namespace NUMINAMATH_CALUDE_parabola_properties_l1507_150790

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * a * x - 5 * a

theorem parabola_properties :
  ∀ a : ℝ, a ≠ 0 →
  -- 1. Intersections with x-axis
  (∃ x : ℝ, parabola a x = 0 ↔ x = -1 ∨ x = 5) ∧
  -- 2. Conditions for a = 1
  (a > 0 → (∀ m n : ℝ, parabola a m = n → m ≥ 0 → n ≥ -9) → a = 1 ∧ 
    ∀ x : ℝ, parabola 1 x = x^2 - 4*x - 5) ∧
  -- 3. Range of m for shifted parabola
  (∀ m : ℝ, m > 0 → 
    (∃ t : ℝ, -1/2 < t ∧ t < 5/2 ∧ parabola 1 t + m = 0) →
    11/4 < m ∧ m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1507_150790


namespace NUMINAMATH_CALUDE_fraction_simplification_l1507_150786

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1507_150786


namespace NUMINAMATH_CALUDE_negation_of_root_existence_l1507_150791

theorem negation_of_root_existence :
  ¬(∀ a : ℝ, a > 0 → a ≠ 1 → ∃ x : ℝ, a^x - x - a = 0) ↔
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a^x - x - a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_root_existence_l1507_150791


namespace NUMINAMATH_CALUDE_lucas_100_mod5_l1507_150707

/-- The Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- The Lucas sequence modulo 5 -/
def lucas_mod5 (n : ℕ) : ℕ := lucas n % 5

/-- The cycle length of the Lucas sequence modulo 5 -/
def cycle_length : ℕ := 10

theorem lucas_100_mod5 :
  lucas_mod5 100 = 3 := by sorry

end NUMINAMATH_CALUDE_lucas_100_mod5_l1507_150707


namespace NUMINAMATH_CALUDE_siblings_ages_sum_l1507_150747

theorem siblings_ages_sum (x y z : ℕ+) 
  (h1 : y = x + 1)
  (h2 : x * y * z = 96) :
  x + y + z = 15 := by
sorry

end NUMINAMATH_CALUDE_siblings_ages_sum_l1507_150747


namespace NUMINAMATH_CALUDE_find_divisor_l1507_150729

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 166) 
  (h2 : quotient = 8) 
  (h3 : remainder = 6) 
  (h4 : dividend = quotient * (dividend / quotient) + remainder) : 
  dividend / quotient = 20 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1507_150729


namespace NUMINAMATH_CALUDE_milk_ratio_l1507_150755

/-- Given a cafeteria that sells two types of milk (regular and chocolate),
    this theorem proves the ratio of chocolate to regular milk cartons sold. -/
theorem milk_ratio (total : ℕ) (regular : ℕ) 
    (h1 : total = 24) 
    (h2 : regular = 3) : 
    (total - regular) / regular = 7 := by
  sorry

#check milk_ratio

end NUMINAMATH_CALUDE_milk_ratio_l1507_150755


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_three_l1507_150727

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_equals_negative_three :
  log10 4 - log10 4000 = -3 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_three_l1507_150727


namespace NUMINAMATH_CALUDE_average_salary_of_all_employees_l1507_150764

/-- Calculates the average salary of all employees in an office -/
theorem average_salary_of_all_employees 
  (officer_salary : ℝ) 
  (non_officer_salary : ℝ) 
  (num_officers : ℕ) 
  (num_non_officers : ℕ) 
  (h1 : officer_salary = 420)
  (h2 : non_officer_salary = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 450) :
  (officer_salary * num_officers + non_officer_salary * num_non_officers) / (num_officers + num_non_officers) = 120 :=
by
  sorry

#check average_salary_of_all_employees

end NUMINAMATH_CALUDE_average_salary_of_all_employees_l1507_150764


namespace NUMINAMATH_CALUDE_combined_tennis_preference_l1507_150706

theorem combined_tennis_preference (east_total : ℕ) (west_total : ℕ) 
  (east_tennis_percent : ℚ) (west_tennis_percent : ℚ) :
  east_total = 2000 →
  west_total = 2500 →
  east_tennis_percent = 22 / 100 →
  west_tennis_percent = 40 / 100 →
  (east_total * east_tennis_percent + west_total * west_tennis_percent) / 
  (east_total + west_total) = 32 / 100 :=
by sorry

end NUMINAMATH_CALUDE_combined_tennis_preference_l1507_150706


namespace NUMINAMATH_CALUDE_paint_calculation_l1507_150768

theorem paint_calculation (P : ℝ) 
  (h1 : (1/3 : ℝ) * P + (1/5 : ℝ) * (2/3 : ℝ) * P = 168) : 
  P = 360 :=
sorry

end NUMINAMATH_CALUDE_paint_calculation_l1507_150768


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l1507_150709

/-- The ratio of areas between an inscribed square and a larger square -/
theorem inscribed_square_area_ratio :
  let large_square_side : ℝ := 4
  let inscribed_square_horizontal_offset : ℝ := 1.5
  let inscribed_square_vertical_offset : ℝ := 4/3
  let inscribed_square_side : ℝ := large_square_side - 2 * inscribed_square_horizontal_offset
  let large_square_area : ℝ := large_square_side ^ 2
  let inscribed_square_area : ℝ := inscribed_square_side ^ 2
  inscribed_square_area / large_square_area = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l1507_150709


namespace NUMINAMATH_CALUDE_number_satisfying_proportion_l1507_150795

theorem number_satisfying_proportion : 
  let x : ℚ := 3
  (x + 1) / (x + 5) = (x + 5) / (x + 13) := by
sorry

end NUMINAMATH_CALUDE_number_satisfying_proportion_l1507_150795


namespace NUMINAMATH_CALUDE_servant_payment_is_40_l1507_150714

/-- Calculates the cash payment to a servant who leaves early -/
def servant_cash_payment (total_yearly_salary : ℚ) (turban_price : ℚ) (months_worked : ℚ) : ℚ :=
  (total_yearly_salary * (months_worked / 12)) - turban_price

/-- Proof that the servant receives Rs. 40 in cash -/
theorem servant_payment_is_40 :
  let total_yearly_salary : ℚ := 200
  let turban_price : ℚ := 110
  let months_worked : ℚ := 9
  servant_cash_payment total_yearly_salary turban_price months_worked = 40 := by
sorry

#eval servant_cash_payment 200 110 9

end NUMINAMATH_CALUDE_servant_payment_is_40_l1507_150714
