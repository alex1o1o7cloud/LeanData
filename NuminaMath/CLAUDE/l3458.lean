import Mathlib

namespace NUMINAMATH_CALUDE_test_scores_l3458_345851

/-- Given a test with 50 questions, each worth 2 marks, prove that the total combined score
    for three students (Meghan, Jose, and Alisson) is 210 marks, given the following conditions:
    - Meghan scored 20 marks less than Jose
    - Jose scored 40 more marks than Alisson
    - Jose got 5 questions wrong -/
theorem test_scores (total_questions : Nat) (marks_per_question : Nat)
    (meghan_jose_diff : Nat) (jose_alisson_diff : Nat) (jose_wrong : Nat) :
  total_questions = 50 →
  marks_per_question = 2 →
  meghan_jose_diff = 20 →
  jose_alisson_diff = 40 →
  jose_wrong = 5 →
  ∃ (meghan_score jose_score alisson_score : Nat),
    meghan_score + jose_score + alisson_score = 210 :=
by sorry


end NUMINAMATH_CALUDE_test_scores_l3458_345851


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l3458_345839

theorem system_of_inequalities_solution (x : ℝ) :
  (x - 3 * (x - 2) ≥ 4 ∧ 2 * x + 1 < x - 1) ↔ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l3458_345839


namespace NUMINAMATH_CALUDE_rectangle_area_l3458_345821

/-- The area of a rectangle bounded by y = a, y = -b, x = -c, and x = 2d, 
    where a, b, c, and d are positive numbers. -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) * (2 * d + c) = 2 * a * d + a * c + 2 * b * d + b * c := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l3458_345821


namespace NUMINAMATH_CALUDE_remainder_sum_l3458_345881

theorem remainder_sum (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5 = 5) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3458_345881


namespace NUMINAMATH_CALUDE_x_squared_plus_y_cubed_eq_neg_seven_l3458_345848

theorem x_squared_plus_y_cubed_eq_neg_seven 
  (x y : ℝ) 
  (h : |x - 1| + (y + 2)^2 = 0) : 
  x^2 + y^3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_cubed_eq_neg_seven_l3458_345848


namespace NUMINAMATH_CALUDE_museum_artifact_count_l3458_345895

/-- Represents a museum with paintings and artifacts --/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  large_painting_count : ℕ
  small_paintings_per_wing : ℕ
  artifact_multiplier : ℕ

/-- Calculates the number of artifacts in each artifact wing --/
def artifacts_per_wing (m : Museum) : ℕ :=
  let total_paintings := m.large_painting_count + (m.painting_wings - 1) * m.small_paintings_per_wing
  let total_artifacts := ((m.artifact_multiplier * total_paintings) / 8) * 8
  let artifact_wings := m.total_wings - m.painting_wings
  total_artifacts / artifact_wings

/-- Theorem: In the given museum setup, each artifact wing contains 34 artifacts --/
theorem museum_artifact_count (m : Museum) 
  (h1 : m.total_wings = 12)
  (h2 : m.painting_wings = 4)
  (h3 : m.large_painting_count = 1)
  (h4 : m.small_paintings_per_wing = 15)
  (h5 : m.artifact_multiplier = 6) :
  artifacts_per_wing m = 34 := by
  sorry

end NUMINAMATH_CALUDE_museum_artifact_count_l3458_345895


namespace NUMINAMATH_CALUDE_problem_solution_l3458_345810

theorem problem_solution (a b : ℝ) 
  (h1 : Real.log a + b = -2)
  (h2 : a ^ b = 10) : 
  a = (1 : ℝ) / 10 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3458_345810


namespace NUMINAMATH_CALUDE_expression_simplification_l3458_345840

theorem expression_simplification :
  Real.sqrt 5 * (5 ^ (1/2 : ℝ)) + 20 / 4 * 3 - 9 ^ (3/2 : ℝ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3458_345840


namespace NUMINAMATH_CALUDE_proportional_segment_length_l3458_345897

/-- Triangle ABC with sides a, b, c, and an interior point P creating parallel segments of length d -/
structure ProportionalTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The condition that the parallel segments split the sides proportionally -/
def is_proportional (t : ProportionalTriangle) : Prop :=
  t.d / t.c * t.b + t.d / t.a * t.b = t.b

/-- The theorem stating that for the given triangle, the proportional segments have length 28.25 -/
theorem proportional_segment_length 
  (t : ProportionalTriangle) 
  (h1 : t.a = 500) 
  (h2 : t.b = 550) 
  (h3 : t.c = 650) 
  (h4 : is_proportional t) : 
  t.d = 28.25 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segment_length_l3458_345897


namespace NUMINAMATH_CALUDE_sum_nth_from_both_ends_l3458_345867

/-- A set of consecutive integers -/
structure ConsecutiveIntegerSet where
  first : ℤ
  last : ℤ
  h_consecutive : last ≥ first

/-- The median of a set of consecutive integers -/
def median (s : ConsecutiveIntegerSet) : ℚ :=
  (s.first + s.last : ℚ) / 2

/-- The nth number from the beginning of the set -/
def nth_from_beginning (s : ConsecutiveIntegerSet) (n : ℕ) : ℤ :=
  s.first + n - 1

/-- The nth number from the end of the set -/
def nth_from_end (s : ConsecutiveIntegerSet) (n : ℕ) : ℤ :=
  s.last - n + 1

theorem sum_nth_from_both_ends (s : ConsecutiveIntegerSet) (n : ℕ) 
  (h_median : median s = 60) :
  nth_from_beginning s n + nth_from_end s n = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_nth_from_both_ends_l3458_345867


namespace NUMINAMATH_CALUDE_circle_C_equation_range_of_a_symmetry_condition_l3458_345814

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the chord line
def chord_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop := a * x - y + 5 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, 4)

-- Theorem 1: Prove that the equation represents circle C
theorem circle_C_equation :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (x y : ℝ), circle_C x y ↔ (x - m)^2 + y^2 = 25) ∧
  (∃ (x y : ℝ), chord_line x y ∧ circle_C x y ∧
    ∃ (x' y' : ℝ), chord_line x' y' ∧ circle_C x' y' ∧
    (x - x')^2 + (y - y')^2 = 4 * 17) :=
sorry

-- Theorem 2: Prove the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∃ (x y : ℝ), intersecting_line a x y ∧ circle_C x y) ↔
  (a < 0 ∨ a > 5/12) :=
sorry

-- Theorem 3: Prove the symmetry condition
theorem symmetry_condition :
  ∃ (a : ℝ), a = 3/4 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    intersecting_line a x₁ y₁ ∧ circle_C x₁ y₁ ∧
    intersecting_line a x₂ y₂ ∧ circle_C x₂ y₂ ∧
    x₁ ≠ x₂ →
    (x₁ + x₂) * (point_P.1 + 2) + (y₁ + y₂) * (point_P.2 - 4) = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_range_of_a_symmetry_condition_l3458_345814


namespace NUMINAMATH_CALUDE_phone_number_A_equals_9_l3458_345802

def phone_number (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  D % 3 = 0 ∧ E % 3 = 0 ∧ F % 3 = 0 ∧
  E = D - 3 ∧ F = E - 3 ∧
  J % 2 = 0 ∧ G = J + 3 ∧ H = J + 2 ∧ I = J + 1 ∧
  A + B + C = 15 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem phone_number_A_equals_9 :
  ∀ A B C D E F G H I J : ℕ,
  phone_number A B C D E F G H I J → A = 9 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_A_equals_9_l3458_345802


namespace NUMINAMATH_CALUDE_apartment_complexes_count_l3458_345832

/-- The maximum number of apartment complexes that can be built on a rectangular land -/
def max_apartment_complexes (land_width land_length complex_side : ℕ) : ℕ :=
  (land_width / complex_side) * (land_length / complex_side)

/-- Theorem stating the maximum number of apartment complexes that can be built -/
theorem apartment_complexes_count :
  max_apartment_complexes 262 185 18 = 140 := by
  sorry

end NUMINAMATH_CALUDE_apartment_complexes_count_l3458_345832


namespace NUMINAMATH_CALUDE_quadratic_general_form_l3458_345836

/-- Given a quadratic equation x² = 3x + 1, its general form is x² - 3x - 1 = 0 -/
theorem quadratic_general_form :
  (fun x : ℝ => x^2) = (fun x : ℝ => 3*x + 1) →
  (fun x : ℝ => x^2 - 3*x - 1) = (fun x : ℝ => 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l3458_345836


namespace NUMINAMATH_CALUDE_tangent_line_circle_l3458_345893

/-- A line is tangent to a circle if the distance from the center of the circle to the line is equal to the radius of the circle -/
def is_tangent (r : ℝ) : Prop :=
  r > 0 ∧ (r / Real.sqrt 2 = 2 * Real.sqrt r)

theorem tangent_line_circle (r : ℝ) : is_tangent r ↔ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l3458_345893


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l3458_345877

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l3458_345877


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3458_345856

theorem quadratic_inequality_condition (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c < 0) → (∀ x, a*x^2 + b*x + c > 0) ∧
  ¬(∀ x, a*x^2 + b*x + c > 0 → (a > 0 ∧ b^2 - 4*a*c < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3458_345856


namespace NUMINAMATH_CALUDE_mikes_earnings_l3458_345888

/-- Calculates the total earnings from selling working video games -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Mike's earnings from selling his working video games is $56 -/
theorem mikes_earnings : 
  calculate_earnings 16 8 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_mikes_earnings_l3458_345888


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3458_345817

theorem function_passes_through_point 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) - 2
  f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3458_345817


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l3458_345860

theorem max_value_theorem (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : 4 * u + 3 * v < 84) :
  u * v * (84 - 4 * u - 3 * v)^2 ≤ 259308 :=
sorry

theorem max_value_achieved (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : 4 * u + 3 * v < 84) :
  ∃ (u₀ v₀ : ℝ), u₀ > 0 ∧ v₀ > 0 ∧ 4 * u₀ + 3 * v₀ < 84 ∧
    u₀ * v₀ * (84 - 4 * u₀ - 3 * v₀)^2 = 259308 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l3458_345860


namespace NUMINAMATH_CALUDE_roberto_outfits_l3458_345884

/-- The number of different outfits Roberto can assemble -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ := trousers * shirts * jackets

/-- Theorem stating the number of outfits Roberto can assemble -/
theorem roberto_outfits : number_of_outfits 5 5 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l3458_345884


namespace NUMINAMATH_CALUDE_pizza_tip_percentage_l3458_345837

/-- Calculates the tip percentage for Harry's pizza order --/
theorem pizza_tip_percentage
  (large_pizza_cost : ℝ)
  (topping_cost : ℝ)
  (num_pizzas : ℕ)
  (toppings_per_pizza : ℕ)
  (total_cost_with_tip : ℝ)
  (h1 : large_pizza_cost = 14)
  (h2 : topping_cost = 2)
  (h3 : num_pizzas = 2)
  (h4 : toppings_per_pizza = 3)
  (h5 : total_cost_with_tip = 50)
  : (total_cost_with_tip - (num_pizzas * large_pizza_cost + num_pizzas * toppings_per_pizza * topping_cost)) /
    (num_pizzas * large_pizza_cost + num_pizzas * toppings_per_pizza * topping_cost) = 0.25 := by
  sorry


end NUMINAMATH_CALUDE_pizza_tip_percentage_l3458_345837


namespace NUMINAMATH_CALUDE_conditional_probability_A_given_B_l3458_345864

def group_A : List Nat := [76, 90, 84, 86, 81, 87, 86, 82, 85, 83]
def group_B : List Nat := [82, 84, 85, 89, 79, 80, 91, 89, 79, 74]

def total_students : Nat := group_A.length + group_B.length

def students_A_above_85 : Nat := (group_A.filter (λ x => x ≥ 85)).length
def students_B_above_85 : Nat := (group_B.filter (λ x => x ≥ 85)).length
def total_above_85 : Nat := students_A_above_85 + students_B_above_85

def P_B : Rat := total_above_85 / total_students
def P_AB : Rat := students_A_above_85 / total_students

theorem conditional_probability_A_given_B :
  P_AB / P_B = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_A_given_B_l3458_345864


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3458_345870

def f (x : ℤ) : ℤ := x^3 - 3*x^2 - 13*x + 15

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {-3, 1, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3458_345870


namespace NUMINAMATH_CALUDE_school_supplies_ratio_l3458_345862

/-- Proves the ratio of school supplies spending to remaining money after textbooks is 1:4 --/
theorem school_supplies_ratio (total : ℕ) (remaining : ℕ) : 
  total = 960 →
  remaining = 360 →
  (total - total / 2 - remaining) / (total / 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_ratio_l3458_345862


namespace NUMINAMATH_CALUDE_set_equality_l3458_345849

theorem set_equality (A B X : Set α) 
  (h1 : A ∩ X = B ∩ X)
  (h2 : A ∩ X = A ∩ B)
  (h3 : A ∪ B ∪ X = A ∪ B) : 
  X = A ∩ B := by
sorry

end NUMINAMATH_CALUDE_set_equality_l3458_345849


namespace NUMINAMATH_CALUDE_sum_of_roots_l3458_345800

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 18*a^2 + 75*a - 200 = 0)
  (hb : 8*b^3 - 72*b^2 - 350*b + 3200 = 0) : 
  a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3458_345800


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3458_345822

theorem imaginary_part_of_z (x y : ℝ) (h : (x - Complex.I) * Complex.I = y + 2 * Complex.I) :
  (x + y * Complex.I).im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3458_345822


namespace NUMINAMATH_CALUDE_committee_selection_l3458_345846

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 35) : Nat.choose n 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l3458_345846


namespace NUMINAMATH_CALUDE_min_dot_product_l3458_345890

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus
def focus : ℝ × ℝ := (0, 1)

-- Define the line passing through the focus
def line_through_focus (x y : ℝ) : Prop := y = x + 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x - 1

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Statement of the theorem
theorem min_dot_product :
  ∃ (M N : ℝ × ℝ),
    parabola M.1 M.2 ∧
    parabola N.1 N.2 ∧
    line_through_focus M.1 M.2 ∧
    line_through_focus N.1 N.2 ∧
    (∀ (P : ℝ × ℝ), tangent_line P.1 P.2 →
      dot_product (M.1 - P.1, M.2 - P.2) (N.1 - P.1, N.2 - P.2) ≥ -14) ∧
    (∃ (P : ℝ × ℝ), tangent_line P.1 P.2 ∧
      dot_product (M.1 - P.1, M.2 - P.2) (N.1 - P.1, N.2 - P.2) = -14) :=
by
  sorry

end NUMINAMATH_CALUDE_min_dot_product_l3458_345890


namespace NUMINAMATH_CALUDE_parallelogram_cut_slope_sum_l3458_345898

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Defines the specific parallelogram from the problem -/
def problemParallelogram : Parallelogram :=
  { v1 := { x := 15, y := 70 }
  , v2 := { x := 15, y := 210 }
  , v3 := { x := 45, y := 280 }
  , v4 := { x := 45, y := 140 }
  }

/-- A line through the origin with slope m/n -/
structure Line where
  m : ℕ
  n : ℕ
  coprime : Nat.Coprime m n

/-- Predicate to check if a line cuts the parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (l : Line) (p : Parallelogram) : Prop :=
  sorry -- Definition omitted for brevity

theorem parallelogram_cut_slope_sum :
  ∃ (l : Line), cutsIntoCongruentPolygons l problemParallelogram ∧ l.m + l.n = 41 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_cut_slope_sum_l3458_345898


namespace NUMINAMATH_CALUDE_grade_assignment_count_l3458_345815

theorem grade_assignment_count : 
  (Nat.choose 12 2) * (3^10) = 3906234 := by sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l3458_345815


namespace NUMINAMATH_CALUDE_eight_mile_taxi_cost_l3458_345896

/-- Calculates the cost of a taxi ride given the fixed cost, cost per mile, and distance traveled. -/
def taxi_cost (fixed_cost : ℚ) (cost_per_mile : ℚ) (distance : ℚ) : ℚ :=
  fixed_cost + cost_per_mile * distance

/-- Theorem: The cost of an 8-mile taxi ride with a $2.00 fixed cost and $0.30 per mile is $4.40. -/
theorem eight_mile_taxi_cost :
  taxi_cost 2 (3/10) 8 = 44/10 := by
  sorry

end NUMINAMATH_CALUDE_eight_mile_taxi_cost_l3458_345896


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3458_345838

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 3 * x * (x - 1) = 2 * (x + 2) + 8 ↔ a * x^2 + b * x + c = 0) ∧
    a = 3 ∧ b = -5 ∧ c = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3458_345838


namespace NUMINAMATH_CALUDE_kelly_apple_count_l3458_345875

/-- 
Theorem: Given Kelly's initial apple count and the number of additional apples picked,
prove that the total number of apples is the sum of these two quantities.
-/
theorem kelly_apple_count (initial_apples additional_apples : ℕ) :
  initial_apples = 56 →
  additional_apples = 49 →
  initial_apples + additional_apples = 105 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apple_count_l3458_345875


namespace NUMINAMATH_CALUDE_apartment_cost_splitting_l3458_345872

/-- The number of people splitting the cost of the new apartment -/
def num_people_splitting_cost : ℕ := 3

/-- John's two brothers -/
def num_brothers : ℕ := 2

/-- The total number of people splitting the cost is John plus his brothers -/
theorem apartment_cost_splitting :
  num_people_splitting_cost = 1 + num_brothers := by
  sorry

end NUMINAMATH_CALUDE_apartment_cost_splitting_l3458_345872


namespace NUMINAMATH_CALUDE_circular_garden_ratio_l3458_345892

theorem circular_garden_ratio : 
  let r : ℝ := 8
  let circumference := 2 * Real.pi * r
  let area := Real.pi * r^2
  circumference / area = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_circular_garden_ratio_l3458_345892


namespace NUMINAMATH_CALUDE_kiana_and_twins_ages_l3458_345818

theorem kiana_and_twins_ages (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_kiana_and_twins_ages_l3458_345818


namespace NUMINAMATH_CALUDE_student_count_l3458_345886

theorem student_count (total_pencils : ℕ) (pencils_per_student : ℕ) 
  (h1 : total_pencils = 18) 
  (h2 : pencils_per_student = 9) : 
  total_pencils / pencils_per_student = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3458_345886


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l3458_345899

theorem ice_cream_flavors (total_flavors : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : 
  total_flavors = 100 →
  tried_two_years_ago = total_flavors / 4 →
  tried_last_year = 2 * tried_two_years_ago →
  total_flavors - (tried_two_years_ago + tried_last_year) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l3458_345899


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3458_345891

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_cond : a 1 * a 5 = 4) :
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 ∨ a 1 * a 2 * a 3 * a 4 * a 5 = -32 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3458_345891


namespace NUMINAMATH_CALUDE_equal_intercepts_equation_not_in_second_quadrant_range_l3458_345869

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 + a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop :=
  ∃ k, k = -a - 2 ∧ k = (-a - 2) / (a + 1)

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  a = -1 ∨ (-(a + 1) > 0 ∧ -a - 2 ≤ 0)

-- Theorem for equal intercepts
theorem equal_intercepts_equation (a : ℝ) :
  equal_intercepts a → (a = 0 ∨ a = -2) :=
sorry

-- Theorem for not passing through the second quadrant
theorem not_in_second_quadrant_range (a : ℝ) :
  not_in_second_quadrant a → -2 ≤ a ∧ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_equation_not_in_second_quadrant_range_l3458_345869


namespace NUMINAMATH_CALUDE_original_room_length_l3458_345880

theorem original_room_length :
  ∀ (x : ℝ),
  (4 * ((x + 2) * 20) + 2 * ((x + 2) * 20) = 1800) →
  x = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_original_room_length_l3458_345880


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l3458_345894

theorem abs_inequality_equivalence (x : ℝ) : 2 ≤ |x - 5| ∧ |x - 5| ≤ 8 ↔ x ∈ Set.Icc (-3) 3 ∪ Set.Icc 7 13 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l3458_345894


namespace NUMINAMATH_CALUDE_f_at_negative_two_l3458_345820

def f (x : ℝ) : ℝ := 2*x^5 + 5*x^4 + 5*x^3 + 10*x^2 + 6*x + 1

theorem f_at_negative_two : f (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_two_l3458_345820


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3458_345824

/-- An isosceles triangle with sides a, b, and c, where b = c -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : b = c
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, t.a = 3 → t.b = 6 → perimeter t = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3458_345824


namespace NUMINAMATH_CALUDE_equal_money_in_40_days_l3458_345826

/-- The number of days it takes for Taehyung and Minwoo to have the same amount of money -/
def days_to_equal_money (taehyung_initial : ℕ) (minwoo_initial : ℕ) 
  (taehyung_daily : ℕ) (minwoo_daily : ℕ) : ℕ :=
  (taehyung_initial - minwoo_initial) / (minwoo_daily - taehyung_daily)

/-- Theorem stating that it takes 40 days for Taehyung and Minwoo to have the same amount of money -/
theorem equal_money_in_40_days :
  days_to_equal_money 12000 4000 300 500 = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_money_in_40_days_l3458_345826


namespace NUMINAMATH_CALUDE_min_triangles_cover_chessboard_l3458_345804

/-- Represents the area of an 8x8 chessboard with one corner square removed -/
def remaining_area : ℕ := 63

/-- Represents the maximum possible area of a single triangle that can fit in the corner -/
def max_triangle_area : ℚ := 7/2

/-- The minimum number of congruent triangles needed to cover the remaining area -/
def min_triangles : ℕ := 18

/-- Theorem stating that the minimum number of congruent triangles needed to cover
    the remaining area of the chessboard is 18 -/
theorem min_triangles_cover_chessboard :
  (remaining_area : ℚ) / max_triangle_area = min_triangles := by sorry

end NUMINAMATH_CALUDE_min_triangles_cover_chessboard_l3458_345804


namespace NUMINAMATH_CALUDE_f_properties_l3458_345852

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin x * Real.cos x + (1 + Real.tan x ^ 2) * Real.cos x ^ 2

theorem f_properties : 
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), f x ≤ 3/2) ∧ 
  (∃ (x : ℝ), f x = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3458_345852


namespace NUMINAMATH_CALUDE_julies_salary_l3458_345883

/-- Calculates the monthly salary for a worker given specific conditions -/
def monthlySalary (hourlyRate : ℕ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) (missedDays : ℕ) : ℕ :=
  let dailyEarnings := hourlyRate * hoursPerDay
  let weeklyEarnings := dailyEarnings * daysPerWeek
  let monthlyEarnings := weeklyEarnings * 4
  monthlyEarnings - (dailyEarnings * missedDays)

/-- Proves that given the specific conditions, the monthly salary is $920 -/
theorem julies_salary : 
  monthlySalary 5 8 6 1 = 920 := by
  sorry

#eval monthlySalary 5 8 6 1

end NUMINAMATH_CALUDE_julies_salary_l3458_345883


namespace NUMINAMATH_CALUDE_square_side_length_from_rectangle_l3458_345813

theorem square_side_length_from_rectangle (width height : ℝ) (h1 : width = 10) (h2 : height = 20) :
  ∃ y : ℝ, y^2 = width * height ∧ y = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_from_rectangle_l3458_345813


namespace NUMINAMATH_CALUDE_range_of_m_l3458_345835

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3458_345835


namespace NUMINAMATH_CALUDE_binary_147_ones_zeros_difference_l3458_345831

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_ones (l : List Bool) : ℕ :=
  sorry

def count_zeros (l : List Bool) : ℕ :=
  sorry

theorem binary_147_ones_zeros_difference :
  let bin_147 := binary_representation 147
  let ones := count_ones bin_147
  let zeros := count_zeros bin_147
  ones - zeros = 0 := by sorry

end NUMINAMATH_CALUDE_binary_147_ones_zeros_difference_l3458_345831


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l3458_345825

/-- The number of ways to choose 2 items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem rectangle_formation_count : 
  let horizontal_lines : ℕ := 5
  let vertical_lines : ℕ := 6
  (choose horizontal_lines 2) * (choose vertical_lines 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l3458_345825


namespace NUMINAMATH_CALUDE_ratio_unchanged_l3458_345807

theorem ratio_unchanged (a b : ℝ) (h : b ≠ 0) :
  (3 * a) / (b / (1 / 3)) = a / b :=
by sorry

end NUMINAMATH_CALUDE_ratio_unchanged_l3458_345807


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l3458_345827

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def contains_distinct_digits (n : ℕ) : Prop :=
  ∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ contains_digit n d₁ ∧ contains_digit n d₂

theorem smallest_n_with_properties : 
  (∀ m : ℕ, m < 128 → ¬(is_terminating_decimal m ∧ contains_digit m 9 ∧ contains_distinct_digits m)) ∧
  (is_terminating_decimal 128 ∧ contains_digit 128 9 ∧ contains_distinct_digits 128) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l3458_345827


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3458_345829

/-- An isosceles triangle with two sides of length 8 cm and perimeter 26 cm has a base of length 10 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 26 →
  base = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3458_345829


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l3458_345830

theorem car_rental_cost_per_mile 
  (base_cost : ℝ) 
  (total_miles : ℝ) 
  (total_cost : ℝ) 
  (h1 : base_cost = 150)
  (h2 : total_miles = 1364)
  (h3 : total_cost = 832) :
  (total_cost - base_cost) / total_miles = 0.50 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l3458_345830


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3458_345823

/-- A plane equation represented by integers A, B, C, and D -/
structure PlaneEquation where
  A : Int
  B : Int
  C : Int
  D : Int
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- Check if a point (x, y, z) lies on a plane -/
def lies_on_plane (p : PlaneEquation) (x y z : ℝ) : Prop :=
  p.A * x + p.B * y + p.C * z + p.D = 0

/-- Check if two planes are perpendicular -/
def perpendicular_planes (p1 p2 : PlaneEquation) : Prop :=
  p1.A * p2.A + p1.B * p2.B + p1.C * p2.C = 0

theorem plane_equation_correct (p : PlaneEquation) 
  (h1 : p.A = 2 ∧ p.B = -2 ∧ p.C = 1 ∧ p.D = 1) 
  (h2 : lies_on_plane p 0 2 3) 
  (h3 : lies_on_plane p 2 0 3) 
  (h4 : perpendicular_planes p { A := 1, B := -1, C := 4, D := -7, A_pos := by norm_num, gcd_one := by norm_num }) : 
  p.A = 2 ∧ p.B = -2 ∧ p.C = 1 ∧ p.D = 1 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3458_345823


namespace NUMINAMATH_CALUDE_rectangle_intersection_theorem_l3458_345803

/-- Represents a rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- Represents the configuration of three rectangles in a square -/
structure Configuration where
  square_side : ℝ
  rect1 : Rectangle
  rect2 : Rectangle
  rect3 : Rectangle

/-- The theorem to be proved -/
theorem rectangle_intersection_theorem (config : Configuration) 
  (h1 : config.square_side = 4)
  (h2 : config.rect1.area = 6)
  (h3 : config.rect2.area = 6)
  (h4 : config.rect3.area = 6) :
  ∃ (inter_area : ℝ), inter_area ≥ 2/3 ∧ 
  ((inter_area = (config.rect1.area + config.rect2.area - (config.square_side^2 - config.rect3.area)) / 2 ∨
    inter_area = (config.rect2.area + config.rect3.area - (config.square_side^2 - config.rect1.area)) / 2 ∨
    inter_area = (config.rect3.area + config.rect1.area - (config.square_side^2 - config.rect2.area)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_intersection_theorem_l3458_345803


namespace NUMINAMATH_CALUDE_half_reporters_not_cover_politics_l3458_345819

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 35

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_political_coverage : ℝ := 30

/-- Theorem stating that 50% of reporters do not cover politics -/
theorem half_reporters_not_cover_politics : 
  local_politics_coverage = 35 ∧ 
  non_local_political_coverage = 30 → 
  (100 : ℝ) - (local_politics_coverage / ((100 : ℝ) - non_local_political_coverage) * 100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_half_reporters_not_cover_politics_l3458_345819


namespace NUMINAMATH_CALUDE_min_m_value_l3458_345859

/-- The minimum value of m that satisfies the given conditions -/
theorem min_m_value (m : ℝ) (h_m : m > 0) : 
  (∀ x₁ x₂ : ℝ, 
    let y₁ := Real.exp x₁
    let y₂ := 1 + Real.log (x₂ - m)
    y₁ = y₂ → |x₂ - x₁| ≥ Real.exp 1) → 
  m ≥ Real.exp 1 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l3458_345859


namespace NUMINAMATH_CALUDE_min_markers_to_sell_is_1200_l3458_345843

/-- Represents the number of markers bought -/
def markers_bought : ℕ := 2000

/-- Represents the cost price of each marker in cents -/
def cost_price : ℕ := 20

/-- Represents the selling price of each marker in cents -/
def selling_price : ℕ := 50

/-- Represents the minimum profit desired in cents -/
def min_profit : ℕ := 20000

/-- Calculates the minimum number of markers that must be sold to achieve the desired profit -/
def min_markers_to_sell : ℕ :=
  (markers_bought * cost_price + min_profit) / (selling_price - cost_price)

/-- Theorem stating that the minimum number of markers to sell is 1200 -/
theorem min_markers_to_sell_is_1200 : min_markers_to_sell = 1200 := by
  sorry

#eval min_markers_to_sell

end NUMINAMATH_CALUDE_min_markers_to_sell_is_1200_l3458_345843


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l3458_345847

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

theorem symmetric_points_difference (a b : ℝ) :
  symmetric_wrt_origin a 1 5 b → a - b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l3458_345847


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3458_345806

/-- The equation p^2 - 6pq + q^2 + 3q - 1 = 0 has only one solution in prime numbers. -/
theorem unique_prime_solution :
  ∃! (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3458_345806


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l3458_345871

/-- Given two parabolas that intersect the coordinate axes at four points forming a rectangle with area 36, prove that the sum of their coefficients is 4/27 -/
theorem parabola_intersection_sum (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x^2 + 3 ∧ (x = 0 ∨ y = 0)) ∧ 
  (∃ x y : ℝ, y = 7 - b * x^2 ∧ (x = 0 ∨ y = 0)) ∧
  (∃ x1 x2 y1 y2 : ℝ, 
    (x1 ≠ 0 ∧ y1 = 0 ∧ y1 = a * x1^2 + 3) ∧
    (x2 ≠ 0 ∧ y2 = 0 ∧ y2 = 7 - b * x2^2) ∧
    (x1 * y2 = 36)) →
  a + b = 4/27 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l3458_345871


namespace NUMINAMATH_CALUDE_product_of_solutions_l3458_345828

theorem product_of_solutions (x : ℝ) : 
  (3 * x^2 + 5 * x - 40 = 0) → 
  (∃ y : ℝ, 3 * y^2 + 5 * y - 40 = 0 ∧ x * y = -40/3) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3458_345828


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3458_345854

/-- A geometric sequence with a_1 = 1 and a_3 = 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 3 = 2 ∧ ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 5 + a 10) / (a 1 + a 6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3458_345854


namespace NUMINAMATH_CALUDE_olivia_wallet_remaining_l3458_345805

/-- The amount of money remaining in Olivia's wallet after shopping -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that Olivia has 29 dollars left in her wallet -/
theorem olivia_wallet_remaining : remaining_money 54 25 = 29 := by
  sorry

end NUMINAMATH_CALUDE_olivia_wallet_remaining_l3458_345805


namespace NUMINAMATH_CALUDE_remainder_theorem_l3458_345844

theorem remainder_theorem (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3458_345844


namespace NUMINAMATH_CALUDE_cloth_sale_total_price_l3458_345841

-- Define the parameters of the problem
def quantity : ℕ := 80
def profit_per_meter : ℕ := 7
def cost_price_per_meter : ℕ := 118

-- Define the theorem
theorem cloth_sale_total_price :
  (quantity * (cost_price_per_meter + profit_per_meter)) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_total_price_l3458_345841


namespace NUMINAMATH_CALUDE_road_trip_driving_hours_l3458_345865

/-- Proves that in a 3-day road trip where one person drives 6 hours each day
    and the total driving time is 42 hours, the other person drives 8 hours each day. -/
theorem road_trip_driving_hours (total_days : ℕ) (krista_hours_per_day : ℕ) (total_hours : ℕ)
    (h1 : total_days = 3)
    (h2 : krista_hours_per_day = 6)
    (h3 : total_hours = 42) :
    (total_hours - krista_hours_per_day * total_days) / total_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_driving_hours_l3458_345865


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_seven_sixteenths_l3458_345842

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor where
  pattern_size : Nat
  corner_symmetry : Bool
  dark_tiles_in_quadrant : Nat

/-- Calculates the fraction of dark tiles on the floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  sorry

/-- Theorem stating that for a floor with the given properties, 
    the fraction of dark tiles is 7/16 -/
theorem dark_tile_fraction_is_seven_sixteenths 
  (floor : TiledFloor) 
  (h1 : floor.pattern_size = 8) 
  (h2 : floor.corner_symmetry = true) 
  (h3 : floor.dark_tiles_in_quadrant = 7) : 
  dark_tile_fraction floor = 7 / 16 :=
sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_seven_sixteenths_l3458_345842


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l3458_345887

/-- A function that returns the digits of an integer -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if each digit is twice the previous one -/
def doubling_digits (l : List ℕ) : Prop := sorry

/-- A function that calculates the sum of squares of a list of digits -/
def sum_of_squares (l : List ℕ) : ℕ := sorry

/-- A function that calculates the product of a list of digits -/
def product_of_digits (l : List ℕ) : ℕ := sorry

theorem largest_integer_with_conditions (n : ℕ) :
  (∀ m : ℕ, m > n → ¬(sum_of_squares (digits m) = 65 ∧ doubling_digits (digits m))) →
  sum_of_squares (digits n) = 65 →
  doubling_digits (digits n) →
  product_of_digits (digits n) = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l3458_345887


namespace NUMINAMATH_CALUDE_total_sales_given_april_l3458_345882

/-- Bennett's window screen sales pattern --/
structure BennettSales where
  january : ℕ
  february : ℕ := 2 * january
  march : ℕ := (january + february) / 2
  april : ℕ := min (2 * march) 20000

/-- Theorem: Total sales given April sales of 18000 --/
theorem total_sales_given_april (sales : BennettSales) 
  (h_april : sales.april = 18000) : 
  sales.january + sales.february + sales.march + sales.april = 45000 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_given_april_l3458_345882


namespace NUMINAMATH_CALUDE_distribution_theorem_l3458_345833

/-- The number of ways to distribute 4 men and 5 women into three groups of three people each,
    with at least one man and one woman in each group. -/
def distribution_ways : ℕ := 360

/-- The number of men -/
def num_men : ℕ := 4

/-- The number of women -/
def num_women : ℕ := 5

/-- The size of each group -/
def group_size : ℕ := 3

/-- The total number of groups -/
def num_groups : ℕ := 3

theorem distribution_theorem :
  (∀ (group : Fin num_groups), ∃ (m w : ℕ), m ≥ 1 ∧ w ≥ 1 ∧ m + w = group_size) →
  (num_men + num_women = num_groups * group_size) →
  distribution_ways = 360 := by
  sorry

end NUMINAMATH_CALUDE_distribution_theorem_l3458_345833


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l3458_345811

theorem repeating_decimal_difference : 
  (4 : ℚ) / 11 - (7 : ℚ) / 20 = (3 : ℚ) / 220 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l3458_345811


namespace NUMINAMATH_CALUDE_perpendicular_implies_intersects_parallel_perpendicular_transitive_perpendicular_implies_parallel_l3458_345853

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)

-- Statement 1
theorem perpendicular_implies_intersects (l : Line) (a : Plane) :
  perpendicular l a → intersects l a :=
sorry

-- Statement 3
theorem parallel_perpendicular_transitive (l m n : Line) (a : Plane) :
  parallel l m → parallel m n → perpendicular l a → perpendicular n a :=
sorry

-- Statement 4
theorem perpendicular_implies_parallel (l m n : Line) (a : Plane) :
  parallel l m → perpendicular m a → perpendicular n a → parallel l n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_intersects_parallel_perpendicular_transitive_perpendicular_implies_parallel_l3458_345853


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3458_345812

/-- Given an annual interest rate and time period, calculates the compound interest
    if the simple interest is known. -/
theorem compound_interest_calculation
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Annual interest rate (as a percentage)
  (T : ℝ) -- Time period in years
  (h1 : R = 20)
  (h2 : T = 2)
  (h3 : P * R * T / 100 = 400) -- Simple interest formula
  : P * (1 + R/100)^T - P = 440 := by
  sorry

#check compound_interest_calculation

end NUMINAMATH_CALUDE_compound_interest_calculation_l3458_345812


namespace NUMINAMATH_CALUDE_brownies_count_l3458_345874

/-- Given a box that can hold 7 brownies and 49 full boxes of brownies,
    prove that the total number of brownies is 343. -/
theorem brownies_count (brownies_per_box : ℕ) (full_boxes : ℕ) 
  (h1 : brownies_per_box = 7)
  (h2 : full_boxes = 49) : 
  brownies_per_box * full_boxes = 343 := by
  sorry

end NUMINAMATH_CALUDE_brownies_count_l3458_345874


namespace NUMINAMATH_CALUDE_unique_five_digit_divisible_by_72_l3458_345834

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

theorem unique_five_digit_divisible_by_72 :
  ∀ (a b : ℕ), a < 10 → b < 10 →
    (is_divisible_by (a * 10000 + 6790 + b) 72 ↔ a = 3 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_divisible_by_72_l3458_345834


namespace NUMINAMATH_CALUDE_parabola_properties_l3458_345861

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 4

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through (1, 2)
  parabola 1 2 ∧
  -- If A and B are intersection points of the line and parabola
  ∀ (A B : ℝ × ℝ), 
    (parabola A.1 A.2 ∧ line A.1 A.2) →
    (parabola B.1 B.2 ∧ line B.1 B.2) →
    A ≠ B →
    -- Then OA is perpendicular to OB
    (A.1 * B.1 + A.2 * B.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3458_345861


namespace NUMINAMATH_CALUDE_linear_regression_transformation_l3458_345857

-- Define the variables and functions
variable (a b x : ℝ)
variable (y : ℝ)
variable (μ : ℝ)
variable (c : ℝ)
variable (v : ℝ)

-- Define the conditions
def condition_y : Prop := y = a * Real.exp (b / x)
def condition_μ : Prop := μ = Real.log y
def condition_c : Prop := c = Real.log a
def condition_v : Prop := v = 1 / x

-- State the theorem
theorem linear_regression_transformation 
  (h1 : condition_y a b x y)
  (h2 : condition_μ y μ)
  (h3 : condition_c a c)
  (h4 : condition_v x v) :
  μ = c + b * v :=
by sorry

end NUMINAMATH_CALUDE_linear_regression_transformation_l3458_345857


namespace NUMINAMATH_CALUDE_inequality_for_real_numbers_l3458_345816

theorem inequality_for_real_numbers (a b : ℝ) : a * b ≤ ((a + b) / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_real_numbers_l3458_345816


namespace NUMINAMATH_CALUDE_job_selection_probability_l3458_345845

theorem job_selection_probability (carol_prob bernie_prob : ℚ) 
  (h_carol : carol_prob = 4/5)
  (h_bernie : bernie_prob = 3/5) : 
  carol_prob * bernie_prob = 12/25 := by
sorry

end NUMINAMATH_CALUDE_job_selection_probability_l3458_345845


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3458_345885

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 2 = 0 ∧ y^2 - 2*y + m - 2 = 0) → m < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3458_345885


namespace NUMINAMATH_CALUDE_complex_division_simplification_l3458_345879

theorem complex_division_simplification (z : ℂ) : 
  z = (4 + 3*I) / (1 + 2*I) → z = 2 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l3458_345879


namespace NUMINAMATH_CALUDE_T_equals_five_l3458_345808

noncomputable def T : ℝ :=
  1 / (3 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
  1 / (Real.sqrt 7 - Real.sqrt 6) - 1 / (Real.sqrt 6 - Real.sqrt 5) + 
  1 / (Real.sqrt 5 - 2)

theorem T_equals_five : T = 5 := by
  sorry

end NUMINAMATH_CALUDE_T_equals_five_l3458_345808


namespace NUMINAMATH_CALUDE_anna_weekly_salary_l3458_345855

/-- Represents a worker's salary information -/
structure WorkerSalary where
  daysWorkedPerWeek : ℕ
  missedDays : ℕ
  deductionAmount : ℚ

/-- Calculates the usual weekly salary of a worker -/
def usualWeeklySalary (w : WorkerSalary) : ℚ :=
  (w.deductionAmount / w.missedDays) * w.daysWorkedPerWeek

theorem anna_weekly_salary :
  let anna : WorkerSalary := {
    daysWorkedPerWeek := 5,
    missedDays := 2,
    deductionAmount := 985
  }
  usualWeeklySalary anna = 2462.5 := by
  sorry

end NUMINAMATH_CALUDE_anna_weekly_salary_l3458_345855


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3458_345809

def divisible_by_2_or_5_ends_with_0 (n : ℕ) : Prop :=
  (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0

def last_digit_not_0_not_divisible_by_2_and_5 (n : ℕ) : Prop :=
  n % 10 ≠ 0 → (n % 2 ≠ 0 ∧ n % 5 ≠ 0)

theorem contrapositive_equivalence :
  ∀ n : ℕ, divisible_by_2_or_5_ends_with_0 n ↔ last_digit_not_0_not_divisible_by_2_and_5 n :=
by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3458_345809


namespace NUMINAMATH_CALUDE_min_third_altitude_l3458_345858

/-- Represents a scalene triangle with specific altitude properties -/
structure ScaleneTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Altitudes
  h_D : ℝ
  h_E : ℝ
  h_F : ℝ
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Scalene property
  scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a
  -- Given altitude values
  altitude_D : h_D = 18
  altitude_E : h_E = 8
  -- Relation between sides
  side_relation : b = 2 * a
  -- Area consistency
  area_consistency : a * h_D / 2 = b * h_E / 2

/-- The minimum possible integer length of the third altitude is 17 -/
theorem min_third_altitude (t : ScaleneTriangle) : 
  ∃ (n : ℕ), n ≥ 17 ∧ t.h_F = n ∧ ∀ (m : ℕ), m < 17 → t.h_F ≠ m :=
sorry

end NUMINAMATH_CALUDE_min_third_altitude_l3458_345858


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l3458_345863

theorem negation_of_universal_positive_square_plus_one :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l3458_345863


namespace NUMINAMATH_CALUDE_cube_root_problem_l3458_345850

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l3458_345850


namespace NUMINAMATH_CALUDE_vector_projection_and_perpendicular_l3458_345889

/-- Given two vectors a and b in ℝ², and a scalar k, we define vector c and prove properties about their relationships. -/
theorem vector_projection_and_perpendicular (a b : ℝ × ℝ) (k : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (3, 1)
  let c : ℝ × ℝ := b - k • a
  (a.1 * c.1 + a.2 * c.2 = 0) →
  (let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2)
   proj = Real.sqrt 5 ∧ k = 1 ∧ c = (2, -1)) := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_and_perpendicular_l3458_345889


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_with_condition_l3458_345873

theorem no_infinite_prime_sequence_with_condition :
  ¬ ∃ (p : ℕ → ℕ), 
    (∀ n, Prime (p n)) ∧ 
    (∀ n, p n < p (n + 1)) ∧ 
    (∀ k, p (k + 1) = 2 * p k - 1 ∨ p (k + 1) = 2 * p k + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_with_condition_l3458_345873


namespace NUMINAMATH_CALUDE_abs_5e_minus_15_l3458_345801

-- Define e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem abs_5e_minus_15 : |5 * e - 15| = 1.4086 := by sorry

end NUMINAMATH_CALUDE_abs_5e_minus_15_l3458_345801


namespace NUMINAMATH_CALUDE_opponent_total_score_l3458_345868

def team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def games_lost_by_one (scores : List ℕ) : ℕ := 6

def score_ratio_in_other_games : ℕ := 3

theorem opponent_total_score :
  let opponent_scores := team_scores.map (λ score =>
    if score % 2 = 1 then score + 1
    else score / score_ratio_in_other_games)
  opponent_scores.sum = 60 := by sorry

end NUMINAMATH_CALUDE_opponent_total_score_l3458_345868


namespace NUMINAMATH_CALUDE_products_inspected_fraction_l3458_345866

/-- The fraction of products inspected by John, Jane, and Roy is 1 -/
theorem products_inspected_fraction (j n r : ℝ) : 
  j ≥ 0 → n ≥ 0 → r ≥ 0 →
  0.007 * j + 0.008 * n + 0.01 * r = 0.0085 →
  j + n + r = 1 :=
by sorry

end NUMINAMATH_CALUDE_products_inspected_fraction_l3458_345866


namespace NUMINAMATH_CALUDE_barbara_tuna_packs_l3458_345878

/-- The number of tuna packs Barbara bought -/
def tuna_packs : ℕ := sorry

/-- The price of each tuna pack in dollars -/
def tuna_price : ℚ := 2

/-- The number of water bottles Barbara bought -/
def water_bottles : ℕ := 4

/-- The price of each water bottle in dollars -/
def water_price : ℚ := (3 : ℚ) / 2

/-- The amount spent on different goods in dollars -/
def different_goods_cost : ℚ := 40

/-- The total amount Barbara paid in dollars -/
def total_paid : ℚ := 56

theorem barbara_tuna_packs : 
  tuna_packs = 5 ∧ 
  (tuna_packs : ℚ) * tuna_price + (water_bottles : ℚ) * water_price + different_goods_cost = total_paid :=
sorry

end NUMINAMATH_CALUDE_barbara_tuna_packs_l3458_345878


namespace NUMINAMATH_CALUDE_three_numbers_with_square_sums_l3458_345876

theorem three_numbers_with_square_sums : ∃ (a b c : ℕ+), 
  (∃ (x : ℕ), (a + b + c : ℕ) = x^2) ∧
  (∃ (y : ℕ), (a + b : ℕ) = y^2) ∧
  (∃ (z : ℕ), (b + c : ℕ) = z^2) ∧
  (∃ (w : ℕ), (a + c : ℕ) = w^2) ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_with_square_sums_l3458_345876
