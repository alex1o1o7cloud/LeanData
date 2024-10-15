import Mathlib

namespace NUMINAMATH_CALUDE_prism_18_edges_8_faces_l969_96926

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_18_edges_8_faces_l969_96926


namespace NUMINAMATH_CALUDE_problem_solution_l969_96976

theorem problem_solution (x z : ℝ) (h1 : x ≠ 0) (h2 : x/3 = z^2) (h3 : x/5 = 5*z) : x = 625/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l969_96976


namespace NUMINAMATH_CALUDE_volume_pyramid_section_l969_96915

/-- The volume of a section of a regular triangular pyramid -/
theorem volume_pyramid_section (H α β : Real) 
  (h_positive : H > 0)
  (h_angle_α : 0 < α ∧ α < π / 2)
  (h_angle_β : 0 < β ∧ β < π / 2 - α) :
  ∃ V : Real, V = (3 * Real.sqrt 3 * H^3 * Real.sin α * Real.tan α^2 * Real.cos (α - β)) / (8 * Real.sin β) :=
sorry

end NUMINAMATH_CALUDE_volume_pyramid_section_l969_96915


namespace NUMINAMATH_CALUDE_problem_2023_l969_96943

theorem problem_2023 : (2023^2 - 2023 + 1) / 2023 = 2022 + 1/2023 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_l969_96943


namespace NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l969_96968

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 4}
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Part 1
theorem intersection_and_union (a : ℝ) (h : a = 0) :
  (A a ∩ B = {x | 0 ≤ x ∧ x ≤ 3}) ∧
  (A a ∪ (Set.univ \ B) = {x | x < -2 ∨ x ≥ 0}) := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | A a ∪ B = B} = {a : ℝ | -2 ≤ a ∧ a ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l969_96968


namespace NUMINAMATH_CALUDE_trajectory_of_C_l969_96941

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ :=
  dist t.A t.B + dist t.B t.C + dist t.C t.A

-- Theorem statement
theorem trajectory_of_C (x y : ℝ) :
  let t := Triangle.mk (0, 2) (0, -2) (x, y)
  perimeter t = 10 ∧ x ≠ 0 →
  x^2 / 5 + y^2 / 9 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_C_l969_96941


namespace NUMINAMATH_CALUDE_mcgees_bakery_pies_l969_96971

theorem mcgees_bakery_pies (smiths_pies mcgees_pies : ℕ) : 
  smiths_pies = 70 → 
  smiths_pies = 4 * mcgees_pies + 6 → 
  mcgees_pies = 16 := by
sorry

end NUMINAMATH_CALUDE_mcgees_bakery_pies_l969_96971


namespace NUMINAMATH_CALUDE_letter_at_unknown_position_l969_96945

/-- Represents the letters that can be used in the grid -/
inductive Letter : Type
| A | B | C | D | E

/-- Represents a position in the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the 5x5 grid -/
def Grid := Position → Letter

/-- Check if each letter appears exactly once in each row -/
def valid_rows (g : Grid) : Prop :=
  ∀ r : Fin 5, ∀ l : Letter, ∃! c : Fin 5, g ⟨r, c⟩ = l

/-- Check if each letter appears exactly once in each column -/
def valid_columns (g : Grid) : Prop :=
  ∀ c : Fin 5, ∀ l : Letter, ∃! r : Fin 5, g ⟨r, c⟩ = l

/-- Check if each letter appears exactly once in the main diagonal -/
def valid_main_diagonal (g : Grid) : Prop :=
  ∀ l : Letter, ∃! i : Fin 5, g ⟨i, i⟩ = l

/-- Check if each letter appears exactly once in the anti-diagonal -/
def valid_anti_diagonal (g : Grid) : Prop :=
  ∀ l : Letter, ∃! i : Fin 5, g ⟨i, 4 - i⟩ = l

/-- Check if the grid satisfies all constraints -/
def valid_grid (g : Grid) : Prop :=
  valid_rows g ∧ valid_columns g ∧ valid_main_diagonal g ∧ valid_anti_diagonal g

/-- The theorem to prove -/
theorem letter_at_unknown_position (g : Grid) 
  (h_valid : valid_grid g)
  (h_A : g ⟨0, 0⟩ = Letter.A)
  (h_D : g ⟨3, 0⟩ = Letter.D)
  (h_E : g ⟨4, 0⟩ = Letter.E) :
  ∃ p : Position, g p = Letter.B :=
by sorry

end NUMINAMATH_CALUDE_letter_at_unknown_position_l969_96945


namespace NUMINAMATH_CALUDE_prove_b_equals_one_l969_96950

theorem prove_b_equals_one (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 49 * 45 * b) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_prove_b_equals_one_l969_96950


namespace NUMINAMATH_CALUDE_tangent_circles_t_value_l969_96962

-- Define the circles
def circle1 (t : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = t^2
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y + 24 = 0

-- Define tangency
def tangent (t : ℝ) : Prop := ∃ x y : ℝ, circle1 t x y ∧ circle2 x y

-- Theorem statement
theorem tangent_circles_t_value :
  ∀ t : ℝ, t > 0 → tangent t → t = 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_t_value_l969_96962


namespace NUMINAMATH_CALUDE_last_digit_2_power_2010_l969_96986

/-- The last digit of 2^n for n ≥ 1 -/
def lastDigitPowerOfTwo (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur

theorem last_digit_2_power_2010 : lastDigitPowerOfTwo 2010 = 4 := by
  sorry

#eval lastDigitPowerOfTwo 2010

end NUMINAMATH_CALUDE_last_digit_2_power_2010_l969_96986


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l969_96908

theorem number_subtraction_problem (x : ℝ) : 
  0.30 * x - 70 = 20 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l969_96908


namespace NUMINAMATH_CALUDE_inequality_solution_l969_96973

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 5/x + 21/10) ↔ -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l969_96973


namespace NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_fourth_l969_96977

theorem tan_two_alpha_plus_pi_fourth (α : ℝ) 
  (h : (2 * (Real.cos α)^2 + Real.cos (π/2 + 2*α) - 1) / (Real.sqrt 2 * Real.sin (2*α + π/4)) = 4) : 
  Real.tan (2*α + π/4) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_fourth_l969_96977


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l969_96967

-- Define the radical conjugate
def radical_conjugate (a b : ℝ) : ℝ := a + b

-- State the theorem
theorem sum_with_radical_conjugate :
  let x := 8 - Real.sqrt 1369
  x + radical_conjugate 8 (Real.sqrt 1369) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l969_96967


namespace NUMINAMATH_CALUDE_negative_of_negative_is_positive_l969_96919

theorem negative_of_negative_is_positive (y : ℝ) (h : y < 0) : -y > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_is_positive_l969_96919


namespace NUMINAMATH_CALUDE_price_reduction_effect_l969_96953

theorem price_reduction_effect (original_price : ℝ) (original_sales : ℝ) 
  (price_reduction_percent : ℝ) (net_effect_percent : ℝ) : 
  price_reduction_percent = 40 →
  net_effect_percent = 8 →
  ∃ (sales_increase_percent : ℝ),
    sales_increase_percent = 80 ∧
    (1 - price_reduction_percent / 100) * (1 + sales_increase_percent / 100) = 1 + net_effect_percent / 100 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_effect_l969_96953


namespace NUMINAMATH_CALUDE_final_value_of_A_l969_96924

-- Define the initial value of A
def A_initial : ℤ := 15

-- Define the operation as a function
def operation (x : ℤ) : ℤ := -x + 5

-- Theorem stating the final value of A after the operation
theorem final_value_of_A : operation A_initial = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l969_96924


namespace NUMINAMATH_CALUDE_four_liars_in_group_l969_96997

/-- Represents a person who is either a knight or a liar -/
inductive Person
  | Knight
  | Liar

/-- Represents an answer to the question "How many liars are among you?" -/
def Answer := Fin 5

/-- A function that determines whether a person is telling the truth given their answer and the actual number of liars -/
def isTellingTruth (p : Person) (answer : Answer) (actualLiars : Nat) : Prop :=
  match p with
  | Person.Knight => answer.val + 1 = actualLiars
  | Person.Liar => answer.val + 1 ≠ actualLiars

/-- The main theorem -/
theorem four_liars_in_group (group : Fin 5 → Person) (answers : Fin 5 → Answer) 
    (h_distinct : ∀ i j, i ≠ j → answers i ≠ answers j) :
    (∃ (actualLiars : Nat), actualLiars = 4 ∧ 
      ∀ i, isTellingTruth (group i) (answers i) actualLiars) := by
  sorry

end NUMINAMATH_CALUDE_four_liars_in_group_l969_96997


namespace NUMINAMATH_CALUDE_square_of_103_product_of_998_and_1002_l969_96951

-- Problem 1
theorem square_of_103 : 103^2 = 10609 := by sorry

-- Problem 2
theorem product_of_998_and_1002 : 998 * 1002 = 999996 := by sorry

end NUMINAMATH_CALUDE_square_of_103_product_of_998_and_1002_l969_96951


namespace NUMINAMATH_CALUDE_solution_x_is_three_fourths_l969_96949

-- Define the * operation
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- State the theorem
theorem solution_x_is_three_fourths :
  ∃ x : ℝ, star 7 (star 3 (x - 1)) = 3 ∧ x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_is_three_fourths_l969_96949


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l969_96929

/-- The area of an isosceles trapezoid with given dimensions -/
theorem isosceles_trapezoid_area (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let leg := a
  let base1 := b
  let base2 := c
  let height := Real.sqrt (a^2 - ((c - b)/2)^2)
  (base1 + base2) * height / 2 = 36 ↔ a = 5 ∧ b = 6 ∧ c = 12 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l969_96929


namespace NUMINAMATH_CALUDE_perpendicular_condition_l969_96907

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: 2x - y - 1 = 0 -/
def line1 : Line := { a := 2, b := -1, c := -1 }

/-- The second line: mx + y + 1 = 0 -/
def line2 (m : ℝ) : Line := { a := m, b := 1, c := 1 }

/-- Theorem: The necessary and sufficient condition for perpendicularity -/
theorem perpendicular_condition :
  ∀ m : ℝ, perpendicular line1 (line2 m) ↔ m = 1/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l969_96907


namespace NUMINAMATH_CALUDE_haley_car_distance_l969_96980

/-- Calculates the distance covered given a fuel-to-distance ratio and fuel consumption -/
def distance_covered (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) : ℕ :=
  (fuel_used / fuel_ratio) * distance_ratio

/-- Theorem stating that for a 4:7 fuel-to-distance ratio and 44 gallons of fuel, the distance covered is 77 miles -/
theorem haley_car_distance :
  distance_covered 4 7 44 = 77 := by
  sorry

end NUMINAMATH_CALUDE_haley_car_distance_l969_96980


namespace NUMINAMATH_CALUDE_ball_path_length_l969_96918

theorem ball_path_length (A B C M : ℝ × ℝ) : 
  -- Triangle ABC is a right triangle with ∠ABC = 90° and ∠BAC = 60°
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = (B.1 - A.1)^2 + (B.2 - A.2)^2 →
  -- AB = 6
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →
  -- M is the midpoint of BC
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  -- The length of the path from M to AB to AC and back to M is 3√21
  ∃ (P Q : ℝ × ℝ), 
    P.1 = B.1 ∧ 
    (Q.1 - A.1) * (C.1 - A.1) + (Q.2 - A.2) * (C.2 - A.2) = 0 ∧
    (P.1 - M.1)^2 + (P.2 - M.2)^2 + 
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 + 
    (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 189 := by
  sorry

end NUMINAMATH_CALUDE_ball_path_length_l969_96918


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l969_96961

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l969_96961


namespace NUMINAMATH_CALUDE_trees_in_yard_l969_96910

/-- The number of trees in a yard with given conditions -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating the number of trees in the yard under given conditions -/
theorem trees_in_yard :
  let yard_length : ℕ := 150
  let tree_distance : ℕ := 15
  number_of_trees yard_length tree_distance = 11 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l969_96910


namespace NUMINAMATH_CALUDE_mom_talia_age_ratio_l969_96979

-- Define Talia's current age
def talia_current_age : ℕ := 20 - 7

-- Define Talia's father's current age
def father_current_age : ℕ := 36

-- Define Talia's mother's current age
def mother_current_age : ℕ := father_current_age + 3

-- Theorem stating the ratio of Talia's mom's age to Talia's age
theorem mom_talia_age_ratio :
  mother_current_age / talia_current_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_mom_talia_age_ratio_l969_96979


namespace NUMINAMATH_CALUDE_cd_product_value_l969_96983

/-- An equilateral triangle with vertices at (0,0), (c,17), and (d,43) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ

/-- The product of c and d in the equilateral triangle -/
def cd_product (triangle : EquilateralTriangle) : ℝ := triangle.c * triangle.d

/-- Theorem stating that the product cd equals -1689/24 for the given equilateral triangle -/
theorem cd_product_value (triangle : EquilateralTriangle) :
  cd_product triangle = -1689 / 24 := by sorry

end NUMINAMATH_CALUDE_cd_product_value_l969_96983


namespace NUMINAMATH_CALUDE_challenge_result_l969_96995

theorem challenge_result (x : ℕ) : 3 * (3 * (x + 1) + 3) = 63 := by
  sorry

#check challenge_result

end NUMINAMATH_CALUDE_challenge_result_l969_96995


namespace NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l969_96930

/-- The Tuning Day Method function -/
def tuningDayMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

/-- Check if a fraction is simpler than another -/
def isSimpler (a b c d : ℕ) : Bool :=
  a + b < c + d ∨ (a + b = c + d ∧ a < c)

theorem tuning_day_method_pi_approximation :
  let initial_lower : ℚ := 31 / 10
  let initial_upper : ℚ := 49 / 15
  let step1 : ℚ := tuningDayMethod 10 31 15 49
  let step2 : ℚ := tuningDayMethod 10 31 5 16
  let step3 : ℚ := tuningDayMethod 15 47 5 16
  let step4 : ℚ := tuningDayMethod 15 47 20 63
  initial_lower < Real.pi ∧ Real.pi < initial_upper ∧
  step1 = 16 / 5 ∧
  step2 = 47 / 15 ∧
  step3 = 63 / 20 ∧
  step4 = 22 / 7 ∧
  isSimpler 22 7 63 20 ∧
  isSimpler 22 7 47 15 ∧
  isSimpler 22 7 16 5 ∧
  47 / 15 < Real.pi ∧ Real.pi < 22 / 7 :=
by sorry

end NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l969_96930


namespace NUMINAMATH_CALUDE_triangle_tangent_circles_l969_96937

/-- Given a triangle with side lengths a, b, and c, there exist radii r₁, r₂, and r₃ for circles
    centered at the triangle's vertices that satisfy both external and internal tangency conditions. -/
theorem triangle_tangent_circles
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  ∃ (r₁ r₂ r₃ : ℝ),
    (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) ∧
    (r₁ + r₂ = c ∧ r₂ + r₃ = a ∧ r₃ + r₁ = b) ∧
    ∃ (r₁' r₂' r₃' : ℝ),
      (r₁' > 0 ∧ r₂' > 0 ∧ r₃' > 0) ∧
      (r₃' - r₂' = a ∧ r₃' - r₁' = b ∧ r₁' + r₂' = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_circles_l969_96937


namespace NUMINAMATH_CALUDE_raisin_cookies_sold_l969_96970

theorem raisin_cookies_sold (raisin oatmeal : ℕ) : 
  (raisin : ℚ) / oatmeal = 6 / 1 →
  raisin + oatmeal = 49 →
  raisin = 42 := by
sorry

end NUMINAMATH_CALUDE_raisin_cookies_sold_l969_96970


namespace NUMINAMATH_CALUDE_article_price_l969_96936

theorem article_price (P : ℝ) : 
  P * (1 - 0.1) * (1 - 0.2) = 72 → P = 100 := by
  sorry

end NUMINAMATH_CALUDE_article_price_l969_96936


namespace NUMINAMATH_CALUDE_range_of_sin_plus_cos_l969_96934

theorem range_of_sin_plus_cos :
  ∀ y : ℝ, (∃ x : ℝ, y = Real.sin x + Real.cos x) ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sin_plus_cos_l969_96934


namespace NUMINAMATH_CALUDE_fibonacci_identity_l969_96916

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_identity (n : ℕ) (h : n ≥ 1) :
  (fib (2 * n - 1))^2 + (fib (2 * n + 1))^2 + 1 = 3 * (fib (2 * n - 1)) * (fib (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_identity_l969_96916


namespace NUMINAMATH_CALUDE_bike_shop_profit_is_8206_l969_96942

/-- Represents the profit calculation for Jim's bike shop -/
def bike_shop_profit (tire_repair_price tire_repair_cost tire_repairs_count
                      chain_repair_price chain_repair_cost chain_repairs_count
                      overhaul_price overhaul_cost overhaul_count
                      retail_sales retail_cost
                      parts_discount_threshold parts_discount_rate
                      tax_rate fixed_expenses : ℚ) : ℚ :=
  let total_income := tire_repair_price * tire_repairs_count +
                      chain_repair_price * chain_repairs_count +
                      overhaul_price * overhaul_count +
                      retail_sales

  let total_parts_cost := tire_repair_cost * tire_repairs_count +
                          chain_repair_cost * chain_repairs_count +
                          overhaul_cost * overhaul_count

  let parts_discount := if total_parts_cost ≥ parts_discount_threshold
                        then total_parts_cost * parts_discount_rate
                        else 0

  let final_parts_cost := total_parts_cost - parts_discount

  let profit_before_tax := total_income - final_parts_cost - retail_cost

  let taxes := total_income * tax_rate

  profit_before_tax - taxes - fixed_expenses

/-- Theorem stating that the bike shop's profit is $8206 given the specified conditions -/
theorem bike_shop_profit_is_8206 :
  bike_shop_profit 20 5 300
                   75 25 50
                   300 50 8
                   2000 1200
                   2500 0.1
                   0.06 4000 = 8206 := by sorry

end NUMINAMATH_CALUDE_bike_shop_profit_is_8206_l969_96942


namespace NUMINAMATH_CALUDE_smallest_y_value_l969_96966

theorem smallest_y_value (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 5) → (y ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l969_96966


namespace NUMINAMATH_CALUDE_contest_probability_l969_96999

/-- The probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- The number of preset questions -/
def n : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := 2 * p^3 * (1 - p)

theorem contest_probability :
  prob_four_questions = 0.128 :=
sorry

end NUMINAMATH_CALUDE_contest_probability_l969_96999


namespace NUMINAMATH_CALUDE_only_vegetarian_count_l969_96920

/-- Represents the number of people in different dietary categories in a family --/
structure FamilyDiet where
  only_nonveg : ℕ
  both_veg_and_nonveg : ℕ
  total_veg : ℕ

/-- Theorem stating the number of people who eat only vegetarian --/
theorem only_vegetarian_count (f : FamilyDiet) 
  (h1 : f.only_nonveg = 8)
  (h2 : f.both_veg_and_nonveg = 6)
  (h3 : f.total_veg = 19) :
  f.total_veg - f.both_veg_and_nonveg = 13 := by
  sorry

#check only_vegetarian_count

end NUMINAMATH_CALUDE_only_vegetarian_count_l969_96920


namespace NUMINAMATH_CALUDE_complex_number_problem_l969_96947

theorem complex_number_problem (m : ℝ) (z : ℂ) :
  let z₁ : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z₁.re = 0 ∧ z₁.im ≠ 0) →
  (3 + z₁) * z = 4 + 2 * Complex.I →
  m = 0 ∧ z = 1 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l969_96947


namespace NUMINAMATH_CALUDE_variance_and_shifted_average_l969_96948

theorem variance_and_shifted_average
  (x₁ x₂ x₃ x₄ : ℝ)
  (pos_x : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0)
  (variance : (1/4) * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) = (1/4) * ((x₁ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₂ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₃ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₄ - (x₁ + x₂ + x₃ + x₄)/4)^2)) :
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_variance_and_shifted_average_l969_96948


namespace NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_2_l969_96981

def z : ℂ := Complex.I - 2 * Complex.I^2 + 3 * Complex.I^3

theorem abs_z_equals_2_sqrt_2 : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_2_l969_96981


namespace NUMINAMATH_CALUDE_problem_statement_l969_96921

theorem problem_statement (a : ℝ) (h : 2 * a - 1 / a = 3) : 16 * a^4 + 1 / a^4 = 161 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l969_96921


namespace NUMINAMATH_CALUDE_unripe_oranges_per_day_is_65_l969_96939

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges after the harvest period -/
def total_unripe_oranges : ℕ := 390

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := total_unripe_oranges / harvest_days

/-- Theorem stating that the number of sacks of unripe oranges harvested per day is 65 -/
theorem unripe_oranges_per_day_is_65 : unripe_oranges_per_day = 65 := by
  sorry

end NUMINAMATH_CALUDE_unripe_oranges_per_day_is_65_l969_96939


namespace NUMINAMATH_CALUDE_quadratic_one_root_l969_96911

/-- A quadratic function with coefficients a, b, and c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The discriminant of a quadratic function -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_one_root (k : ℝ) : 
  (∃! x, QuadraticFunction 1 (-2) k x = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l969_96911


namespace NUMINAMATH_CALUDE_embankment_completion_time_l969_96923

/-- The time required for a group of workers to complete an embankment -/
def embankment_time (workers : ℕ) (portion : ℚ) (days : ℚ) : Prop :=
  ∃ (rate : ℚ), rate > 0 ∧ portion = (workers : ℚ) * rate * days

theorem embankment_completion_time :
  embankment_time 60 (1/2) 5 →
  embankment_time 80 1 (15/2) :=
by sorry

end NUMINAMATH_CALUDE_embankment_completion_time_l969_96923


namespace NUMINAMATH_CALUDE_tangent_parallel_to_4x_l969_96984

/-- The curve function f(x) = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_to_4x :
  ∃ x : ℝ, f x = 0 ∧ f' x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_4x_l969_96984


namespace NUMINAMATH_CALUDE_total_seashells_l969_96932

theorem total_seashells (red_shells green_shells other_shells : ℕ) 
  (h1 : red_shells = 76)
  (h2 : green_shells = 49)
  (h3 : other_shells = 166) :
  red_shells + green_shells + other_shells = 291 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l969_96932


namespace NUMINAMATH_CALUDE_prob_b_greater_a_value_l969_96922

/-- The number of possible choices for each person -/
def n : ℕ := 1000

/-- The probability of B picking a number greater than A -/
def prob_b_greater_a : ℚ :=
  (n * (n - 1) / 2) / (n * n)

/-- Theorem: The probability of B picking a number greater than A is 499500/1000000 -/
theorem prob_b_greater_a_value : prob_b_greater_a = 499500 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_prob_b_greater_a_value_l969_96922


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_regular_hexagon_interior_angle_is_120_l969_96938

/-- The measure of one interior angle of a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle : ℝ :=
  let n : ℕ := 6  -- number of sides in a hexagon
  let sum_interior_angles : ℝ := 180 * (n - 2)
  let num_angles : ℕ := n
  sum_interior_angles / num_angles

/-- The result of regular_hexagon_interior_angle is equal to 120. -/
theorem regular_hexagon_interior_angle_is_120 : 
  regular_hexagon_interior_angle = 120 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_regular_hexagon_interior_angle_is_120_l969_96938


namespace NUMINAMATH_CALUDE_right_triangle_sets_l969_96994

theorem right_triangle_sets :
  let set1 : Fin 3 → ℝ := ![3, 4, 5]
  let set2 : Fin 3 → ℝ := ![9, 12, 15]
  let set3 : Fin 3 → ℝ := ![Real.sqrt 3, 2, Real.sqrt 5]
  let set4 : Fin 3 → ℝ := ![0.3, 0.4, 0.5]

  (set1 0)^2 + (set1 1)^2 = (set1 2)^2 ∧
  (set2 0)^2 + (set2 1)^2 = (set2 2)^2 ∧
  (set3 0)^2 + (set3 1)^2 ≠ (set3 2)^2 ∧
  (set4 0)^2 + (set4 1)^2 = (set4 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l969_96994


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l969_96904

theorem fractional_equation_solution :
  ∀ x : ℚ, x ≠ 1 → 3*x - 3 ≠ 0 →
  (2*x / (x - 1) = x / (3*x - 3) + 1) ↔ (x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l969_96904


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_0_and_1_l969_96996

theorem smallest_value_for_x_between_0_and_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 ≤ x ∧ x^3 ≤ x^2 ∧ x^3 ≤ x^3 ∧ x^3 ≤ Real.sqrt x ∧ x^3 ≤ 2*x ∧ x^3 ≤ 1/x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_0_and_1_l969_96996


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l969_96946

/-- The polynomial division theorem for z^2023 - 1 divided by z^2 + z + 1 -/
theorem polynomial_division_remainder (z : ℂ) : ∃ (Q R : ℂ → ℂ),
  z^2023 - 1 = (z^2 + z + 1) * Q z + R z ∧ 
  (∀ x, R x = -x - 1) ∧
  (∃ a b, ∀ x, R x = a * x + b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l969_96946


namespace NUMINAMATH_CALUDE_percentage_problem_l969_96985

theorem percentage_problem (P : ℝ) : 
  P * 140 = (4/5) * 140 - 21 → P = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l969_96985


namespace NUMINAMATH_CALUDE_rounding_estimate_l969_96944

theorem rounding_estimate (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (a' : ℕ) (ha' : a' ≥ a)
  (b' : ℕ) (hb' : b' ≤ b)
  (c' : ℕ) (hc' : c' ≥ c)
  (d' : ℕ) (hd' : d' ≥ d) :
  (a' * d' : ℚ) / b' + c' > (a * d : ℚ) / b + c :=
sorry

end NUMINAMATH_CALUDE_rounding_estimate_l969_96944


namespace NUMINAMATH_CALUDE_angle_measure_proof_l969_96902

theorem angle_measure_proof (x : ℝ) : 
  (x = 21) → 
  (90 - x = 3 * x + 6) ∧
  (x + (90 - x) = 90) :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l969_96902


namespace NUMINAMATH_CALUDE_rectangles_and_triangles_on_4x3_grid_l969_96998

/-- The number of rectangles on an m × n grid -/
def count_rectangles (m n : ℕ) : ℕ := (m.choose 2) * (n.choose 2)

/-- The number of right-angled triangles (with right angles at grid points) on an m × n grid -/
def count_right_triangles (m n : ℕ) : ℕ := 2 * (m - 1) * (n - 1)

/-- The total number of rectangles and right-angled triangles on a 4×3 grid is 30 -/
theorem rectangles_and_triangles_on_4x3_grid :
  count_rectangles 4 3 + count_right_triangles 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_and_triangles_on_4x3_grid_l969_96998


namespace NUMINAMATH_CALUDE_maria_cookie_baggies_l969_96928

/-- The number of baggies Maria can make with her cookies -/
def num_baggies (cookies_per_baggie : ℕ) (chocolate_chip_cookies : ℕ) (oatmeal_cookies : ℕ) : ℕ :=
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_baggie

/-- Theorem stating that Maria can make 7 baggies of cookies -/
theorem maria_cookie_baggies :
  num_baggies 5 33 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookie_baggies_l969_96928


namespace NUMINAMATH_CALUDE_sum_product_remainder_l969_96900

theorem sum_product_remainder : (1789 * 1861 * 1945 + 1533 * 1607 * 1688) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_remainder_l969_96900


namespace NUMINAMATH_CALUDE_polygon_diagonals_l969_96975

theorem polygon_diagonals (n : ℕ) (h : n = 150) : (n * (n - 3)) / 2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l969_96975


namespace NUMINAMATH_CALUDE_morning_run_distance_l969_96954

/-- Represents a person's daily activities and distances --/
structure DailyActivities where
  n : ℕ  -- number of stores visited
  x : ℝ  -- morning run distance
  total_distance : ℝ  -- total distance for the day
  bike_distance : ℝ  -- evening bike ride distance

/-- Theorem stating the relationship between morning run distance and other factors --/
theorem morning_run_distance (d : DailyActivities) 
  (h1 : d.total_distance = 18) 
  (h2 : d.bike_distance = 12) 
  (h3 : d.total_distance = d.x + 2 * d.n * d.x + d.bike_distance) :
  d.x = 6 / (1 + 2 * d.n) := by
  sorry

end NUMINAMATH_CALUDE_morning_run_distance_l969_96954


namespace NUMINAMATH_CALUDE_inequality_proof_l969_96952

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z ≥ 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 ∧
  ((1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l969_96952


namespace NUMINAMATH_CALUDE_janet_muffins_count_l969_96960

theorem janet_muffins_count :
  ∀ (muffin_cost : ℚ) (paid : ℚ) (change : ℚ),
    muffin_cost = 75 / 100 →
    paid = 20 →
    change = 11 →
    (paid - change) / muffin_cost = 12 :=
by sorry

end NUMINAMATH_CALUDE_janet_muffins_count_l969_96960


namespace NUMINAMATH_CALUDE_pencils_to_library_l969_96982

theorem pencils_to_library (total_pencils : Nat) (num_classrooms : Nat) 
    (h1 : total_pencils = 935) 
    (h2 : num_classrooms = 9) : 
  total_pencils % num_classrooms = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencils_to_library_l969_96982


namespace NUMINAMATH_CALUDE_fraction_inequality_l969_96903

theorem fraction_inequality (a b c d : ℕ+) (h1 : a + c < 1988) 
  (h2 : (1 : ℚ) - a / b - c / d > 0) : (1 : ℚ) - a / b - c / d > 1 / (1988^3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l969_96903


namespace NUMINAMATH_CALUDE_mixed_nuts_cost_l969_96987

/-- Represents the price and amount of a type of nut -/
structure NutInfo where
  price : ℚ  -- Price in dollars
  amount : ℚ  -- Amount in ounces
  deriving Repr

/-- Calculates the discounted price per ounce -/
def discountedPricePerOz (info : NutInfo) (discount : ℚ) : ℚ :=
  (info.price / info.amount) * (1 - discount)

/-- Calculates the cost of a nut in the mix -/
def nutCostInMix (pricePerOz : ℚ) (proportion : ℚ) : ℚ :=
  pricePerOz * proportion

/-- The main theorem stating the minimum cost of the mixed nuts -/
theorem mixed_nuts_cost
  (almond_info : NutInfo)
  (cashew_info : NutInfo)
  (walnut_info : NutInfo)
  (almond_discount cashew_discount walnut_discount : ℚ)
  (h_almond_price : almond_info.price = 18)
  (h_almond_amount : almond_info.amount = 32)
  (h_cashew_price : cashew_info.price = 45/2)
  (h_cashew_amount : cashew_info.amount = 28)
  (h_walnut_price : walnut_info.price = 15)
  (h_walnut_amount : walnut_info.amount = 24)
  (h_almond_discount : almond_discount = 1/10)
  (h_cashew_discount : cashew_discount = 3/20)
  (h_walnut_discount : walnut_discount = 1/5)
  : ∃ (cost : ℕ), cost = 56 ∧ 
    cost * (1/100 : ℚ) ≥ 
      nutCostInMix (discountedPricePerOz almond_info almond_discount) (1/2) +
      nutCostInMix (discountedPricePerOz cashew_info cashew_discount) (3/10) +
      nutCostInMix (discountedPricePerOz walnut_info walnut_discount) (1/5) :=
sorry

end NUMINAMATH_CALUDE_mixed_nuts_cost_l969_96987


namespace NUMINAMATH_CALUDE_no_polynomial_exists_l969_96914

theorem no_polynomial_exists : ¬∃ (P : ℤ → ℤ) (a b c d : ℤ),
  (∀ n : ℕ, ∃ k : ℤ, P n = k) ∧  -- P has integer coefficients
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- a, b, c, d are distinct
  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4 := by
sorry

end NUMINAMATH_CALUDE_no_polynomial_exists_l969_96914


namespace NUMINAMATH_CALUDE_lighthouse_min_fuel_l969_96974

/-- Represents the lighthouse generator's operation parameters -/
structure LighthouseGenerator where
  fuel_per_hour : ℝ
  startup_fuel : ℝ
  total_hours : ℝ
  max_stop_time : ℝ
  min_run_time : ℝ

/-- Calculates the minimum fuel required for the lighthouse generator -/
def min_fuel_required (g : LighthouseGenerator) : ℝ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating the minimum fuel required for the given parameters -/
theorem lighthouse_min_fuel :
  let g : LighthouseGenerator := {
    fuel_per_hour := 6,
    startup_fuel := 0.5,
    total_hours := 10,
    max_stop_time := 1/6,  -- 10 minutes in hours
    min_run_time := 1/4    -- 15 minutes in hours
  }
  min_fuel_required g = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_lighthouse_min_fuel_l969_96974


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l969_96990

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ x ∈ Set.Icc (-2) 1 ∪ Set.Icc 5 8 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l969_96990


namespace NUMINAMATH_CALUDE_min_value_floor_sum_l969_96958

theorem min_value_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(2*x+y)/z⌋ + ⌊(2*y+z)/x⌋ + ⌊(2*z+x)/y⌋ ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_floor_sum_l969_96958


namespace NUMINAMATH_CALUDE_f_non_monotonic_iff_a_in_range_l969_96933

-- Define the piecewise function f(x)
noncomputable def f (a t x : ℝ) : ℝ :=
  if x ≤ t then (4*a - 3)*x + 2*a - 4 else 2*x^3 - 6*x

-- Define the property of being non-monotonic on ℝ
def is_non_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ (f x < f y ∧ f y > f z ∨ f x > f y ∧ f y < f z)

-- State the theorem
theorem f_non_monotonic_iff_a_in_range :
  (∀ t : ℝ, is_non_monotonic (f a t)) ↔ a ∈ Set.Iic (3/4) :=
sorry

end NUMINAMATH_CALUDE_f_non_monotonic_iff_a_in_range_l969_96933


namespace NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l969_96989

theorem largest_four_digit_number_with_conditions : 
  ∀ n : ℕ, 
  n ≤ 9999 ∧ n ≥ 1000 ∧ 
  ∃ k : ℕ, n = 11 * k + 2 ∧
  ∃ m : ℕ, n = 7 * m + 4 
  → n ≤ 9979 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l969_96989


namespace NUMINAMATH_CALUDE_vector_operations_l969_96909

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (-1, 3)

theorem vector_operations :
  (a.1 + b.1, a.2 + b.2) = (1, 3) ∧
  (a.1 - b.1, a.2 - b.2) = (3, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l969_96909


namespace NUMINAMATH_CALUDE_polynomial_factorization_l969_96913

theorem polynomial_factorization (a b : ℝ) : 
  a^2 - b^2 + 2*a + 1 = (a-b+1)*(a+b+1) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l969_96913


namespace NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l969_96988

/-- Plane represented by its normal vector and constant term -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Line represented by a point and a direction vector -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

def plane_intersection (p1 p2 : Plane) : Line := sorry

def distance_point_to_plane (point : ℝ × ℝ × ℝ) (plane : Plane) : ℝ := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem plane_Q_satisfies_conditions : 
  let π₁ : Plane := ⟨2, -3, 4, -5⟩
  let π₂ : Plane := ⟨3, 1, -2, -1⟩
  let Q : Plane := ⟨6, -1, 10, -11⟩
  let intersection := plane_intersection π₁ π₂
  let point := (1, 2, 3)
  line_in_plane intersection Q ∧ 
  distance_point_to_plane point Q = 3 / Real.sqrt 5 ∧
  Q ≠ π₁ ∧ 
  Q ≠ π₂ := by
  sorry


end NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l969_96988


namespace NUMINAMATH_CALUDE_power_inequality_l969_96935

theorem power_inequality (a b x y : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hab : a^5 + b^5 ≤ 1) (hxy : x^5 + y^5 ≤ 1) : 
  a^2 * x^3 + b^2 * y^3 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_power_inequality_l969_96935


namespace NUMINAMATH_CALUDE_smallest_equal_flock_size_l969_96906

theorem smallest_equal_flock_size (duck_flock_size crane_flock_size : ℕ) 
  (duck_flock_size_pos : duck_flock_size > 0)
  (crane_flock_size_pos : crane_flock_size > 0)
  (duck_flock_size_eq : duck_flock_size = 13)
  (crane_flock_size_eq : crane_flock_size = 17) :
  ∃ n : ℕ, n > 0 ∧ 
    n % duck_flock_size = 0 ∧ 
    n % crane_flock_size = 0 ∧
    (∀ m : ℕ, m > 0 ∧ m % duck_flock_size = 0 ∧ m % crane_flock_size = 0 → m ≥ n) ∧
    n = 221 :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_flock_size_l969_96906


namespace NUMINAMATH_CALUDE_cubic_three_zeros_l969_96955

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem: The cubic function f(x) = x^3 + ax + 2 has exactly 3 real zeros if and only if a < -3 -/
theorem cubic_three_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_l969_96955


namespace NUMINAMATH_CALUDE_intersection_and_subsets_l969_96965

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x ≥ 0}

theorem intersection_and_subsets :
  (A ∩ B = {1, 2}) ∧
  (Set.powerset (A ∩ B) = {{1}, {2}, ∅, {1, 2}}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_subsets_l969_96965


namespace NUMINAMATH_CALUDE_remainder_problem_l969_96940

theorem remainder_problem (a : ℤ) : ∃ (n : ℕ), n > 1 ∧
  (1108 + a) % n = 4 ∧
  1453 % n = 4 ∧
  (1844 + 2*a) % n = 4 ∧
  2281 % n = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l969_96940


namespace NUMINAMATH_CALUDE_sin_480_degrees_l969_96956

theorem sin_480_degrees : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_480_degrees_l969_96956


namespace NUMINAMATH_CALUDE_smallest_with_ten_divisors_l969_96969

/-- A function that counts the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 10 positive divisors -/
def hasTenDivisors (n : ℕ) : Prop := countDivisors n = 10

/-- The theorem stating that 48 is the smallest natural number with exactly 10 positive divisors -/
theorem smallest_with_ten_divisors :
  (∀ m : ℕ, m < 48 → ¬(hasTenDivisors m)) ∧ hasTenDivisors 48 := by sorry

end NUMINAMATH_CALUDE_smallest_with_ten_divisors_l969_96969


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l969_96993

/-- If three lines intersect at one point, then a specific value of a is determined -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, (a * p.1 + 2 * p.2 + 8 = 0) ∧ 
                  (4 * p.1 + 3 * p.2 - 10 = 0) ∧ 
                  (2 * p.1 - p.2 = 0)) → 
  a = -12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l969_96993


namespace NUMINAMATH_CALUDE_greatest_possible_award_l969_96991

theorem greatest_possible_award (total_prize : ℝ) (num_winners : ℕ) (min_award : ℝ) 
  (prize_fraction : ℝ) (winner_fraction : ℝ) :
  total_prize = 400 →
  num_winners = 20 →
  min_award = 20 →
  prize_fraction = 2/5 →
  winner_fraction = 3/5 →
  ∃ (max_award : ℝ), 
    max_award = 100 ∧ 
    max_award ≤ total_prize ∧
    max_award ≥ min_award ∧
    (∀ (award : ℝ), 
      award ≤ total_prize ∧ 
      award ≥ min_award → 
      award ≤ max_award) ∧
    (prize_fraction * total_prize ≤ winner_fraction * num_winners * min_award) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l969_96991


namespace NUMINAMATH_CALUDE_extra_interest_proof_l969_96957

/-- Calculates simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem extra_interest_proof (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) :
  principal = 5000 ∧ rate1 = 0.18 ∧ rate2 = 0.12 ∧ time = 2 →
  simpleInterest principal rate1 time - simpleInterest principal rate2 time = 600 := by
  sorry

end NUMINAMATH_CALUDE_extra_interest_proof_l969_96957


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_30_l969_96905

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_30 :
  units_digit (sum_factorials 30) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_30_l969_96905


namespace NUMINAMATH_CALUDE_kindergarten_allergies_l969_96992

/-- Given a kindergarten with the following conditions:
  - T is the total number of children
  - Half of the children are allergic to peanuts
  - 10 children are not allergic to cashew nuts
  - 10 children are allergic to both peanuts and cashew nuts
  - Some children are allergic to cashew nuts
Prove that the number of children not allergic to peanuts and not allergic to cashew nuts is 10 -/
theorem kindergarten_allergies (T : ℕ) : 
  T > 0 →
  T / 2 = (T - T / 2) → -- Half of the children are allergic to peanuts
  ∃ (cashew_allergic : ℕ), cashew_allergic > 0 ∧ cashew_allergic < T → -- Some children are allergic to cashew nuts
  10 = T - cashew_allergic → -- 10 children are not allergic to cashew nuts
  10 ≤ T / 2 → -- 10 children are allergic to both peanuts and cashew nuts
  10 = T - (T / 2 + cashew_allergic - 10) -- Number of children not allergic to peanuts and not allergic to cashew nuts
  := by sorry

end NUMINAMATH_CALUDE_kindergarten_allergies_l969_96992


namespace NUMINAMATH_CALUDE_compute_expression_l969_96925

theorem compute_expression : 25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l969_96925


namespace NUMINAMATH_CALUDE_jimin_tape_length_l969_96917

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Define Jungkook's tape length in cm
def jungkook_tape_cm : ℝ := 45

-- Define the difference between Jimin's and Jungkook's tape lengths in mm
def tape_difference_mm : ℝ := 26

-- State the theorem
theorem jimin_tape_length :
  (jungkook_tape_cm * cm_to_mm + tape_difference_mm) / cm_to_mm = 47.6 := by
  sorry

end NUMINAMATH_CALUDE_jimin_tape_length_l969_96917


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l969_96901

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  a = 2 * Real.sqrt 3 →
  c = 2 * Real.sqrt 2 →
  A = π / 3 →
  (a / Real.sin A = c / Real.sin C) →
  C < π / 2 →
  C = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l969_96901


namespace NUMINAMATH_CALUDE_square_root_equality_l969_96931

theorem square_root_equality (x a : ℝ) (hx : x > 0) :
  (Real.sqrt x = 2 * a - 1 ∧ Real.sqrt x = -a + 2) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l969_96931


namespace NUMINAMATH_CALUDE_no_upper_bound_for_a_l969_96963

/-- The number of different representations of n as the sum of different divisors -/
def a (n : ℕ) : ℕ := sorry

/-- There is no upper bound M for a(n) that holds for all n -/
theorem no_upper_bound_for_a : ∀ M : ℕ, ∃ n : ℕ, a n > M := by sorry

end NUMINAMATH_CALUDE_no_upper_bound_for_a_l969_96963


namespace NUMINAMATH_CALUDE_slope_at_five_is_zero_l969_96912

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem slope_at_five_is_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_period : has_period f 5) :
  deriv f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_slope_at_five_is_zero_l969_96912


namespace NUMINAMATH_CALUDE_expression_simplification_l969_96978

theorem expression_simplification (x : ℝ) (h : x^2 + x - 6 = 0) :
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l969_96978


namespace NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l969_96927

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l969_96927


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l969_96972

theorem complex_number_quadrant : ∃ (z : ℂ), z = (Complex.I : ℂ) / (1 + Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l969_96972


namespace NUMINAMATH_CALUDE_money_division_l969_96964

theorem money_division (a b c : ℚ) 
  (h1 : a = (2/3) * (b + c))
  (h2 : b = (6/9) * (a + c))
  (h3 : a = 280) : 
  a + b + c = 700 := by
sorry

end NUMINAMATH_CALUDE_money_division_l969_96964


namespace NUMINAMATH_CALUDE_most_likely_white_balls_l969_96959

/-- Represents a box of balls -/
structure BallBox where
  total : ℕ
  white : ℕ
  black : ℕ
  white_le_total : white ≤ total
  black_eq_total_sub_white : black = total - white

/-- Represents the result of multiple draws -/
structure DrawResult where
  total_draws : ℕ
  white_draws : ℕ
  white_draws_le_total : white_draws ≤ total_draws

/-- The probability of drawing a white ball given a box configuration -/
def draw_probability (box : BallBox) : ℚ :=
  box.white / box.total

/-- The likelihood of a draw result given a box configuration -/
def draw_likelihood (box : BallBox) (result : DrawResult) : ℚ :=
  (draw_probability box) ^ result.white_draws * (1 - draw_probability box) ^ (result.total_draws - result.white_draws)

/-- Theorem: Given 10 balls and 240 white draws out of 400, 6 white balls is most likely -/
theorem most_likely_white_balls 
  (box : BallBox) 
  (result : DrawResult) 
  (h_total : box.total = 10) 
  (h_draws : result.total_draws = 400) 
  (h_white_draws : result.white_draws = 240) :
  (∀ (other_box : BallBox), other_box.total = 10 → 
    draw_likelihood box result ≥ draw_likelihood other_box result) →
  box.white = 6 :=
sorry

end NUMINAMATH_CALUDE_most_likely_white_balls_l969_96959
