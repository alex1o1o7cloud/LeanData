import Mathlib

namespace NUMINAMATH_CALUDE_total_morning_afternoon_emails_l2037_203749

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- Theorem: The total number of emails Jack received in the morning and afternoon is 13 -/
theorem total_morning_afternoon_emails : morning_emails + afternoon_emails = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_morning_afternoon_emails_l2037_203749


namespace NUMINAMATH_CALUDE_student_D_most_stable_l2037_203768

-- Define the students
inductive Student : Type
  | A
  | B
  | C
  | D

-- Define the variance function
def variance : Student → Real
  | Student.A => 2.1
  | Student.B => 3.5
  | Student.C => 9
  | Student.D => 0.7

-- Define the concept of stability
def most_stable (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

-- Theorem statement
theorem student_D_most_stable :
  most_stable Student.D :=
by sorry

end NUMINAMATH_CALUDE_student_D_most_stable_l2037_203768


namespace NUMINAMATH_CALUDE_michelle_needs_one_more_rack_l2037_203738

/-- Represents the pasta making scenario for Michelle -/
structure PastaMaking where
  flour_per_pound : ℕ  -- cups of flour needed per pound of pasta
  pounds_per_rack : ℕ  -- pounds of pasta that can fit on one rack
  owned_racks : ℕ     -- number of racks Michelle currently owns
  flour_bags : ℕ      -- number of flour bags
  cups_per_bag : ℕ    -- cups of flour in each bag

/-- Calculates the number of additional racks Michelle needs -/
def additional_racks_needed (pm : PastaMaking) : ℕ :=
  let total_flour := pm.flour_bags * pm.cups_per_bag
  let total_pounds := total_flour / pm.flour_per_pound
  let total_racks_needed := (total_pounds + pm.pounds_per_rack - 1) / pm.pounds_per_rack
  (total_racks_needed - pm.owned_racks).max 0

/-- Theorem stating that Michelle needs one more rack -/
theorem michelle_needs_one_more_rack :
  let pm : PastaMaking := {
    flour_per_pound := 2,
    pounds_per_rack := 3,
    owned_racks := 3,
    flour_bags := 3,
    cups_per_bag := 8
  }
  additional_racks_needed pm = 1 := by sorry

end NUMINAMATH_CALUDE_michelle_needs_one_more_rack_l2037_203738


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l2037_203751

/-- The number of lamp posts along the alley -/
def total_posts : ℕ := 400

/-- The lamp post number where Alla is observed -/
def alla_observed : ℕ := 55

/-- The lamp post number where Boris is observed -/
def boris_observed : ℕ := 321

/-- The function to calculate the meeting point of Alla and Boris -/
def meeting_point : ℕ :=
  let alla_traveled := alla_observed - 1
  let boris_traveled := total_posts - boris_observed
  let total_traveled := alla_traveled + boris_traveled
  let alla_to_meeting := 3 * alla_traveled
  1 + alla_to_meeting

/-- Theorem stating that Alla and Boris will meet at lamp post 163 -/
theorem alla_boris_meeting :
  meeting_point = 163 := by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l2037_203751


namespace NUMINAMATH_CALUDE_circumcenter_property_implies_isosceles_l2037_203752

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector addition and scalar multiplication
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Define an isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2 ∨
  (t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2 = (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 ∨
  (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 = (t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2

-- The main theorem
theorem circumcenter_property_implies_isosceles (t : Triangle) :
  let O := circumcenter t
  vec_add (vec_add (vec_add O (vec_scale (-1) t.A)) (vec_add O (vec_scale (-1) t.B))) (vec_scale (Real.sqrt 2) (vec_add O (vec_scale (-1) t.C))) = (0, 0)
  → is_isosceles t :=
sorry

end NUMINAMATH_CALUDE_circumcenter_property_implies_isosceles_l2037_203752


namespace NUMINAMATH_CALUDE_x_value_in_set_l2037_203770

theorem x_value_in_set (x : ℝ) : x ∈ ({1, 2, x^2 - x} : Set ℝ) → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_l2037_203770


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l2037_203750

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - 2*x*y = 0) :
  2*x + y ≥ 9/2 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ - 2*x₀*y₀ = 0 ∧ 2*x₀ + y₀ = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l2037_203750


namespace NUMINAMATH_CALUDE_janinas_pancakes_sales_l2037_203703

/-- The number of pancakes Janina must sell daily to cover her expenses -/
def pancakes_to_sell (daily_rent : ℕ) (daily_supplies : ℕ) (price_per_pancake : ℕ) : ℕ :=
  (daily_rent + daily_supplies) / price_per_pancake

/-- Theorem stating that Janina must sell 21 pancakes daily to cover her expenses -/
theorem janinas_pancakes_sales : pancakes_to_sell 30 12 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_janinas_pancakes_sales_l2037_203703


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2037_203790

theorem sqrt_equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3) ↔
  x ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2037_203790


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l2037_203728

theorem no_rational_solutions_for_positive_k : 
  ¬ ∃ (k : ℕ+) (x : ℚ), k * x^2 + 30 * x + k = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l2037_203728


namespace NUMINAMATH_CALUDE_max_profit_at_optimal_price_l2037_203753

/-- Profit function for a store selling items -/
def profit_function (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (sales_increase_rate : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_price - price_reduction - cost_price) * (initial_sales + sales_increase_rate * price_reduction)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit_at_optimal_price 
  (cost_price : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : cost_price = 2)
  (h2 : initial_price = 13)
  (h3 : initial_sales = 500)
  (h4 : sales_increase_rate = 100) :
  ∃ (optimal_price_reduction : ℝ),
    optimal_price_reduction = 3 ∧
    profit_function cost_price initial_price initial_sales sales_increase_rate optimal_price_reduction = 6400 ∧
    ∀ (price_reduction : ℝ), 
      profit_function cost_price initial_price initial_sales sales_increase_rate price_reduction ≤ 
      profit_function cost_price initial_price initial_sales sales_increase_rate optimal_price_reduction :=
by
  sorry

#check max_profit_at_optimal_price

end NUMINAMATH_CALUDE_max_profit_at_optimal_price_l2037_203753


namespace NUMINAMATH_CALUDE_bike_travel_time_l2037_203735

-- Constants
def highway_length : Real := 5280  -- in feet
def highway_width : Real := 50     -- in feet
def bike_speed : Real := 6         -- in miles per hour

-- Theorem
theorem bike_travel_time :
  let semicircle_radius : Real := highway_width / 2
  let num_semicircles : Real := highway_length / highway_width
  let total_distance : Real := num_semicircles * (π * semicircle_radius)
  let total_distance_miles : Real := total_distance / 5280
  let time_taken : Real := total_distance_miles / bike_speed
  time_taken = π / 12 := by sorry

end NUMINAMATH_CALUDE_bike_travel_time_l2037_203735


namespace NUMINAMATH_CALUDE_average_of_unknowns_l2037_203788

/-- Given 5 numbers with an average of 20, prove that the average of the two unknown numbers is 40.5 -/
theorem average_of_unknowns (x y : ℝ) : 
  (4 + 6 + 9 + x + y) / 5 = 20 → (x + y) / 2 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_unknowns_l2037_203788


namespace NUMINAMATH_CALUDE_total_cash_is_correct_l2037_203765

/-- Calculates the total cash realized from three stocks after accounting for brokerage fees. -/
def total_cash_realized (stock_a_proceeds stock_b_proceeds stock_c_proceeds : ℝ)
                        (stock_a_brokerage_rate stock_b_brokerage_rate stock_c_brokerage_rate : ℝ) : ℝ :=
  let stock_a_cash := stock_a_proceeds * (1 - stock_a_brokerage_rate)
  let stock_b_cash := stock_b_proceeds * (1 - stock_b_brokerage_rate)
  let stock_c_cash := stock_c_proceeds * (1 - stock_c_brokerage_rate)
  stock_a_cash + stock_b_cash + stock_c_cash

/-- Theorem stating that the total cash realized from the given stock sales is equal to 463.578625. -/
theorem total_cash_is_correct : 
  total_cash_realized 107.25 155.40 203.50 (1/400) (1/200) (3/400) = 463.578625 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_is_correct_l2037_203765


namespace NUMINAMATH_CALUDE_correct_probability_order_l2037_203757

/-- Enum representing the five types of phenomena -/
inductive Phenomenon
  | CertainToHappen
  | VeryLikelyToHappen
  | PossibleToHappen
  | ImpossibleToHappen
  | UnlikelyToHappen

/-- Function to compare the probability of two phenomena -/
def probabilityLessThan (a b : Phenomenon) : Prop :=
  match a, b with
  | Phenomenon.ImpossibleToHappen, _ => a ≠ b
  | Phenomenon.UnlikelyToHappen, Phenomenon.ImpossibleToHappen => False
  | Phenomenon.UnlikelyToHappen, _ => a ≠ b
  | Phenomenon.PossibleToHappen, Phenomenon.ImpossibleToHappen => False
  | Phenomenon.PossibleToHappen, Phenomenon.UnlikelyToHappen => False
  | Phenomenon.PossibleToHappen, _ => a ≠ b
  | Phenomenon.VeryLikelyToHappen, Phenomenon.CertainToHappen => True
  | Phenomenon.VeryLikelyToHappen, _ => False
  | Phenomenon.CertainToHappen, _ => False

/-- Theorem stating the correct order of phenomena by probability -/
theorem correct_probability_order :
  probabilityLessThan Phenomenon.ImpossibleToHappen Phenomenon.UnlikelyToHappen ∧
  probabilityLessThan Phenomenon.UnlikelyToHappen Phenomenon.PossibleToHappen ∧
  probabilityLessThan Phenomenon.PossibleToHappen Phenomenon.VeryLikelyToHappen ∧
  probabilityLessThan Phenomenon.VeryLikelyToHappen Phenomenon.CertainToHappen :=
sorry

end NUMINAMATH_CALUDE_correct_probability_order_l2037_203757


namespace NUMINAMATH_CALUDE_age_problem_solution_l2037_203705

def age_problem (thomas_age shay_age james_age violet_age emily_age : ℕ) : Prop :=
  thomas_age = 6 ∧
  shay_age = thomas_age + 13 ∧
  james_age = shay_age + 5 ∧
  violet_age = thomas_age - 3 ∧
  emily_age = shay_age ∧
  (james_age + (thomas_age - violet_age) = 27) ∧
  (emily_age + (thomas_age - violet_age) = 22)

theorem age_problem_solution :
  ∃ (thomas_age shay_age james_age violet_age emily_age : ℕ),
    age_problem thomas_age shay_age james_age violet_age emily_age :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_solution_l2037_203705


namespace NUMINAMATH_CALUDE_ratio_problem_l2037_203747

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.75 * a) (h5 : m = b - 0.8 * b) : m / x = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2037_203747


namespace NUMINAMATH_CALUDE_descending_order_of_powers_l2037_203741

theorem descending_order_of_powers : 
  2^(2/3) > (-1.8)^(2/3) ∧ (-1.8)^(2/3) > (-2)^(1/3) := by sorry

end NUMINAMATH_CALUDE_descending_order_of_powers_l2037_203741


namespace NUMINAMATH_CALUDE_triangle_inequality_l2037_203759

theorem triangle_inequality (A B C : ℝ) (x y z : ℝ) (n : ℕ) 
  (h_triangle : A + B + C = π) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0) : 
  x^n * Real.cos (A/2) + y^n * Real.cos (B/2) + z^n * Real.cos (C/2) ≥ 
  (y*z)^(n/2) * Real.sin A + (z*x)^(n/2) * Real.sin B + (x*y)^(n/2) * Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2037_203759


namespace NUMINAMATH_CALUDE_polynomial_value_at_3_l2037_203758

def is_valid_coeff (b : ℤ) : Prop := 0 ≤ b ∧ b < 5

def P (b : Fin 6 → ℤ) (x : ℝ) : ℝ :=
  (b 0) + (b 1) * x + (b 2) * x^2 + (b 3) * x^3 + (b 4) * x^4 + (b 5) * x^5

theorem polynomial_value_at_3 (b : Fin 6 → ℤ) :
  (∀ i : Fin 6, is_valid_coeff (b i)) →
  P b (Real.sqrt 5) = 40 + 31 * Real.sqrt 5 →
  P b 3 = 381 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_3_l2037_203758


namespace NUMINAMATH_CALUDE_matrix_power_50_l2037_203740

/-- Given a 2x2 matrix C, prove that its 50th power is equal to a specific matrix. -/
theorem matrix_power_50 (C : Matrix (Fin 2) (Fin 2) ℤ) : 
  C = !![5, 2; -16, -6] → C^50 = !![-299, -100; 800, 249] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_50_l2037_203740


namespace NUMINAMATH_CALUDE_emilees_earnings_l2037_203700

/-- Given the earnings and work conditions of Jermaine, Terrence, and Emilee, prove Emilee's earnings. -/
theorem emilees_earnings 
  (total_earnings : ℝ)
  (j_hours r_j : ℝ)
  (t_hours r_t : ℝ)
  (e_hours r_e : ℝ)
  (h1 : total_earnings = 90)
  (h2 : r_j * j_hours = r_t * t_hours + 5)
  (h3 : r_t * t_hours = 30)
  (h4 : total_earnings = r_j * j_hours + r_t * t_hours + r_e * e_hours) :
  r_e * e_hours = 25 := by
  sorry

end NUMINAMATH_CALUDE_emilees_earnings_l2037_203700


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2037_203792

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {2, 3, 4}

theorem intersection_complement_theorem : A ∩ (U \ B) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2037_203792


namespace NUMINAMATH_CALUDE_prism_volume_l2037_203754

/-- Given a right rectangular prism with dimensions a, b, and c,
    if the areas of three faces are 30, 45, and 54 square centimeters,
    then the volume of the prism is 270 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 45) 
  (h3 : b * c = 54) : 
  a * b * c = 270 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l2037_203754


namespace NUMINAMATH_CALUDE_triangle_area_l2037_203737

/-- Given a triangle ABC with the following properties:
  1. The side opposite to angle B has length 1
  2. Angle B measures π/6 radians
  3. 1/tan(A) + 1/tan(C) = 2
Prove that the area of the triangle is 1/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b = 1 → 
  B = π / 6 → 
  1 / Real.tan A + 1 / Real.tan C = 2 → 
  (1 / 2) * a * b * Real.sin C = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2037_203737


namespace NUMINAMATH_CALUDE_largest_integer_K_l2037_203785

theorem largest_integer_K : ∃ K : ℕ, (∀ n : ℕ, n^200 < 5^300 → n ≤ K) ∧ K^200 < 5^300 ∧ K = 11 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_K_l2037_203785


namespace NUMINAMATH_CALUDE_number_division_problem_l2037_203769

theorem number_division_problem (x : ℝ) : (x / 5 = 75 + x / 6) ↔ (x = 2250) := by sorry

end NUMINAMATH_CALUDE_number_division_problem_l2037_203769


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l2037_203773

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + 2*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + 2*a = 0) :
  2*a + b ≥ 2 * Real.sqrt (3 * Real.rpow (8/3) (1/3)) + Real.rpow (8/3) (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l2037_203773


namespace NUMINAMATH_CALUDE_molecular_weight_CaI2_l2037_203761

/-- Given that the molecular weight of 3 moles of CaI2 is 882 g/mol,
    prove that the molecular weight of 1 mole of CaI2 is 294 g/mol. -/
theorem molecular_weight_CaI2 (weight_3_moles : ℝ) (h : weight_3_moles = 882) :
  weight_3_moles / 3 = 294 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CaI2_l2037_203761


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_increasing_l2037_203766

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The condition a₁ < a₂ < a₃ -/
def FirstThreeIncreasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_increasing_iff_first_three_increasing
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  IncreasingSequence a ↔ FirstThreeIncreasing a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_increasing_l2037_203766


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2037_203784

theorem unique_three_digit_number : ∃! (m g u : ℕ),
  m ≠ g ∧ m ≠ u ∧ g ≠ u ∧
  m < 10 ∧ g < 10 ∧ u < 10 ∧
  m ≥ 1 ∧
  100 * m + 10 * g + u = (m + g + u) * (m + g + u - 2) ∧
  100 * m + 10 * g + u = 195 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2037_203784


namespace NUMINAMATH_CALUDE_river_speed_l2037_203733

/-- Proves that the speed of the river is 1.2 kmph given the conditions -/
theorem river_speed (rowing_speed : ℝ) (total_time : ℝ) (total_distance : ℝ)
  (h1 : rowing_speed = 8)
  (h2 : total_time = 1)
  (h3 : total_distance = 7.82) :
  ∃ v : ℝ, v = 1.2 ∧
  (total_distance / 2) / (rowing_speed - v) + (total_distance / 2) / (rowing_speed + v) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_speed_l2037_203733


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2037_203756

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 9241 ↔ 
  n > 1 ∧ 
  (∀ d ∈ ({5, 7, 8, 10, 11, 12} : Set ℕ), n % d = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({5, 7, 8, 10, 11, 12} : Set ℕ), m % d = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2037_203756


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2037_203796

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 2 * x₁^2 - 7 * x₁ + k = 0 ∧ 2 * x₂^2 - 7 * x₂ + k = 0 ∧ x₁ = 2) →
  (∃ x₂ : ℝ, x₂ = 3/2 ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2037_203796


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2037_203720

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y ≠ 1 ∧ 3 * y^2 - m * y - 3 = 0 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2037_203720


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l2037_203744

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The number of possible sets of four collinear dots in a 5x5 grid -/
def collinearSets : ℕ := 16

/-- The total number of ways to choose 4 dots from 25 -/
def totalChoices : ℕ := 12650

/-- The probability of selecting four collinear dots in a 5x5 grid -/
theorem collinear_dots_probability :
  (collinearSets : ℚ) / totalChoices = 8 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l2037_203744


namespace NUMINAMATH_CALUDE_smallest_number_with_divisibility_property_l2037_203746

theorem smallest_number_with_divisibility_property : 
  ∀ n : ℕ, n > 0 → (n + 9) % 8 = 0 ∧ (n + 9) % 11 = 0 ∧ (∃ k : ℕ, k > 1 ∧ k ≠ 8 ∧ k ≠ 11 ∧ n % k = 0) → n ≥ 255 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_divisibility_property_l2037_203746


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2037_203767

def has_solution (n : ℕ) : Prop :=
  ∃ (a b c : ℤ), a^n + b^n = c^n + n

theorem diophantine_equation_solutions :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → (has_solution n ↔ n = 1 ∨ n = 2 ∨ n = 3)) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2037_203767


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2037_203726

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {-1, 1}
def B : Finset Int := {0, 1, 2}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2037_203726


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2037_203783

/-- Given a quadratic function f(x) = x^2 + bx - 5 with symmetric axis x = 2,
    prove that the solutions to f(x) = 2x - 13 are x₁ = 2 and x₂ = 4. -/
theorem quadratic_equation_solution (b : ℝ) :
  (∀ x, x^2 + b*x - 5 = x^2 + b*x - 5) →  -- f(x) is a well-defined function
  (-b/2 = 2) →                            -- symmetric axis is x = 2
  (∃ x₁ x₂, x₁ = 2 ∧ x₂ = 4 ∧
    (∀ x, x^2 + b*x - 5 = 2*x - 13 ↔ x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2037_203783


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2037_203732

theorem quadratic_one_solution_sum (b : ℝ) : 
  let equation := fun (x : ℝ) => 3 * x^2 + b * x + 6 * x + 14
  let discriminant := (b + 6)^2 - 4 * 3 * 14
  (∃! x, equation x = 0) → 
  (∃ b₁ b₂, b = b₁ ∨ b = b₂) ∧ (b₁ + b₂ = -12) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2037_203732


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_near_twin_primes_divisible_by_240_l2037_203762

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q = p + 2

theorem product_of_consecutive_integers_near_twin_primes_divisible_by_240 
  (p : ℕ) (h1 : p > 7) (h2 : are_twin_primes p (p + 2)) : 
  240 ∣ ((p - 1) * p * (p + 1)) :=
sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_near_twin_primes_divisible_by_240_l2037_203762


namespace NUMINAMATH_CALUDE_roots_product_plus_one_l2037_203748

theorem roots_product_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (1+a)*(1+b)*(1+c) = 51 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_plus_one_l2037_203748


namespace NUMINAMATH_CALUDE_simplify_expression_l2037_203716

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 16) :
  (4 - Real.sqrt (x^2 - 16))^2 = x^2 - 8 * Real.sqrt (x^2 - 16) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2037_203716


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2037_203731

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 7) = 9 → x = 74 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2037_203731


namespace NUMINAMATH_CALUDE_mirasol_account_balance_l2037_203799

def remaining_balance (initial_balance : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial_balance - (expense1 + expense2)

theorem mirasol_account_balance : remaining_balance 50 10 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mirasol_account_balance_l2037_203799


namespace NUMINAMATH_CALUDE_determinant_of_geometric_sequence_l2037_203722

-- Define a geometric sequence of four terms
def is_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r

-- State the theorem
theorem determinant_of_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) :
  is_geometric_sequence a₁ a₂ a₃ a₄ → a₁ * a₄ - a₂ * a₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_geometric_sequence_l2037_203722


namespace NUMINAMATH_CALUDE_expression_value_l2037_203723

theorem expression_value (a b c d x : ℝ) : 
  a = -b → cd = 1 → abs x = 2 → 
  x^2 - (a + b + cd) * x + (a + b)^2021 + (-cd)^2022 = 3 ∨
  x^2 - (a + b + cd) * x + (a + b)^2021 + (-cd)^2022 = 7 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2037_203723


namespace NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l2037_203771

theorem abc_and_fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l2037_203771


namespace NUMINAMATH_CALUDE_jeremy_song_count_l2037_203730

/-- The number of songs Jeremy listened to yesterday -/
def songs_yesterday : ℕ := 9

/-- The difference in songs between today and yesterday -/
def song_difference : ℕ := 5

/-- The number of songs Jeremy listened to today -/
def songs_today : ℕ := songs_yesterday + song_difference

/-- The total number of songs Jeremy listened to in two days -/
def total_songs : ℕ := songs_yesterday + songs_today

theorem jeremy_song_count : total_songs = 23 := by sorry

end NUMINAMATH_CALUDE_jeremy_song_count_l2037_203730


namespace NUMINAMATH_CALUDE_cubic_roots_negative_real_parts_l2037_203701

theorem cubic_roots_negative_real_parts
  (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℂ, a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃ = 0 → (x.re < 0)) ↔
  ((a₀ > 0 ∧ a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) ∨ (a₀ < 0 ∧ a₁ < 0 ∧ a₂ < 0 ∧ a₃ < 0)) ∧
  a₁ * a₂ - a₀ * a₃ > 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_negative_real_parts_l2037_203701


namespace NUMINAMATH_CALUDE_smallest_with_ten_factors_l2037_203718

/-- A function that returns the number of distinct positive factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly ten distinct positive factors -/
def has_ten_factors (n : ℕ) : Prop := num_factors n = 10

theorem smallest_with_ten_factors :
  ∃ (n : ℕ), has_ten_factors n ∧ ∀ m : ℕ, m < n → ¬(has_ten_factors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_ten_factors_l2037_203718


namespace NUMINAMATH_CALUDE_robot_path_area_l2037_203706

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a closed path on a 2D plane -/
structure ClosedPath where
  vertices : List Point

/-- Function to calculate the area of a closed path -/
noncomputable def areaOfClosedPath (path : ClosedPath) : ℝ :=
  sorry

/-- The specific closed path followed by the robot -/
def robotPath : ClosedPath :=
  sorry

/-- Theorem stating that the area of the robot's path is 13√3/4 -/
theorem robot_path_area :
  areaOfClosedPath robotPath = (13 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_robot_path_area_l2037_203706


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2037_203711

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ + 4 = 5*x₁ + 6) ∧ 
  (x₂^2 + 2*x₂ + 4 = 5*x₂ + 6) → 
  x₁*x₂ + x₁ + x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2037_203711


namespace NUMINAMATH_CALUDE_janet_paper_clips_used_l2037_203727

/-- Calculates the number of paper clips Janet used during the day -/
def paperClipsUsed (initial : ℝ) (found : ℝ) (givenPerFriend : ℝ) (numFriends : ℕ) (final : ℝ) : ℝ :=
  initial + found - givenPerFriend * (numFriends : ℝ) - final

/-- Theorem stating that Janet used 62.5 paper clips during the day -/
theorem janet_paper_clips_used :
  paperClipsUsed 85 17.5 3.5 4 26 = 62.5 := by
  sorry

#eval paperClipsUsed 85 17.5 3.5 4 26

end NUMINAMATH_CALUDE_janet_paper_clips_used_l2037_203727


namespace NUMINAMATH_CALUDE_negation_equivalence_l2037_203712

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ + 2 < 0) ↔ (∀ x : ℝ, x^2 + x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2037_203712


namespace NUMINAMATH_CALUDE_initial_roses_l2037_203794

theorem initial_roses (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 10 → total = 16 → total = initial + added → initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_l2037_203794


namespace NUMINAMATH_CALUDE_T_equals_five_l2037_203745

theorem T_equals_five :
  let T := 1 / (3 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
           1 / (Real.sqrt 7 - Real.sqrt 6) - 1 / (Real.sqrt 6 - Real.sqrt 5) + 
           1 / (Real.sqrt 5 - 2)
  T = 5 := by sorry

end NUMINAMATH_CALUDE_T_equals_five_l2037_203745


namespace NUMINAMATH_CALUDE_sequence_non_negative_l2037_203715

/-- Sequence a_n defined recursively -/
def a : ℕ → ℝ → ℝ
  | 0, a₀ => a₀
  | n + 1, a₀ => 2 * a n a₀ - n^2

/-- Theorem stating the condition for non-negativity of the sequence -/
theorem sequence_non_negative (a₀ : ℝ) :
  (∀ n : ℕ, a n a₀ ≥ 0) ↔ a₀ ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_negative_l2037_203715


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2037_203778

/-- An equilateral triangle with area twice the side length has perimeter 8√3 -/
theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2037_203778


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2037_203736

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 2 + a 3 = 1 ∧
  a 2 + a 3 + a 4 = 2

/-- The theorem stating the sum of the 6th, 7th, and 8th terms -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2037_203736


namespace NUMINAMATH_CALUDE_four_digit_sum_plus_2001_l2037_203710

theorem four_digit_sum_plus_2001 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧
  n = (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10) + 2001 ∧
  n = 1977 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_plus_2001_l2037_203710


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2037_203760

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x + 16) → (∃ y : ℝ, y^2 = 10*y + 16 ∧ x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2037_203760


namespace NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_l2037_203729

-- Define the angle α
def α : Real := sorry

-- Define the conditions
axiom vertex_at_origin : True
axiom initial_side_on_x_axis : True
axiom terminal_side_on_line : ∀ (x y : Real), y = Real.sqrt 3 * x → (∃ t : Real, t > 0 ∧ x = t * Real.cos α ∧ y = t * Real.sin α)

-- State the theorem
theorem tan_two_alpha_plus_pi : Real.tan (2 * α + Real.pi) = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_l2037_203729


namespace NUMINAMATH_CALUDE_sara_peaches_total_l2037_203786

/-- The total number of peaches Sara picked -/
def total_peaches (initial_peaches additional_peaches : Float) : Float :=
  initial_peaches + additional_peaches

/-- Theorem stating that Sara picked 85.0 peaches in total -/
theorem sara_peaches_total :
  let initial_peaches : Float := 61.0
  let additional_peaches : Float := 24.0
  total_peaches initial_peaches additional_peaches = 85.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_peaches_total_l2037_203786


namespace NUMINAMATH_CALUDE_rectangular_grid_toothpicks_l2037_203798

/-- Calculates the number of toothpicks in a rectangular grid. -/
def toothpick_count (height : ℕ) (width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height

/-- Theorem: A rectangular grid of toothpicks that is 20 high and 10 wide uses 430 toothpicks. -/
theorem rectangular_grid_toothpicks : toothpick_count 20 10 = 430 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_grid_toothpicks_l2037_203798


namespace NUMINAMATH_CALUDE_slope_point_relation_l2037_203725

theorem slope_point_relation (m : ℝ) : 
  m > 0 → 
  ((m + 1 - 4) / (2 - m) = Real.sqrt 5) → 
  m = (10 - Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_slope_point_relation_l2037_203725


namespace NUMINAMATH_CALUDE_choir_size_proof_choir_size_minimum_l2037_203764

theorem choir_size_proof : 
  ∀ n : ℕ, (n % 9 = 0 ∧ n % 11 = 0 ∧ n % 13 = 0 ∧ n % 10 = 0) → n ≥ 12870 :=
by
  sorry

theorem choir_size_minimum : 
  12870 % 9 = 0 ∧ 12870 % 11 = 0 ∧ 12870 % 13 = 0 ∧ 12870 % 10 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_choir_size_proof_choir_size_minimum_l2037_203764


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2037_203713

-- Problem 1
theorem problem_1 (a b : ℝ) (ha : a ≠ 0) : (2 * a^2 * b) * a * b^2 / (4 * a^3) = (1/2) * b^3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (2*x + 5) * (x - 3) = 2*x^2 - x - 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2037_203713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2037_203734

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : seq.S 3 = 9) (h₂ : seq.S 6 = 36) : 
  seq.a 6 + seq.a 7 + seq.a 8 = 39 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2037_203734


namespace NUMINAMATH_CALUDE_power_of_product_l2037_203777

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2037_203777


namespace NUMINAMATH_CALUDE_trapezoid_area_proof_l2037_203782

/-- The area of a trapezoid bounded by the lines y = x + 1, y = 15, y = 8, and the y-axis -/
def trapezoid_area : ℝ := 73.5

/-- The line y = x + 1 -/
def line1 (x : ℝ) : ℝ := x + 1

/-- The line y = 15 -/
def line2 : ℝ := 15

/-- The line y = 8 -/
def line3 : ℝ := 8

theorem trapezoid_area_proof :
  let x1 := (line2 - 1 : ℝ)  -- x-coordinate where y = x + 1 intersects y = 15
  let x2 := (line3 - 1 : ℝ)  -- x-coordinate where y = x + 1 intersects y = 8
  let base1 := x1
  let base2 := x2
  let height := line2 - line3
  (base1 + base2) * height / 2 = trapezoid_area := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_proof_l2037_203782


namespace NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_line_correct_l2037_203774

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- Define the point (1,0)
def point : ℝ × ℝ := (1, 0)

-- Define parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + 2 * y - 1 = 0

-- Theorem for parallel line
theorem parallel_line_correct :
  (∀ x y : ℝ, parallel_line x y ↔ (given_line x y ∧ parallel_line x y)) ∧
  parallel_line point.1 point.2 :=
sorry

-- Theorem for perpendicular line
theorem perpendicular_line_correct :
  (∀ x y : ℝ, perpendicular_line x y ↔ (given_line x y ∧ perpendicular_line x y)) ∧
  perpendicular_line point.1 point.2 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_line_correct_l2037_203774


namespace NUMINAMATH_CALUDE_pattern1_unique_violation_l2037_203714

/-- Represents a square in a pattern --/
structure Square where
  color : String

/-- Represents a pattern of squares --/
structure Pattern where
  squares : List Square
  arrangement : String

/-- Checks if a pattern can be folded into a cube --/
def can_fold_to_cube (p : Pattern) : Prop :=
  p.squares.length = 6 ∧ p.arrangement ≠ "linear"

/-- Checks if a pattern violates the adjacent color rule --/
def violates_adjacent_color_rule (p : Pattern) : Prop :=
  ∃ (s1 s2 : Square), s1 ∈ p.squares ∧ s2 ∈ p.squares ∧ s1.color = s2.color

/-- The four patterns described in the problem --/
def pattern1 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" }
    ],
    arrangement := "cross"
  }

def pattern2 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }
    ],
    arrangement := "T"
  }

def pattern3 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" },
      { color := "red" }
    ],
    arrangement := "custom"
  }

def pattern4 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" }
    ],
    arrangement := "linear"
  }

/-- The main theorem --/
theorem pattern1_unique_violation :
  (can_fold_to_cube pattern1 ∧ violates_adjacent_color_rule pattern1) ∧
  (¬can_fold_to_cube pattern2 ∨ ¬violates_adjacent_color_rule pattern2) ∧
  (¬can_fold_to_cube pattern3 ∨ ¬violates_adjacent_color_rule pattern3) ∧
  (¬can_fold_to_cube pattern4 ∨ ¬violates_adjacent_color_rule pattern4) :=
sorry

end NUMINAMATH_CALUDE_pattern1_unique_violation_l2037_203714


namespace NUMINAMATH_CALUDE_emily_total_songs_l2037_203775

/-- Calculates the total number of songs Emily has after buying more. -/
def total_songs (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Emily's total number of songs is 13 given the initial and bought amounts. -/
theorem emily_total_songs :
  total_songs 6 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_emily_total_songs_l2037_203775


namespace NUMINAMATH_CALUDE_first_week_rate_is_18_l2037_203763

/-- The daily rate for the first week in a student youth hostel -/
def first_week_rate : ℝ := 18

/-- The daily rate for additional weeks in a student youth hostel -/
def additional_week_rate : ℝ := 14

/-- The total number of days stayed -/
def total_days : ℕ := 23

/-- The total cost for the stay -/
def total_cost : ℝ := 350

/-- Theorem stating that the daily rate for the first week is $18.00 -/
theorem first_week_rate_is_18 :
  first_week_rate * 7 + additional_week_rate * (total_days - 7) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_first_week_rate_is_18_l2037_203763


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2037_203780

theorem initial_money_calculation (X : ℝ) : 
  X - (X / 2 + 50) = 25 → X = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2037_203780


namespace NUMINAMATH_CALUDE_pet_store_gerbils_l2037_203709

theorem pet_store_gerbils : 
  ∀ (initial_gerbils sold_gerbils remaining_gerbils : ℕ),
  sold_gerbils = 69 →
  remaining_gerbils = 16 →
  initial_gerbils = sold_gerbils + remaining_gerbils →
  initial_gerbils = 85 := by
sorry

end NUMINAMATH_CALUDE_pet_store_gerbils_l2037_203709


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2037_203724

/-- The number of distinct intersection points of diagonals in the interior of a regular decagon -/
def diagonal_intersections (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  diagonal_intersections 10 = 210 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2037_203724


namespace NUMINAMATH_CALUDE_decimal_multiplication_division_l2037_203707

theorem decimal_multiplication_division : (0.5 * 0.6) / 0.2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_division_l2037_203707


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2037_203708

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a * b > 4) ∧
  (∃ a b, a * b > 4 ∧ ¬(a > 2 ∧ b > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2037_203708


namespace NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_l2037_203797

theorem smallest_angle_for_complete_circle : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ t → ∃ (r : ℝ), r = Real.sin θ) ∧
  (∀ (s : ℝ), s > 0 ∧ s < t → ¬(∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ s → ∃ (r : ℝ), r = Real.sin θ)) ∧
  t = π :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_l2037_203797


namespace NUMINAMATH_CALUDE_pushup_ratio_l2037_203721

theorem pushup_ratio : 
  ∀ (monday tuesday wednesday thursday friday : ℕ),
    monday = 5 →
    tuesday = 7 →
    wednesday = 2 * tuesday →
    friday = monday + tuesday + wednesday + thursday →
    friday = 39 →
    2 * thursday = monday + tuesday + wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_pushup_ratio_l2037_203721


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2037_203743

theorem rectangular_box_volume (l w h : ℝ) (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2037_203743


namespace NUMINAMATH_CALUDE_carol_peanuts_l2037_203793

/-- The total number of peanuts Carol has -/
def total_peanuts (tree_peanuts ground_peanuts bags bag_capacity : ℕ) : ℕ :=
  tree_peanuts + ground_peanuts + bags * bag_capacity

/-- Theorem: Carol has 976 peanuts in total -/
theorem carol_peanuts :
  total_peanuts 48 178 3 250 = 976 := by
  sorry

end NUMINAMATH_CALUDE_carol_peanuts_l2037_203793


namespace NUMINAMATH_CALUDE_cupcakes_theorem_l2037_203776

def cupcakes_problem (initial_cupcakes : ℕ) 
                     (delmont_class : ℕ) 
                     (donnelly_class : ℕ) 
                     (teachers : ℕ) 
                     (staff : ℕ) : Prop :=
  let given_away := delmont_class + donnelly_class + teachers + staff
  initial_cupcakes - given_away = 2

theorem cupcakes_theorem : 
  cupcakes_problem 40 18 16 2 2 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_theorem_l2037_203776


namespace NUMINAMATH_CALUDE_division_problem_l2037_203742

theorem division_problem (x y : ℤ) (k : ℕ) (h1 : x > 0) 
  (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = k * (3 * y) + 1) 
  (h4 : 7 * y - x = 3) : 
  k = 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2037_203742


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2037_203719

theorem sum_of_xyz (a b : ℝ) (x y z : ℕ+) : 
  a^2 = 9/14 ∧ 
  b^2 = (3 + Real.sqrt 7)^2 / 14 ∧ 
  a < 0 ∧ 
  b > 0 ∧ 
  (a + b)^3 = (x : ℝ) * Real.sqrt y / z →
  x + y + z = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2037_203719


namespace NUMINAMATH_CALUDE_plates_problem_l2037_203795

theorem plates_problem (x : ℚ) : 
  (1/3 * x - 2/3) - 1/2 * ((2/3 * x) - 4/3) = 9 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_plates_problem_l2037_203795


namespace NUMINAMATH_CALUDE_cloth_sold_l2037_203781

/-- Proves the number of meters of cloth sold by a shopkeeper -/
theorem cloth_sold (total_price : ℝ) (loss_per_meter : ℝ) (cost_price : ℝ) :
  total_price = 18000 ∧ loss_per_meter = 5 ∧ cost_price = 50 →
  (total_price / (cost_price - loss_per_meter) : ℝ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sold_l2037_203781


namespace NUMINAMATH_CALUDE_no_solution_l2037_203772

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem no_solution : ¬∃ x : ℕ, (137 + x = 435) ∧ (reverse_digits x = 672) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l2037_203772


namespace NUMINAMATH_CALUDE_alex_and_father_ages_l2037_203704

theorem alex_and_father_ages :
  ∀ (alex_age father_age : ℕ),
  (father_age = 2 * alex_age + 5) →
  (alex_age - 6 = alex_age / 3) →
  (alex_age = 9 ∧ father_age = 23) :=
by sorry

end NUMINAMATH_CALUDE_alex_and_father_ages_l2037_203704


namespace NUMINAMATH_CALUDE_diameter_of_circumscribing_circle_l2037_203755

/-- The diameter of a circle circumscribing six smaller tangent circles -/
theorem diameter_of_circumscribing_circle (r : ℝ) (h : r = 5) :
  let small_circle_radius : ℝ := r
  let small_circle_count : ℕ := 6
  let inner_hexagon_side : ℝ := 2 * small_circle_radius
  let inner_hexagon_radius : ℝ := inner_hexagon_side
  let large_circle_radius : ℝ := inner_hexagon_radius + small_circle_radius
  large_circle_radius * 2 = 30 := by sorry

end NUMINAMATH_CALUDE_diameter_of_circumscribing_circle_l2037_203755


namespace NUMINAMATH_CALUDE_no_prime_roots_sum_65_l2037_203791

theorem no_prime_roots_sum_65 : ¬∃ (p q k : ℕ), Prime p ∧ Prime q ∧ p + q = 65 ∧ p * q = k ∧ p^2 - 65*p + k = 0 ∧ q^2 - 65*q + k = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_sum_65_l2037_203791


namespace NUMINAMATH_CALUDE_train_journey_constant_speed_time_l2037_203779

/-- Represents the journey of a train with uniform acceleration, constant speed, and uniform deceleration phases. -/
structure TrainJourney where
  totalDistance : ℝ  -- in km
  totalTime : ℝ      -- in hours
  constantSpeed : ℝ  -- in km/h

/-- Calculates the time spent at constant speed during the journey. -/
def timeAtConstantSpeed (journey : TrainJourney) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the time at constant speed is 1/5 hours (12 minutes). -/
theorem train_journey_constant_speed_time 
  (journey : TrainJourney) 
  (h1 : journey.totalDistance = 21) 
  (h2 : journey.totalTime = 4/15)
  (h3 : journey.constantSpeed = 90) :
  timeAtConstantSpeed journey = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_constant_speed_time_l2037_203779


namespace NUMINAMATH_CALUDE_christmas_sale_pricing_l2037_203717

/-- Represents the discount rate as a fraction -/
def discount_rate : ℚ := 2/5

/-- Calculates the sale price given the original price and discount rate -/
def sale_price (original_price : ℚ) : ℚ := original_price * (1 - discount_rate)

/-- Calculates the original price given the sale price and discount rate -/
def original_price (sale_price : ℚ) : ℚ := sale_price / (1 - discount_rate)

theorem christmas_sale_pricing (a b : ℚ) :
  sale_price a = 3/5 * a ∧ original_price b = 5/3 * b := by
  sorry

end NUMINAMATH_CALUDE_christmas_sale_pricing_l2037_203717


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2037_203702

theorem reciprocal_of_negative_two :
  ∃ (x : ℚ), x * (-2) = 1 ∧ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2037_203702


namespace NUMINAMATH_CALUDE_investment_problem_l2037_203739

theorem investment_problem (amount_at_5_percent : ℝ) (total_with_interest : ℝ) :
  amount_at_5_percent = 600 →
  total_with_interest = 1054 →
  ∃ (total_investment : ℝ),
    total_investment = 1034 ∧
    amount_at_5_percent + amount_at_5_percent * 0.05 +
    (total_investment - amount_at_5_percent) +
    (total_investment - amount_at_5_percent) * 0.06 = total_with_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l2037_203739


namespace NUMINAMATH_CALUDE_power_seven_twelve_mod_hundred_l2037_203787

theorem power_seven_twelve_mod_hundred : 7^12 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_twelve_mod_hundred_l2037_203787


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2037_203789

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 → 
  (q + r) / 2 = 26 → 
  r - p = 32 → 
  (q + r) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2037_203789
