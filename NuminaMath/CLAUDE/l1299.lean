import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1299_129915

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 1 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1299_129915


namespace NUMINAMATH_CALUDE_piano_cost_solution_l1299_129960

def piano_cost_problem (total_lessons : ℕ) (lesson_cost : ℚ) (discount_percent : ℚ) (total_cost : ℚ) : Prop :=
  let original_lesson_cost := total_lessons * lesson_cost
  let discount_amount := discount_percent * original_lesson_cost
  let discounted_lesson_cost := original_lesson_cost - discount_amount
  let piano_cost := total_cost - discounted_lesson_cost
  piano_cost = 500

theorem piano_cost_solution :
  piano_cost_problem 20 40 0.25 1100 := by
  sorry

end NUMINAMATH_CALUDE_piano_cost_solution_l1299_129960


namespace NUMINAMATH_CALUDE_sum_of_tags_is_1000_l1299_129912

/-- The sum of tagged numbers on four cards W, X, Y, Z -/
def sum_of_tags (w x y z : ℕ) : ℕ := w + x + y + z

/-- Theorem stating the sum of tagged numbers is 1000 -/
theorem sum_of_tags_is_1000 :
  ∀ (w x y z : ℕ),
  w = 200 →
  x = w / 2 →
  y = x + w →
  z = 400 →
  sum_of_tags w x y z = 1000 := by
sorry

end NUMINAMATH_CALUDE_sum_of_tags_is_1000_l1299_129912


namespace NUMINAMATH_CALUDE_xyz_value_l1299_129996

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 3)
  (eq3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1299_129996


namespace NUMINAMATH_CALUDE_product_negative_implies_zero_l1299_129935

theorem product_negative_implies_zero (a b : ℝ) (h : a * b < 0) :
  a^2 * abs b - b^2 * abs a + a * b * (abs a - abs b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_negative_implies_zero_l1299_129935


namespace NUMINAMATH_CALUDE_x_thirteen_percent_greater_than_80_l1299_129924

theorem x_thirteen_percent_greater_than_80 :
  let x := 80 * (1 + 13 / 100)
  x = 90.4 := by sorry

end NUMINAMATH_CALUDE_x_thirteen_percent_greater_than_80_l1299_129924


namespace NUMINAMATH_CALUDE_square_grid_division_l1299_129998

theorem square_grid_division (m n k : ℕ) (h : m * m = n * k) :
  ∃ (d : ℕ), d ∣ m ∧ d ∣ n ∧ (m / d) * d = k ∧ (n / d) * d = n :=
sorry

end NUMINAMATH_CALUDE_square_grid_division_l1299_129998


namespace NUMINAMATH_CALUDE_sum_of_fractions_value_of_m_l1299_129989

noncomputable section

variable (θ : Real)
variable (m : Real)

-- Define the equation and its roots
def equation (x : Real) := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

-- Conditions
axiom theta_range : 0 < θ ∧ θ < 2 * Real.pi
axiom roots : equation (Real.sin θ) = 0 ∧ equation (Real.cos θ) = 0

-- Theorems to prove
theorem sum_of_fractions :
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem value_of_m : m = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_value_of_m_l1299_129989


namespace NUMINAMATH_CALUDE_omelet_distribution_l1299_129911

theorem omelet_distribution (total_eggs : ℕ) (eggs_per_omelet : ℕ) (num_people : ℕ) :
  total_eggs = 36 →
  eggs_per_omelet = 4 →
  num_people = 3 →
  (total_eggs / eggs_per_omelet) / num_people = 3 := by
sorry

end NUMINAMATH_CALUDE_omelet_distribution_l1299_129911


namespace NUMINAMATH_CALUDE_set_operations_l1299_129980

def A : Set ℤ := {1,2,3,4,5}
def B : Set ℤ := {-1,1,2,3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {1,2,3}) ∧
  (A ∪ B = {-1,1,2,3,4,5}) ∧
  ((U \ B) ∩ A = {4,5}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1299_129980


namespace NUMINAMATH_CALUDE_car_speed_ratio_l1299_129991

theorem car_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > 0 → v₂ > 0 →
  (3 * v₂ / v₁ - 3 * v₁ / v₂ = 1.1) →
  v₂ / v₁ = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_car_speed_ratio_l1299_129991


namespace NUMINAMATH_CALUDE_christmas_gifts_left_l1299_129972

/-- The number of gifts left under the Christmas tree -/
def gifts_left (initial : ℕ) (sent : ℕ) : ℕ := initial - sent

/-- Theorem stating that given 77 initial gifts and 66 sent gifts, 11 gifts are left -/
theorem christmas_gifts_left : gifts_left 77 66 = 11 := by
  sorry

end NUMINAMATH_CALUDE_christmas_gifts_left_l1299_129972


namespace NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l1299_129993

/-- A rectangle in 3D space -/
structure Rectangle3D where
  ab : ℝ
  bc : ℝ

/-- A line segment in 3D space -/
structure Segment3D where
  length : ℝ
  distance_from_plane : ℝ

/-- The volume of a polyhedron formed by a rectangle and a parallel segment -/
def polyhedron_volume (rect : Rectangle3D) (seg : Segment3D) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron ABCDKM -/
theorem volume_of_specific_polyhedron :
  let rect := Rectangle3D.mk 2 3
  let seg := Segment3D.mk 5 1
  polyhedron_volume rect seg = 9/2 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l1299_129993


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1299_129949

/-- Given a, b, s, t, u, v are real numbers satisfying the following conditions:
    - 0 < a < b
    - a, s, t, b form an arithmetic sequence
    - a, u, v, b form a geometric sequence
    Prove that s * t * (s + t) > u * v * (u + v)
-/
theorem arithmetic_geometric_inequality (a b s t u v : ℝ) 
  (h1 : 0 < a) (h2 : a < b)
  (h3 : s = (2*a + b)/3) (h4 : t = (a + 2*b)/3)  -- arithmetic sequence condition
  (h5 : u = (a^2 * b)^(1/3)) (h6 : v = (a * b^2)^(1/3))  -- geometric sequence condition
  : s * t * (s + t) > u * v * (u + v) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1299_129949


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1299_129962

/-- Given an arithmetic sequence with first term 2, last term 2014, and common difference 3,
    prove that it has 671 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 2 →                        -- First term is 2
    (∀ n, a (n + 1) = a n + 3) →     -- Common difference is 3
    (∃ k, a k = 2014) →              -- Last term is 2014
    (∃ k, a k = 2014 ∧ k + 1 = 671)  -- The sequence has 671 terms
    := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1299_129962


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1299_129925

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1299_129925


namespace NUMINAMATH_CALUDE_double_earnings_in_ten_days_l1299_129919

/-- Calculates the number of additional days needed to earn twice the current amount --/
def additional_days_to_double_earnings (days_worked : ℕ) (total_earned : ℚ) : ℕ :=
  let daily_rate := total_earned / days_worked
  let target_amount := 2 * total_earned
  let total_days_needed := (target_amount / daily_rate).ceil.toNat
  total_days_needed - days_worked

/-- Theorem stating that for the given conditions, 10 additional days are needed --/
theorem double_earnings_in_ten_days :
  additional_days_to_double_earnings 10 250 = 10 := by
  sorry

#eval additional_days_to_double_earnings 10 250

end NUMINAMATH_CALUDE_double_earnings_in_ten_days_l1299_129919


namespace NUMINAMATH_CALUDE_supermarket_spending_l1299_129992

theorem supermarket_spending (total : ℚ) : 
  (3/7 : ℚ) * total + (2/5 : ℚ) * total + (1/4 : ℚ) * total + (1/14 : ℚ) * total + 12 = total →
  total = 80 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1299_129992


namespace NUMINAMATH_CALUDE_rectangle_dimensions_area_l1299_129987

theorem rectangle_dimensions_area (x : ℝ) : 
  (2*x - 3 > 0) → 
  (3*x + 4 > 0) → 
  (2*x - 3) * (3*x + 4) = 14*x - 6 → 
  x = (5 + Real.sqrt 41) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_area_l1299_129987


namespace NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l1299_129950

/-- 
Given a tank with a capacity of 48 gallons, initially filled to 3/4 of its capacity,
prove that after adding 8 gallons of gasoline, the tank will be filled to 11/12 of its capacity.
-/
theorem tank_capacity_after_adding_gas (tank_capacity : ℚ) (initial_fill_fraction : ℚ) 
  (added_gas : ℚ) (final_fill_fraction : ℚ) : 
  tank_capacity = 48 → 
  initial_fill_fraction = 3/4 → 
  added_gas = 8 → 
  final_fill_fraction = (initial_fill_fraction * tank_capacity + added_gas) / tank_capacity →
  final_fill_fraction = 11/12 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l1299_129950


namespace NUMINAMATH_CALUDE_min_arcs_for_circle_l1299_129910

theorem min_arcs_for_circle (arc_measure : ℝ) (n : ℕ) : 
  arc_measure = 120 → 
  (n : ℝ) * arc_measure = 360 → 
  n ≥ 3 ∧ ∀ m : ℕ, m < n → (m : ℝ) * arc_measure ≠ 360 :=
by sorry

end NUMINAMATH_CALUDE_min_arcs_for_circle_l1299_129910


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1299_129939

/-- A quadratic equation x^2 - x + 2k = 0 has two equal real roots if and only if k = 1/8 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - x + 2*k = 0 ∧ (∀ y : ℝ, y^2 - y + 2*k = 0 → y = x)) ↔ k = 1/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1299_129939


namespace NUMINAMATH_CALUDE_ali_class_size_l1299_129956

/-- Calculates the total number of students in a class given a student's rank from top and bottom -/
def class_size (rank_from_top : ℕ) (rank_from_bottom : ℕ) : ℕ :=
  rank_from_top + rank_from_bottom - 1

/-- Theorem: In a class where a student ranks 40th from both the top and bottom, the total number of students is 79 -/
theorem ali_class_size :
  class_size 40 40 = 79 := by
  sorry

#eval class_size 40 40

end NUMINAMATH_CALUDE_ali_class_size_l1299_129956


namespace NUMINAMATH_CALUDE_race_probability_l1299_129985

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) 
  (h_total : total_cars = 12)
  (h_x : prob_x = 1 / 6)
  (h_y : prob_y = 1 / 10)
  (h_z : prob_z = 1 / 8)
  (h_no_tie : ∀ a b : ℕ, a ≠ b → a ≤ total_cars → b ≤ total_cars → 
    (∃ t : ℚ, t > 0 ∧ t < 1 ∧ prob_x + prob_y + prob_z + t = 1)) :
  prob_x + prob_y + prob_z = 47 / 120 :=
sorry

end NUMINAMATH_CALUDE_race_probability_l1299_129985


namespace NUMINAMATH_CALUDE_discount_order_difference_l1299_129958

-- Define the original price and discounts
def original_price : ℝ := 30
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.25

-- Define the two discount application orders
def discount_flat_then_percent : ℝ := (original_price - flat_discount) * (1 - percentage_discount)
def discount_percent_then_flat : ℝ := (original_price * (1 - percentage_discount)) - flat_discount

-- Theorem statement
theorem discount_order_difference :
  discount_flat_then_percent - discount_percent_then_flat = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_l1299_129958


namespace NUMINAMATH_CALUDE_swimmers_meet_problem_l1299_129964

/-- Represents the number of times two swimmers meet in a pool -/
def swimmers_meet (pool_length : ℝ) (speed_a speed_b : ℝ) (time : ℝ) : ℕ :=
  sorry

theorem swimmers_meet_problem :
  swimmers_meet 90 3 2 (12 * 60) = 20 := by sorry

end NUMINAMATH_CALUDE_swimmers_meet_problem_l1299_129964


namespace NUMINAMATH_CALUDE_exactly_one_correct_statement_l1299_129967

-- Define the type for geometric statements
inductive GeometricStatement
  | uniquePerpendicular
  | perpendicularIntersect
  | equalVertical
  | distanceDefinition
  | uniqueParallel

-- Function to check if a statement is correct
def isCorrect (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.perpendicularIntersect => True
  | _ => False

-- Theorem stating that exactly one statement is correct
theorem exactly_one_correct_statement :
  ∃! (s : GeometricStatement), isCorrect s :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_correct_statement_l1299_129967


namespace NUMINAMATH_CALUDE_angle_difference_l1299_129929

theorem angle_difference (α β : Real) 
  (h1 : 3 * Real.sin α - Real.cos α = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < α)
  (h4 : α < Real.pi / 2)
  (h5 : Real.pi / 2 < β)
  (h6 : β < Real.pi) :
  2 * α - β = -3 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_difference_l1299_129929


namespace NUMINAMATH_CALUDE_custom_operations_fraction_l1299_129944

-- Define the custom operations
def oplus (a b : ℝ) : ℝ := a * b + b^2
def otimes (a b : ℝ) : ℝ := a - b + a * b^2

-- State the theorem
theorem custom_operations_fraction :
  (oplus 8 3) / (otimes 8 3) = 33 / 77 := by sorry

end NUMINAMATH_CALUDE_custom_operations_fraction_l1299_129944


namespace NUMINAMATH_CALUDE_jiayuan_supermarket_fruit_weight_l1299_129940

theorem jiayuan_supermarket_fruit_weight :
  let apple_baskets : ℕ := 62
  let pear_baskets : ℕ := 38
  let weight_per_basket : ℕ := 25
  apple_baskets * weight_per_basket + pear_baskets * weight_per_basket = 2500 := by
  sorry

end NUMINAMATH_CALUDE_jiayuan_supermarket_fruit_weight_l1299_129940


namespace NUMINAMATH_CALUDE_fourth_root_of_506250000_l1299_129903

theorem fourth_root_of_506250000 : (506250000 : ℝ) ^ (1/4 : ℝ) = 150 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_506250000_l1299_129903


namespace NUMINAMATH_CALUDE_cos_neg_thirty_degrees_l1299_129941

theorem cos_neg_thirty_degrees : Real.cos (-(30 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_thirty_degrees_l1299_129941


namespace NUMINAMATH_CALUDE_train_passing_time_l1299_129938

/-- Time taken for trains to pass each other -/
theorem train_passing_time (man_speed goods_speed : ℝ) (goods_length : ℝ) : 
  man_speed = 80 →
  goods_speed = 32 →
  goods_length = 280 →
  let relative_speed := (man_speed + goods_speed) * 1000 / 3600
  let time := goods_length / relative_speed
  ∃ ε > 0, |time - 8.993| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l1299_129938


namespace NUMINAMATH_CALUDE_solution_in_interval_l1299_129979

def f (x : ℝ) := 4 * x^3 + x - 8

theorem solution_in_interval :
  (f 2 > 0) →
  (f 1.5 > 0) →
  (f 1 < 0) →
  ∃ x, x > 1 ∧ x < 1.5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1299_129979


namespace NUMINAMATH_CALUDE_sum_range_l1299_129901

theorem sum_range : 
  let sum := (25/8 : ℚ) + (31/7 : ℚ) + (128/21 : ℚ)
  (27/2 : ℚ) < sum ∧ sum < 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_l1299_129901


namespace NUMINAMATH_CALUDE_average_of_first_21_multiples_of_6_l1299_129973

/-- The average of the first n multiples of a number -/
def averageOfMultiples (n : ℕ) (x : ℕ) : ℚ :=
  (n * x * (n + 1)) / (2 * n)

/-- Theorem: The average of the first 21 multiples of 6 is 66 -/
theorem average_of_first_21_multiples_of_6 :
  averageOfMultiples 21 6 = 66 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_21_multiples_of_6_l1299_129973


namespace NUMINAMATH_CALUDE_cube_split_contains_31_l1299_129905

def split_cube (m : ℕ) : List ℕ :=
  let start := 2 * m * m - 2 * m + 1
  List.range m |>.map (fun i => start + 2 * i)

theorem cube_split_contains_31 (m : ℕ) (h1 : m > 1) :
  31 ∈ split_cube m → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_contains_31_l1299_129905


namespace NUMINAMATH_CALUDE_min_sequence_length_l1299_129961

def S : Finset Nat := {1, 2, 3, 4}

def ValidSequence (seq : List Nat) : Prop :=
  ∀ B : Finset Nat, B ⊆ S → B.Nonempty → 
    ∃ subseq : List Nat, subseq.length = B.card ∧ 
      subseq.toFinset = B ∧ seq.Sublist subseq

theorem min_sequence_length : 
  (∃ seq : List Nat, ValidSequence seq ∧ seq.length = 8) ∧
  (∀ seq : List Nat, ValidSequence seq → seq.length ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_min_sequence_length_l1299_129961


namespace NUMINAMATH_CALUDE_y_coordinate_of_C_l1299_129937

-- Define the pentagon ABCDE
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

-- Define the properties of the pentagon
def symmetricPentagon (p : Pentagon) : Prop :=
  p.A.1 = 0 ∧ p.A.2 = 0 ∧
  p.B.1 = 0 ∧ p.B.2 = 5 ∧
  p.D.1 = 5 ∧ p.D.2 = 5 ∧
  p.E.1 = 5 ∧ p.E.2 = 0 ∧
  p.C.1 = 2.5 -- Vertical line of symmetry

-- Define the area of the pentagon
def pentagonArea (p : Pentagon) : ℝ :=
  50 -- Given area

-- Theorem: The y-coordinate of vertex C is 15
theorem y_coordinate_of_C (p : Pentagon) 
  (h1 : symmetricPentagon p) 
  (h2 : pentagonArea p = 50) : 
  p.C.2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_of_C_l1299_129937


namespace NUMINAMATH_CALUDE_vikki_tax_percentage_l1299_129977

/-- Calculates the tax percentage given the working conditions and take-home pay --/
def calculate_tax_percentage (hours_worked : ℕ) (hourly_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) (take_home_pay : ℚ) : ℚ :=
  let gross_earnings := hours_worked * hourly_rate
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := gross_earnings - take_home_pay
  let tax_deduction := total_deductions - insurance_deduction - union_dues
  (tax_deduction / gross_earnings) * 100

/-- Theorem stating that the tax percentage is 20% given Vikki's working conditions --/
theorem vikki_tax_percentage :
  calculate_tax_percentage 42 10 (5/100) 5 310 = 20 := by
  sorry

end NUMINAMATH_CALUDE_vikki_tax_percentage_l1299_129977


namespace NUMINAMATH_CALUDE_symmetric_function_implies_a_eq_neg_four_l1299_129933

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

/-- Theorem: If f(2-x) = f(2+x) for all x, then a = -4 -/
theorem symmetric_function_implies_a_eq_neg_four (a : ℝ) :
  (∀ x, f a (2 - x) = f a (2 + x)) → a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_implies_a_eq_neg_four_l1299_129933


namespace NUMINAMATH_CALUDE_probability_specific_draw_l1299_129906

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards drawn -/
def CardsDrawn : ℕ := 4

/-- Represents the number of 4s in a standard deck -/
def FoursInDeck : ℕ := 4

/-- Represents the number of clubs in a standard deck -/
def ClubsInDeck : ℕ := 13

/-- Represents the number of 2s in a standard deck -/
def TwosInDeck : ℕ := 4

/-- Represents the number of hearts in a standard deck -/
def HeartsInDeck : ℕ := 13

/-- The probability of drawing a 4, then a club, then a 2, then a heart from a standard 52-card deck -/
theorem probability_specific_draw : 
  (FoursInDeck : ℚ) / StandardDeck *
  ClubsInDeck / (StandardDeck - 1) *
  TwosInDeck / (StandardDeck - 2) *
  HeartsInDeck / (StandardDeck - 3) = 4 / 10829 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_draw_l1299_129906


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1299_129970

theorem quadratic_roots_difference (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) → 
  a > b → 
  a - b = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1299_129970


namespace NUMINAMATH_CALUDE_pizzas_with_mushrooms_or_olives_l1299_129900

def num_toppings : ℕ := 8

-- Function to calculate combinations
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of pizzas with 1, 2, or 3 toppings
def total_pizzas : ℕ :=
  combinations num_toppings 1 + combinations num_toppings 2 + combinations num_toppings 3

-- Number of pizzas with mushrooms (or olives)
def pizzas_with_one_topping : ℕ :=
  1 + combinations (num_toppings - 1) 1 + combinations (num_toppings - 1) 2

-- Number of pizzas with both mushrooms and olives
def pizzas_with_both : ℕ :=
  1 + combinations (num_toppings - 2) 1 + combinations (num_toppings - 2) 2

-- Main theorem
theorem pizzas_with_mushrooms_or_olives :
  pizzas_with_one_topping * 2 - pizzas_with_both = 86 :=
sorry

end NUMINAMATH_CALUDE_pizzas_with_mushrooms_or_olives_l1299_129900


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l1299_129920

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l1299_129920


namespace NUMINAMATH_CALUDE_average_of_four_data_points_l1299_129974

theorem average_of_four_data_points
  (n : ℕ)
  (total_average : ℚ)
  (one_data_point : ℚ)
  (h1 : n = 5)
  (h2 : total_average = 81)
  (h3 : one_data_point = 85) :
  (n : ℚ) * total_average - one_data_point = (n - 1 : ℚ) * 80 :=
by sorry

end NUMINAMATH_CALUDE_average_of_four_data_points_l1299_129974


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1299_129966

theorem inequality_and_equality_condition (x y z : ℝ) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 ∧
  (x^2 + y^4 + z^6 = x*y^2 + y^2*z^3 + x*z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1299_129966


namespace NUMINAMATH_CALUDE_range_of_x_l1299_129968

theorem range_of_x (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) →
  x ∈ Set.Icc (Real.pi / 4) (7 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1299_129968


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l1299_129952

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := (Finset.range n).sum sumOfDigits

theorem sum_of_digits_up_to_1000 : sumOfDigitsUpTo 1000 = 14446 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l1299_129952


namespace NUMINAMATH_CALUDE_coins_problem_l1299_129983

theorem coins_problem (x : ℚ) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (4 : ℚ) / 5 * lost
  let remaining := x - lost + found
  x - remaining = (2 : ℚ) / 15 * x :=
by sorry

end NUMINAMATH_CALUDE_coins_problem_l1299_129983


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_doubled_l1299_129902

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The theorem stating that the number of diagonals in a dodecagon, when doubled, is 108 -/
theorem dodecagon_diagonals_doubled :
  2 * (num_diagonals dodecagon_sides) = 108 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_doubled_l1299_129902


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l1299_129963

theorem parabola_intersection_difference : ∃ (a b c d : ℝ),
  (3 * a^2 - 6 * a + 5 = -2 * a^2 - 3 * a + 7) ∧
  (3 * c^2 - 6 * c + 5 = -2 * c^2 - 3 * c + 7) ∧
  c ≥ a ∧
  c - a = 7/5 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l1299_129963


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1299_129909

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (2 * x + 14) = 10 → x = 43 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1299_129909


namespace NUMINAMATH_CALUDE_quadratic_solution_l1299_129954

theorem quadratic_solution (b : ℝ) : 
  ((-5 : ℝ)^2 + b * (-5) - 45 = 0) → b = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1299_129954


namespace NUMINAMATH_CALUDE_number_sequence_problem_l1299_129981

theorem number_sequence_problem :
  ∃ k : ℕ+,
    let a : ℕ+ → ℤ := λ n => (-2) ^ n.val
    let b : ℕ+ → ℤ := λ n => a n + 2
    let c : ℕ+ → ℚ := λ n => (1 / 2 : ℚ) * (a n)
    (a k + b k + c k = 642) ∧ (a k = 256) :=
by
  sorry

end NUMINAMATH_CALUDE_number_sequence_problem_l1299_129981


namespace NUMINAMATH_CALUDE_final_result_l1299_129914

def chosen_number : ℕ := 122
def multiplier : ℕ := 2
def subtractor : ℕ := 138

theorem final_result :
  chosen_number * multiplier - subtractor = 106 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l1299_129914


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1299_129994

theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_R : R > 0) :
  (P * (R + 2) * 5) / 100 - (P * R * 5) / 100 = 250 →
  P = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1299_129994


namespace NUMINAMATH_CALUDE_a_range_is_open_2_5_l1299_129930

-- Define the sequence a_n
def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then (5 - a) * n - 11 else a ^ (n - 4)

-- Theorem statement
theorem a_range_is_open_2_5 :
  ∀ a : ℝ, (∀ n : ℕ, a_n a n < a_n a (n + 1)) →
  (2 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_a_range_is_open_2_5_l1299_129930


namespace NUMINAMATH_CALUDE_quadratic_roots_implies_d_l1299_129943

theorem quadratic_roots_implies_d (d : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + d = 0 ↔ x = (-8 + Real.sqrt 12) / 4 ∨ x = (-8 - Real.sqrt 12) / 4) → 
  d = 6.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_implies_d_l1299_129943


namespace NUMINAMATH_CALUDE_logo_enlargement_l1299_129907

/-- Calculates the height of a proportionally enlarged logo --/
def enlarged_logo_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Proves that a 3x2 inch logo enlarged to 12 inches wide will be 8 inches tall --/
theorem logo_enlargement :
  enlarged_logo_height 3 2 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_logo_enlargement_l1299_129907


namespace NUMINAMATH_CALUDE_angel_envelopes_l1299_129984

/-- The number of large envelopes Angel used --/
def large_envelopes : ℕ := 11

/-- The number of medium envelopes Angel used --/
def medium_envelopes : ℕ := 2 * large_envelopes

/-- The number of letters in small envelopes --/
def small_letters : ℕ := 20

/-- The number of letters in each medium envelope --/
def letters_per_medium : ℕ := 3

/-- The number of letters in each large envelope --/
def letters_per_large : ℕ := 5

/-- The total number of letters --/
def total_letters : ℕ := 150

theorem angel_envelopes :
  small_letters +
  medium_envelopes * letters_per_medium +
  large_envelopes * letters_per_large = total_letters :=
by sorry

end NUMINAMATH_CALUDE_angel_envelopes_l1299_129984


namespace NUMINAMATH_CALUDE_law_of_sines_iff_equilateral_l1299_129921

/-- In a triangle ABC, the law of sines condition is equivalent to the triangle being equilateral -/
theorem law_of_sines_iff_equilateral (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin B = b / Real.sin C ∧ b / Real.sin C = c / Real.sin A) ↔
  (a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_law_of_sines_iff_equilateral_l1299_129921


namespace NUMINAMATH_CALUDE_total_price_is_23_l1299_129936

-- Define the price of cucumbers per kilogram
def cucumber_price : ℝ := 5

-- Define the price of tomatoes as 20% cheaper than cucumbers
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

-- Define the quantity of tomatoes and cucumbers
def tomato_quantity : ℝ := 2
def cucumber_quantity : ℝ := 3

-- Theorem statement
theorem total_price_is_23 :
  tomato_quantity * tomato_price + cucumber_quantity * cucumber_price = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_price_is_23_l1299_129936


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1299_129946

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 * I + 1) / (1 - I) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1299_129946


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1299_129928

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1299_129928


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_minus_one_l1299_129965

theorem gcd_of_powers_of_two_minus_one :
  Nat.gcd (2^2100 - 1) (2^1950 - 1) = 2^150 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_minus_one_l1299_129965


namespace NUMINAMATH_CALUDE_nth_power_divisors_l1299_129917

theorem nth_power_divisors (n : ℕ+) : 
  (∃ (d : ℕ), d = (Finset.card (Nat.divisors (n^n.val)))) → 
  d = 861 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_nth_power_divisors_l1299_129917


namespace NUMINAMATH_CALUDE_intersection_point_a_l1299_129957

/-- A linear function f(x) = 4x + b -/
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

/-- The inverse of f -/
noncomputable def f_inv (b : ℤ) : ℝ → ℝ := λ x ↦ (x - b) / 4

theorem intersection_point_a (b : ℤ) (a : ℤ) :
  f b 4 = a ∧ f_inv b a = 4 → a = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_a_l1299_129957


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l1299_129986

theorem geometric_arithmetic_sequence_ratio 
  (x y z : ℝ) 
  (h_geometric : ∃ q : ℝ, y = x * q ∧ z = y * q) 
  (h_arithmetic : ∃ d : ℝ, y + z = (x + y) + d ∧ z + x = (y + z) + d) :
  ∃ q : ℝ, (y = x * q ∧ z = y * q) ∧ (q = -2 ∨ q = 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l1299_129986


namespace NUMINAMATH_CALUDE_watch_cost_price_l1299_129948

theorem watch_cost_price (loss_price gain_price : ℝ) : 
  loss_price = 0.9 * 1500 →
  gain_price = 1.04 * 1500 →
  gain_price - loss_price = 210 →
  1500 = (210 : ℝ) / 0.14 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1299_129948


namespace NUMINAMATH_CALUDE_negation_existence_cube_plus_one_l1299_129971

theorem negation_existence_cube_plus_one (x : ℝ) :
  (¬ ∃ x, x^3 + 1 = 0) ↔ ∀ x, x^3 + 1 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_cube_plus_one_l1299_129971


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l1299_129904

theorem square_minus_product_plus_square (a b : ℝ) 
  (h1 : a + b = 10) (h2 : a * b = 11) : 
  a^2 - a*b + b^2 = 67 := by sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l1299_129904


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1299_129982

theorem quadratic_transformation (p q r : ℝ) :
  (∀ x, p * x^2 + q * x + r = 7 * (x - 5)^2 + 14) →
  ∃ m k, ∀ x, 5 * p * x^2 + 5 * q * x + 5 * r = m * (x - 5)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1299_129982


namespace NUMINAMATH_CALUDE_inequality_proof_l1299_129990

theorem inequality_proof (a m n p : ℝ) 
  (h1 : a * Real.log a = 1)
  (h2 : m = Real.exp (1/2 + a))
  (h3 : Real.exp n = 3^a)
  (h4 : a^p = 2^Real.exp 1) : 
  n < p ∧ p < m := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1299_129990


namespace NUMINAMATH_CALUDE_magnitude_of_3_minus_i_l1299_129947

/-- Given a complex number z = 3 - i, prove that its magnitude |z| is equal to √10 -/
theorem magnitude_of_3_minus_i :
  let z : ℂ := 3 - I
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_minus_i_l1299_129947


namespace NUMINAMATH_CALUDE_magic_star_sum_l1299_129932

/-- Represents a 6th-order magic star -/
structure MagicStar :=
  (numbers : Finset ℕ)
  (lines : Finset (Finset ℕ))
  (h_numbers : numbers = Finset.range 12)
  (h_lines_count : lines.card = 6)
  (h_line_size : ∀ l ∈ lines, l.card = 4)
  (h_numbers_in_lines : ∀ n ∈ numbers, (lines.filter (λ l => n ∈ l)).card = 2)
  (h_line_sum_equal : ∃ s, ∀ l ∈ lines, l.sum id = s)

/-- The magic sum of a 6th-order magic star is 26 -/
theorem magic_star_sum (ms : MagicStar) : 
  ∃ (s : ℕ), (∀ l ∈ ms.lines, l.sum id = s) ∧ s = 26 := by
  sorry

end NUMINAMATH_CALUDE_magic_star_sum_l1299_129932


namespace NUMINAMATH_CALUDE_fourth_power_inequality_l1299_129951

theorem fourth_power_inequality (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_inequality_l1299_129951


namespace NUMINAMATH_CALUDE_darius_score_l1299_129926

/-- Represents the scores of Darius, Matt, and Marius in a table football game. -/
structure TableFootballScores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the table football game. -/
def game_conditions (scores : TableFootballScores) : Prop :=
  scores.marius = scores.darius + 3 ∧
  scores.matt = scores.darius + 5 ∧
  scores.darius + scores.matt + scores.marius = 38

/-- Theorem stating that under the given conditions, Darius scored 10 points. -/
theorem darius_score (scores : TableFootballScores) 
  (h : game_conditions scores) : scores.darius = 10 := by
  sorry

end NUMINAMATH_CALUDE_darius_score_l1299_129926


namespace NUMINAMATH_CALUDE_largest_c_value_l1299_129923

/-- The function f(x) = x^2 - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- 2 is in the range of f -/
def two_in_range (c : ℝ) : Prop := ∃ x, f c x = 2

theorem largest_c_value :
  (∃ c_max : ℝ, two_in_range c_max ∧ ∀ c > c_max, ¬(two_in_range c)) ∧
  (∀ c_max : ℝ, (two_in_range c_max ∧ ∀ c > c_max, ¬(two_in_range c)) → c_max = 11) :=
sorry

end NUMINAMATH_CALUDE_largest_c_value_l1299_129923


namespace NUMINAMATH_CALUDE_inequality_proof_l1299_129999

theorem inequality_proof (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : 0 ≤ x ∧ x ≤ π / 2) :
  a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1299_129999


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1299_129976

/-- Given two perpendicular lines with direction vectors (4, -5) and (a, 2), prove that a = 5/2 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -5]
  let v2 : Fin 2 → ℝ := ![a, 2]
  (∀ i : Fin 2, (v1 i) * (v2 i) = 0) → a = 5/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1299_129976


namespace NUMINAMATH_CALUDE_marcus_pies_l1299_129942

/-- The number of pies Marcus can fit in his oven at once -/
def oven_capacity : ℕ := 5

/-- The number of pies Marcus dropped -/
def dropped_pies : ℕ := 8

/-- The number of pies Marcus has left -/
def remaining_pies : ℕ := 27

/-- The number of batches Marcus baked -/
def batches : ℕ := 7

theorem marcus_pies :
  oven_capacity * batches = remaining_pies + dropped_pies :=
sorry

end NUMINAMATH_CALUDE_marcus_pies_l1299_129942


namespace NUMINAMATH_CALUDE_pencils_left_l1299_129955

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took out -/
def pencils_taken : ℕ := 22

/-- The number of pencils Dan returned -/
def pencils_returned : ℕ := 5

/-- Theorem: The number of pencils left in the drawer is 17 -/
theorem pencils_left : initial_pencils - (pencils_taken - pencils_returned) = 17 := by
  sorry

#eval initial_pencils - (pencils_taken - pencils_returned)

end NUMINAMATH_CALUDE_pencils_left_l1299_129955


namespace NUMINAMATH_CALUDE_average_correction_problem_l1299_129913

theorem average_correction_problem (initial_avg : ℚ) (misread : ℚ) (correct : ℚ) (correct_avg : ℚ) :
  initial_avg = 14 →
  misread = 26 →
  correct = 36 →
  correct_avg = 15 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg - misread + correct = (n : ℚ) * correct_avg ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_correction_problem_l1299_129913


namespace NUMINAMATH_CALUDE_jack_driving_years_l1299_129978

/-- Represents the number of miles Jack drives in four months -/
def miles_per_four_months : ℕ := 37000

/-- Represents the total number of miles Jack has driven -/
def total_miles_driven : ℕ := 999000

/-- Calculates the number of years Jack has been driving -/
def years_driving : ℚ :=
  total_miles_driven / (miles_per_four_months * 3)

/-- Theorem stating that Jack has been driving for 9 years -/
theorem jack_driving_years :
  years_driving = 9 := by sorry

end NUMINAMATH_CALUDE_jack_driving_years_l1299_129978


namespace NUMINAMATH_CALUDE_factor_expression_l1299_129922

theorem factor_expression (x : ℝ) : 75 * x^2 + 50 * x = 25 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1299_129922


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l1299_129953

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x + 3/2)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l1299_129953


namespace NUMINAMATH_CALUDE_mets_fans_count_l1299_129997

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def baseball_town (fans : FanCounts) : Prop :=
  fans.yankees * 2 = fans.mets * 3 ∧
  fans.mets * 5 = fans.redsox * 4 ∧
  fans.yankees + fans.mets + fans.redsox = 330

/-- The theorem to prove -/
theorem mets_fans_count (fans : FanCounts) :
  baseball_town fans → fans.mets = 88 := by
  sorry


end NUMINAMATH_CALUDE_mets_fans_count_l1299_129997


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_l1299_129934

theorem parallel_lines_intersection (n : ℕ) : 
  (10 - 1) * (n - 1) = 1260 → n = 141 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_l1299_129934


namespace NUMINAMATH_CALUDE_larger_part_of_66_l1299_129988

theorem larger_part_of_66 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_sum : x + y = 66) (h_relation : 0.40 * x = 0.625 * y + 10) : 
  max x y = 50 := by
sorry

end NUMINAMATH_CALUDE_larger_part_of_66_l1299_129988


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1299_129918

/-- Given a rectangle with width x, length 4x, and area 120 square inches,
    prove that the width is √30 inches and the length is 4√30 inches. -/
theorem rectangle_dimensions (x : ℝ) (h1 : x > 0) (h2 : x * (4 * x) = 120) :
  x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 := by
  sorry

#check rectangle_dimensions

end NUMINAMATH_CALUDE_rectangle_dimensions_l1299_129918


namespace NUMINAMATH_CALUDE_calculate_expression_l1299_129959

theorem calculate_expression : 2023^0 + (-1/3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1299_129959


namespace NUMINAMATH_CALUDE_santa_gifts_l1299_129916

theorem santa_gifts (x : ℕ) (h1 : x < 100) (h2 : x % 2 = 0) (h3 : x % 5 = 0) (h4 : x % 7 = 0) :
  x - (x / 2 + x / 5 + x / 7) = 11 :=
by sorry

end NUMINAMATH_CALUDE_santa_gifts_l1299_129916


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l1299_129931

/-- Proves that it takes 2 years for a man's age to be twice his son's age -/
theorem mans_age_twice_sons (
  son_age : ℕ) 
  (man_age : ℕ) 
  (h1 : son_age = 20) 
  (h2 : man_age = son_age + 22) : 
  ∃ y : ℕ, y = 2 ∧ man_age + y = 2 * (son_age + y) :=
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l1299_129931


namespace NUMINAMATH_CALUDE_f_minimum_value_l1299_129969

def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

theorem f_minimum_value : ∀ x : ℕ+, f x ≥ 23/2 := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1299_129969


namespace NUMINAMATH_CALUDE_moose_population_canada_l1299_129975

theorem moose_population_canada :
  ∀ (moose beaver human : ℕ),
    beaver = 2 * moose →
    human = 19 * beaver →
    human = 38000000 →
    moose = 1000000 :=
by
  sorry

end NUMINAMATH_CALUDE_moose_population_canada_l1299_129975


namespace NUMINAMATH_CALUDE_cookie_pattern_holds_l1299_129927

/-- Represents the number of cookies on each plate -/
def cookie_sequence : Fin 6 → ℕ
  | 0 => 5
  | 1 => 7
  | 2 => 10
  | 3 => 14
  | 4 => 19
  | 5 => 25

/-- The difference between consecutive cookie counts increases by 1 each time -/
def increasing_difference (seq : Fin 6 → ℕ) : Prop :=
  ∀ i : Fin 4, seq (i + 1) - seq i = seq (i + 2) - seq (i + 1) + 1

theorem cookie_pattern_holds :
  increasing_difference cookie_sequence ∧ cookie_sequence 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookie_pattern_holds_l1299_129927


namespace NUMINAMATH_CALUDE_solution_product_l1299_129945

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 11) = r^2 - 14 * r + 48 →
  (s - 3) * (3 * s + 11) = s^2 - 14 * s + 48 →
  r ≠ s →
  (r + 4) * (s + 4) = -226 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l1299_129945


namespace NUMINAMATH_CALUDE_min_value_sine_l1299_129995

/-- Given that f(x) = 3sin(x) - cos(x) attains its minimum value when x = θ, prove that sin(θ) = -3√10/10 -/
theorem min_value_sine (θ : ℝ) (h : ∀ x, 3 * Real.sin x - Real.cos x ≥ 3 * Real.sin θ - Real.cos θ) : 
  Real.sin θ = -3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sine_l1299_129995


namespace NUMINAMATH_CALUDE_min_value_z_l1299_129908

theorem min_value_z (x y : ℝ) (h1 : y ≥ x + 2) (h2 : x + y ≤ 6) (h3 : x ≥ 1) :
  ∃ (z : ℝ), z = 2 * |x - 2| + |y| ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = 2 * |x - 2| + |y| → w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l1299_129908
