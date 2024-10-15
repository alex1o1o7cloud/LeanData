import Mathlib

namespace NUMINAMATH_CALUDE_perfume_price_with_tax_l1521_152180

/-- Calculates the total price including tax given the original price and tax rate. -/
def totalPriceWithTax (originalPrice taxRate : ℝ) : ℝ :=
  originalPrice * (1 + taxRate)

/-- Theorem stating that for a product with an original price of $92 and a tax rate of 7.5%,
    the total price including tax is $98.90. -/
theorem perfume_price_with_tax :
  totalPriceWithTax 92 0.075 = 98.90 := by
  sorry

end NUMINAMATH_CALUDE_perfume_price_with_tax_l1521_152180


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_polynomial_l1521_152112

-- Problem 1
theorem factorize_difference_of_squares (x y : ℝ) :
  4 * x^2 - 25 * y^2 = (2*x + 5*y) * (2*x - 5*y) := by
  sorry

-- Problem 2
theorem factorize_polynomial (x y : ℝ) :
  -3 * x * y^3 + 27 * x^3 * y = -3 * x * y * (y + 3*x) * (y - 3*x) := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_polynomial_l1521_152112


namespace NUMINAMATH_CALUDE_octagon_angle_property_l1521_152142

theorem octagon_angle_property (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_octagon_angle_property_l1521_152142


namespace NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l1521_152181

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The given conditions of the problem -/
def fan_problem (fc : FanCounts) : Prop :=
  fc.yankees * 2 = fc.mets * 3 ∧  -- Ratio of Yankees to Mets is 3:2
  fc.yankees + fc.mets + fc.red_sox = 330 ∧  -- Total fans
  fc.mets = 88  -- Number of Mets fans

/-- The theorem to prove -/
theorem mets_to_red_sox_ratio 
  (fc : FanCounts) 
  (h : fan_problem fc) : 
  ∃ (r : Ratio), r.numerator = 4 ∧ r.denominator = 5 ∧
  r.numerator * fc.red_sox = r.denominator * fc.mets :=
sorry

end NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l1521_152181


namespace NUMINAMATH_CALUDE_set_inclusion_implies_upper_bound_l1521_152156

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Define the complement of B in ℝ
def C_R_B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- Theorem statement
theorem set_inclusion_implies_upper_bound (a : ℝ) :
  A ⊆ C_R_B a → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_upper_bound_l1521_152156


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l1521_152128

theorem gcd_lcm_problem (a b : ℕ+) : 
  Nat.gcd a b = 21 ∧ Nat.lcm a b = 3969 → 
  (a = 21 ∧ b = 3969) ∨ (a = 147 ∧ b = 567) ∨ (a = 3969 ∧ b = 21) ∨ (a = 567 ∧ b = 147) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l1521_152128


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l1521_152176

theorem point_in_third_quadrant :
  let angle : ℝ := 2007 * Real.pi / 180
  (Real.cos angle < 0) ∧ (Real.sin angle < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l1521_152176


namespace NUMINAMATH_CALUDE_brick_laying_time_l1521_152113

/-- Given that 2b men can lay 3f bricks in c days, prove that 4c men will take b^2 / f days to lay 6b bricks, assuming constant working rate. -/
theorem brick_laying_time 
  (b f c : ℝ) 
  (h : b > 0 ∧ f > 0 ∧ c > 0) 
  (rate : ℝ := (3 * f) / (2 * b * c)) : 
  (6 * b) / (4 * c * rate) = b^2 / f := by
sorry

end NUMINAMATH_CALUDE_brick_laying_time_l1521_152113


namespace NUMINAMATH_CALUDE_prob_same_club_is_one_third_l1521_152189

/-- The number of clubs -/
def num_clubs : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of two students joining the same club given equal probability of joining any club -/
def prob_same_club : ℚ := 1 / 3

/-- Theorem stating that the probability of two students joining the same club is 1/3 -/
theorem prob_same_club_is_one_third :
  prob_same_club = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_same_club_is_one_third_l1521_152189


namespace NUMINAMATH_CALUDE_crank_slider_motion_l1521_152168

/-- Crank-slider mechanism parameters -/
structure CrankSlider where
  OA : ℝ
  AB : ℝ
  ω : ℝ
  AM : ℝ

/-- Position and velocity of point M -/
structure PointM where
  x : ℝ → ℝ
  y : ℝ → ℝ
  vx : ℝ → ℝ
  vy : ℝ → ℝ

/-- Theorem stating the equations of motion for point M -/
theorem crank_slider_motion (cs : CrankSlider) (t : ℝ) : 
  cs.OA = 90 ∧ cs.AB = 90 ∧ cs.ω = 10 ∧ cs.AM = 60 →
  ∃ (pm : PointM),
    pm.x t = 90 * Real.cos (10 * t) - 60 * Real.sin (10 * t) ∧
    pm.y t = 90 * Real.sin (10 * t) - 60 * Real.cos (10 * t) ∧
    pm.vx t = -900 * Real.sin (10 * t) - 600 * Real.cos (10 * t) ∧
    pm.vy t = 900 * Real.cos (10 * t) + 600 * Real.sin (10 * t) := by
  sorry


end NUMINAMATH_CALUDE_crank_slider_motion_l1521_152168


namespace NUMINAMATH_CALUDE_distribution_of_X_l1521_152117

/-- A discrete random variable with three possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  p₃ : ℝ
  x₁_lt_x₂ : x₁ < x₂
  x₂_lt_x₃ : x₂ < x₃
  prob_sum : p₁ + p₂ + p₃ = 1
  prob_nonneg : 0 ≤ p₁ ∧ 0 ≤ p₂ ∧ 0 ≤ p₃

/-- Expected value of a discrete random variable -/
def expectedValue (X : DiscreteRV) : ℝ :=
  X.x₁ * X.p₁ + X.x₂ * X.p₂ + X.x₃ * X.p₃

/-- Variance of a discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.x₁^2 * X.p₁ + X.x₂^2 * X.p₂ + X.x₃^2 * X.p₃ - (expectedValue X)^2

/-- Theorem stating the distribution of the random variable X -/
theorem distribution_of_X (X : DiscreteRV) 
  (h₁ : X.x₁ = 1)
  (h₂ : X.p₁ = 0.3)
  (h₃ : X.p₂ = 0.2)
  (h₄ : expectedValue X = 2.2)
  (h₅ : variance X = 0.76) :
  X.x₂ = 2 ∧ X.x₃ = 3 ∧ X.p₃ = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_distribution_of_X_l1521_152117


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1521_152124

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1521_152124


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1521_152182

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -4; 3, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-16, -6; 7, 2]
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![5, -7; -2, 3]
  M * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1521_152182


namespace NUMINAMATH_CALUDE_quadratic_decomposition_l1521_152161

theorem quadratic_decomposition :
  ∃ (k : ℤ) (a : ℝ), ∀ y : ℝ, y^2 + 14*y + 60 = (y + a)^2 + k ∧ k = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_decomposition_l1521_152161


namespace NUMINAMATH_CALUDE_triangle_area_l1521_152135

/-- A triangle with side lengths 6, 8, and 10 has an area of 24 square units. -/
theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1521_152135


namespace NUMINAMATH_CALUDE_matrix_cube_computation_l1521_152148

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; 2, 0]

theorem matrix_cube_computation :
  A ^ 3 = !![(-8), 0; 0, (-8)] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_computation_l1521_152148


namespace NUMINAMATH_CALUDE_no_solution_for_system_l1521_152121

theorem no_solution_for_system : ¬∃ x : ℝ, 
  (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l1521_152121


namespace NUMINAMATH_CALUDE_husband_additional_payment_l1521_152171

def medical_procedure_1 : ℚ := 128
def medical_procedure_2 : ℚ := 256
def medical_procedure_3 : ℚ := 64
def house_help_salary : ℚ := 160
def tax_rate : ℚ := 0.05

def total_medical_expenses : ℚ := medical_procedure_1 + medical_procedure_2 + medical_procedure_3
def couple_medical_contribution : ℚ := total_medical_expenses / 2
def house_help_medical_contribution : ℚ := min (total_medical_expenses / 2) house_help_salary
def tax_deduction : ℚ := house_help_salary * tax_rate
def total_couple_expense : ℚ := couple_medical_contribution + (total_medical_expenses / 2 - house_help_medical_contribution) + tax_deduction
def husband_paid : ℚ := couple_medical_contribution

theorem husband_additional_payment (
  split_equally : total_couple_expense / 2 < husband_paid
) : husband_paid - total_couple_expense / 2 = 76 := by sorry

end NUMINAMATH_CALUDE_husband_additional_payment_l1521_152171


namespace NUMINAMATH_CALUDE_system_solution_l1521_152134

theorem system_solution (x y z : ℝ) : 
  x * y = 8 - x - 4 * y →
  y * z = 12 - 3 * y - 6 * z →
  x * z = 40 - 5 * x - 2 * z →
  x > 0 →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1521_152134


namespace NUMINAMATH_CALUDE_average_and_differences_l1521_152122

theorem average_and_differences (y : ℝ) : 
  (50 + y) / 2 = 60 →
  y = 70 ∧ 
  |50 - y| = 20 ∧ 
  50 - y = -20 := by
sorry

end NUMINAMATH_CALUDE_average_and_differences_l1521_152122


namespace NUMINAMATH_CALUDE_golf_strokes_over_par_l1521_152172

/-- Given a golfer who plays 9 rounds with an average of 4 strokes per hole,
    and a par value of 3 per hole, prove that the golfer will be 9 strokes over par. -/
theorem golf_strokes_over_par (rounds : ℕ) (avg_strokes_per_hole : ℕ) (par_value_per_hole : ℕ)
  (h1 : rounds = 9)
  (h2 : avg_strokes_per_hole = 4)
  (h3 : par_value_per_hole = 3) :
  rounds * avg_strokes_per_hole - rounds * par_value_per_hole = 9 :=
by sorry

end NUMINAMATH_CALUDE_golf_strokes_over_par_l1521_152172


namespace NUMINAMATH_CALUDE_f_range_l1521_152195

noncomputable def f (x : ℝ) : ℝ := x^2 / (Real.log x + x)

noncomputable def g (x : ℝ) : ℝ := Real.log x + x

theorem f_range :
  ∃ (a : ℝ), 0 < a ∧ a < 1 ∧ g a = 0 →
  Set.range f = {y | y < 0 ∨ y ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_f_range_l1521_152195


namespace NUMINAMATH_CALUDE_equal_probability_red_black_l1521_152162

/-- Represents a deck of cards after removing face cards and 8's --/
structure Deck :=
  (total_cards : ℕ)
  (red_divisible_by_3 : ℕ)
  (black_divisible_by_3 : ℕ)

/-- Represents the probability of picking a card of a certain color divisible by 3 --/
def probability_divisible_by_3 (deck : Deck) (color : String) : ℚ :=
  if color = "red" then
    (deck.red_divisible_by_3 : ℚ) / deck.total_cards
  else if color = "black" then
    (deck.black_divisible_by_3 : ℚ) / deck.total_cards
  else
    0

/-- The main theorem stating that the probabilities are equal for red and black cards --/
theorem equal_probability_red_black (deck : Deck) 
    (h1 : deck.total_cards = 36)
    (h2 : deck.red_divisible_by_3 = 6)
    (h3 : deck.black_divisible_by_3 = 6) :
  probability_divisible_by_3 deck "red" = probability_divisible_by_3 deck "black" :=
by
  sorry

#check equal_probability_red_black

end NUMINAMATH_CALUDE_equal_probability_red_black_l1521_152162


namespace NUMINAMATH_CALUDE_pants_bought_l1521_152120

def total_cost : ℕ := 1500
def tshirt_cost : ℕ := 100
def pants_cost : ℕ := 250
def num_tshirts : ℕ := 5

theorem pants_bought :
  (total_cost - num_tshirts * tshirt_cost) / pants_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_pants_bought_l1521_152120


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l1521_152104

theorem sum_of_roots_cubic (x : ℝ) : 
  (∃ s : ℝ, (∀ x, x^3 - x^2 - 13*x + 13 = 0 → (∃ y z : ℝ, y ≠ x ∧ z ≠ x ∧ z ≠ y ∧ 
    x + y + z = s))) → 
  (∃ s : ℝ, (∀ x, x^3 - x^2 - 13*x + 13 = 0 → (∃ y z : ℝ, y ≠ x ∧ z ≠ x ∧ z ≠ y ∧ 
    x + y + z = s)) ∧ s = 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l1521_152104


namespace NUMINAMATH_CALUDE_quadrilateral_angle_difference_l1521_152153

/-- A quadrilateral with angles in ratio 3:4:5:6 has a difference of 60° between its largest and smallest angles -/
theorem quadrilateral_angle_difference (a b c d : ℝ) : 
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k ∧ d = 6*k →  -- Angles in ratio 3:4:5:6
  (6*k) - (3*k) = 60 :=  -- Difference between largest and smallest angles
by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_difference_l1521_152153


namespace NUMINAMATH_CALUDE_prob_select_one_from_two_out_of_four_prob_select_one_from_two_out_of_four_proof_l1521_152119

/-- The probability of selecting exactly one person from a group of two when randomly choosing two people from a group of four -/
theorem prob_select_one_from_two_out_of_four : ℚ :=
  2 / 3

/-- The total number of ways to select two people from four -/
def total_selections : ℕ := 6

/-- The number of ways to select exactly one person from a specific group of two when choosing two from four -/
def favorable_selections : ℕ := 4

/-- The probability is equal to the number of favorable outcomes divided by the total number of possible outcomes -/
theorem prob_select_one_from_two_out_of_four_proof :
  prob_select_one_from_two_out_of_four = favorable_selections / total_selections :=
sorry

end NUMINAMATH_CALUDE_prob_select_one_from_two_out_of_four_prob_select_one_from_two_out_of_four_proof_l1521_152119


namespace NUMINAMATH_CALUDE_work_left_fraction_l1521_152115

theorem work_left_fraction (a_days b_days work_days : ℕ) 
  (ha : a_days > 0) (hb : b_days > 0) (hw : work_days > 0) : 
  let total_work : ℚ := 1
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let combined_rate : ℚ := a_rate + b_rate
  let work_done : ℚ := combined_rate * work_days
  let work_left : ℚ := total_work - work_done
  (a_days = 15 ∧ b_days = 20 ∧ work_days = 5) → work_left = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l1521_152115


namespace NUMINAMATH_CALUDE_range_of_m_l1521_152147

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x^4 - x^2 + 1) / x^2 > m

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (-(5-2*m))^y < (-(5-2*m))^x

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m)) ∧ (¬∀ m : ℝ, (p m ∧ q m)) →
  ∃ a b : ℝ, a = 1 ∧ b = 2 ∧ ∀ m : ℝ, a ≤ m ∧ m < b :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1521_152147


namespace NUMINAMATH_CALUDE_franks_allowance_l1521_152102

/-- The amount Frank had saved up -/
def savings : ℕ := 3

/-- The number of toys Frank could buy -/
def num_toys : ℕ := 5

/-- The price of each toy -/
def toy_price : ℕ := 8

/-- The amount Frank received for his allowance -/
def allowance : ℕ := 37

theorem franks_allowance :
  savings + allowance = num_toys * toy_price :=
by sorry

end NUMINAMATH_CALUDE_franks_allowance_l1521_152102


namespace NUMINAMATH_CALUDE_symmetry_probability_l1521_152190

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- The size of the square grid -/
def gridSize : Nat := 11

/-- The center point of the grid -/
def centerPoint : GridPoint := ⟨gridSize / 2 + 1, gridSize / 2 + 1⟩

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of points excluding the center point -/
def remainingPoints : Nat := totalPoints - 1

/-- Checks if a point forms a line of symmetry with the center point -/
def isSymmetryPoint (p : GridPoint) : Bool :=
  p.x = centerPoint.x ∨ 
  p.y = centerPoint.y ∨ 
  p.x - centerPoint.x = p.y - centerPoint.y ∨
  p.x - centerPoint.x = centerPoint.y - p.y

/-- The number of points that form lines of symmetry -/
def symmetryPoints : Nat := 4 * (gridSize - 1)

/-- The probability theorem -/
theorem symmetry_probability : 
  (symmetryPoints : ℚ) / remainingPoints = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_symmetry_probability_l1521_152190


namespace NUMINAMATH_CALUDE_average_bracelets_per_day_l1521_152183

def bike_cost : ℕ := 112
def selling_weeks : ℕ := 2
def bracelet_price : ℕ := 1
def days_per_week : ℕ := 7

theorem average_bracelets_per_day :
  (bike_cost / (selling_weeks * days_per_week)) / bracelet_price = 8 :=
by sorry

end NUMINAMATH_CALUDE_average_bracelets_per_day_l1521_152183


namespace NUMINAMATH_CALUDE_triangle_side_difference_is_12_l1521_152175

def triangle_side_difference (y : ℤ) : Prop :=
  ∃ (a b : ℤ), 
    a = 7 ∧ b = 9 ∧  -- Given side lengths
    y > |a - b| ∧    -- Triangle inequality lower bound
    y < a + b ∧      -- Triangle inequality upper bound
    y ≥ 3 ∧ y ≤ 15   -- Integral bounds for y

theorem triangle_side_difference_is_12 : 
  (∀ y : ℤ, triangle_side_difference y → y ≤ 15) ∧ 
  (∀ y : ℤ, triangle_side_difference y → y ≥ 3) ∧
  (15 - 3 = 12) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_is_12_l1521_152175


namespace NUMINAMATH_CALUDE_gcd_lcm_properties_l1521_152186

theorem gcd_lcm_properties (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 100) : 
  (a * b = 2000) ∧ 
  (Nat.lcm (10 * a) (10 * b) = 10 * Nat.lcm a b) ∧ 
  ((10 * a) * (10 * b) = 100 * (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_properties_l1521_152186


namespace NUMINAMATH_CALUDE_car_speed_problem_l1521_152108

/-- Given a car traveling for 2 hours with a speed of 40 km/h in the second hour
    and an average speed of 65 km/h, prove that the speed in the first hour is 90 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 40 →
  average_speed = 65 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1521_152108


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_algebraic_expression_equality_l1521_152130

-- Part 1
theorem sqrt_expression_equality : 2 * Real.sqrt 20 - Real.sqrt 5 + 2 * Real.sqrt (1/5) = (17 * Real.sqrt 5) / 5 := by sorry

-- Part 2
theorem algebraic_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_algebraic_expression_equality_l1521_152130


namespace NUMINAMATH_CALUDE_successful_meeting_probability_l1521_152125

-- Define the arrival times as real numbers between 0 and 2 (representing hours after 3:00 p.m.)
variable (x y z : ℝ)

-- Define the conditions for a successful meeting
def successful_meeting (x y z : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 ∧
  0 ≤ y ∧ y ≤ 2 ∧
  0 ≤ z ∧ z ≤ 2 ∧
  z > x ∧ z > y ∧
  |x - y| ≤ 1.5

-- Define the probability space
def total_outcomes : ℝ := 8

-- Define the volume of the region where the meeting is successful
noncomputable def successful_volume : ℝ := 8/9

-- Theorem stating the probability of a successful meeting
theorem successful_meeting_probability :
  (successful_volume / total_outcomes) = 1/9 :=
sorry

end NUMINAMATH_CALUDE_successful_meeting_probability_l1521_152125


namespace NUMINAMATH_CALUDE_cubic_factorization_l1521_152101

theorem cubic_factorization (x : ℝ) : x^3 - 2*x^2 + x - 2 = (x^2 + 1)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1521_152101


namespace NUMINAMATH_CALUDE_two_cars_intersection_problem_l1521_152139

/-- Two cars approaching an intersection problem -/
theorem two_cars_intersection_problem 
  (s₁ : ℝ) (s₂ : ℝ) (v₁ : ℝ) (s : ℝ)
  (h₁ : s₁ = 1600) -- Initial distance of first car
  (h₂ : s₂ = 800)  -- Initial distance of second car
  (h₃ : v₁ = 72)   -- Speed of first car in km/h
  (h₄ : s = 200)   -- Distance between cars when first car reaches intersection
  : ∃ v₂ : ℝ, (v₂ = 7.5 ∨ v₂ = 12.5) ∧ 
    v₂ * (s₁ / (v₁ * 1000 / 3600)) = s₂ - s ∨ 
    v₂ * (s₁ / (v₁ * 1000 / 3600)) = s₂ + s :=
by sorry

end NUMINAMATH_CALUDE_two_cars_intersection_problem_l1521_152139


namespace NUMINAMATH_CALUDE_fraction_equality_l1521_152137

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 35) = 7 / 10 → a = 82 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1521_152137


namespace NUMINAMATH_CALUDE_parentheses_number_l1521_152177

theorem parentheses_number (x : ℤ) (h : x - (-2) = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_number_l1521_152177


namespace NUMINAMATH_CALUDE_zd_length_l1521_152138

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let xy := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  xy = 8 ∧ yz = 15 ∧ xz = 17

-- Define the angle bisector ZD
def AngleBisector (X Y Z D : ℝ × ℝ) : Prop :=
  let xd := Real.sqrt ((X.1 - D.1)^2 + (X.2 - D.2)^2)
  let yd := Real.sqrt ((Y.1 - D.1)^2 + (Y.2 - D.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  xd / yd = xz / yz

-- Theorem statement
theorem zd_length (X Y Z D : ℝ × ℝ) : 
  Triangle X Y Z → AngleBisector X Y Z D → 
  Real.sqrt ((Z.1 - D.1)^2 + (Z.2 - D.2)^2) = Real.sqrt 284.484375 :=
by sorry

end NUMINAMATH_CALUDE_zd_length_l1521_152138


namespace NUMINAMATH_CALUDE_smallest_multiple_of_36_and_45_not_25_l1521_152144

theorem smallest_multiple_of_36_and_45_not_25 :
  ∃ (n : ℕ), n > 0 ∧ 36 ∣ n ∧ 45 ∣ n ∧ ¬(25 ∣ n) ∧
  ∀ (m : ℕ), m > 0 → 36 ∣ m → 45 ∣ m → ¬(25 ∣ m) → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_36_and_45_not_25_l1521_152144


namespace NUMINAMATH_CALUDE_checkerboard_squares_l1521_152123

/-- The number of squares of a given size on a rectangular grid -/
def count_squares (rows : ℕ) (cols : ℕ) (size : ℕ) : ℕ :=
  (rows - size + 1) * (cols - size + 1)

/-- The total number of squares on a 3x4 checkerboard -/
def total_squares : ℕ :=
  count_squares 3 4 1 + count_squares 3 4 2 + count_squares 3 4 3

/-- Theorem stating that the total number of squares on a 3x4 checkerboard is 20 -/
theorem checkerboard_squares :
  total_squares = 20 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_squares_l1521_152123


namespace NUMINAMATH_CALUDE_intersection_M_N_l1521_152143

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1521_152143


namespace NUMINAMATH_CALUDE_range_of_a_l1521_152141

/-- Proposition p: For all x in ℝ, ax^2 + ax + 1 > 0 always holds -/
def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The function f(x) = 4x^2 - ax is monotonically increasing on [1, +∞) -/
def proposition_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → (4 * x^2 - a * x) < (4 * y^2 - a * y)

/-- The main theorem -/
theorem range_of_a :
  (∃ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬proposition_p a) →
  (∃ a : ℝ, a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1521_152141


namespace NUMINAMATH_CALUDE_initial_pens_count_prove_initial_pens_l1521_152106

theorem initial_pens_count : ℕ → Prop :=
  fun initial_pens =>
    let after_mike := initial_pens + 20
    let after_cindy := 2 * after_mike
    let after_sharon := after_cindy - 10
    after_sharon = initial_pens ∧ initial_pens = 30

theorem prove_initial_pens : ∃ (n : ℕ), initial_pens_count n := by
  sorry

end NUMINAMATH_CALUDE_initial_pens_count_prove_initial_pens_l1521_152106


namespace NUMINAMATH_CALUDE_guard_circles_l1521_152103

/-- Calculates the number of times a guard should circle a rectangular warehouse --/
def warehouseCircles (length width walked skipped : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let actualCircles := walked / perimeter
  actualCircles + skipped

/-- Theorem stating that for the given warehouse and guard's walk, the number of circles is 10 --/
theorem guard_circles : 
  warehouseCircles 600 400 16000 2 = 10 := by sorry

end NUMINAMATH_CALUDE_guard_circles_l1521_152103


namespace NUMINAMATH_CALUDE_points_are_coplanar_l1521_152100

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem points_are_coplanar
  (h_not_collinear : ¬ ∃ (k : ℝ), e₂ = k • e₁)
  (h_AB : B - A = e₁ + e₂)
  (h_AC : C - A = 2 • e₁ + 8 • e₂)
  (h_AD : D - A = 3 • e₁ - 5 • e₂) :
  ∃ (x y : ℝ), D - A = x • (B - A) + y • (C - A) :=
sorry

end NUMINAMATH_CALUDE_points_are_coplanar_l1521_152100


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l1521_152158

/-- Given a circle with radius 5 cm tangent to three sides of a rectangle, 
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the longer side of the rectangle is 7.5π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_radius = 5)
  (h2 : rectangle_area = 3 * π * circle_radius^2) : 
  rectangle_area / (2 * circle_radius) = 7.5 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l1521_152158


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_l1521_152109

/-- The problem of Tom's fruit purchase -/
theorem toms_fruit_purchase (apple_price : ℕ) (mango_price : ℕ) (apple_quantity : ℕ) (total_cost : ℕ) :
  apple_price = 70 →
  mango_price = 65 →
  apple_quantity = 8 →
  total_cost = 1145 →
  ∃ (mango_quantity : ℕ), 
    apple_price * apple_quantity + mango_price * mango_quantity = total_cost ∧ 
    mango_quantity = 9 := by
  sorry

#check toms_fruit_purchase

end NUMINAMATH_CALUDE_toms_fruit_purchase_l1521_152109


namespace NUMINAMATH_CALUDE_maggies_portion_l1521_152198

theorem maggies_portion (total : ℝ) (maggies_share : ℝ) (debbys_portion : ℝ) :
  total = 6000 →
  maggies_share = 4500 →
  debbys_portion = 0.25 →
  maggies_share / total = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_maggies_portion_l1521_152198


namespace NUMINAMATH_CALUDE_caden_coin_value_l1521_152146

/-- Represents the number of coins of each type Caden has -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in dollars -/
def total_value (coins : CoinCounts) : ℚ :=
  (coins.pennies : ℚ) / 100 +
  (coins.nickels : ℚ) / 20 +
  (coins.dimes : ℚ) / 10 +
  (coins.quarters : ℚ) / 4

/-- Theorem stating that Caden's coins total $8.00 -/
theorem caden_coin_value :
  ∀ (coins : CoinCounts),
    coins.pennies = 120 →
    coins.nickels = coins.pennies / 3 →
    coins.dimes = coins.nickels / 5 →
    coins.quarters = 2 * coins.dimes →
    total_value coins = 8 := by
  sorry

end NUMINAMATH_CALUDE_caden_coin_value_l1521_152146


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_sum_of_ratios_equals_zero_l1521_152151

-- Question 1
theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ p ≠ q →
    (a * Real.log (p + 2) - (p + 1)^2 - (a * Real.log (q + 2) - (q + 1)^2)) / (p - q) > 1) →
  a ≥ 28 :=
sorry

-- Question 2
theorem sum_of_ratios_equals_zero (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let f := fun x => (x - a) * (x - b) * (x - c)
  let f' := fun x => 3 * x^2 - 2 * (a + b + c) * x + (a * b + b * c + c * a)
  a / (f' a) + b / (f' b) + c / (f' c) = 0 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_sum_of_ratios_equals_zero_l1521_152151


namespace NUMINAMATH_CALUDE_nunzio_pizza_consumption_l1521_152166

/-- Calculates the number of whole pizzas eaten given daily pieces, days, and pieces per pizza -/
def pizzas_eaten (daily_pieces : ℕ) (days : ℕ) (pieces_per_pizza : ℕ) : ℕ :=
  (daily_pieces * days) / pieces_per_pizza

theorem nunzio_pizza_consumption :
  pizzas_eaten 3 72 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nunzio_pizza_consumption_l1521_152166


namespace NUMINAMATH_CALUDE_cube_root_of_64_l1521_152157

theorem cube_root_of_64 (m : ℝ) : (64 : ℝ)^(1/3) = 2^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l1521_152157


namespace NUMINAMATH_CALUDE_diamond_three_four_l1521_152178

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - b + 2

theorem diamond_three_four : diamond 3 4 = 142 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l1521_152178


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1521_152167

theorem complex_magnitude_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1521_152167


namespace NUMINAMATH_CALUDE_loan_amount_l1521_152110

/-- Proves that the amount lent is 2000 rupees given the specified conditions -/
theorem loan_amount (P : ℚ) 
  (h1 : P * (17/100 * 4 - 15/100 * 4) = 160) : P = 2000 := by
  sorry

#check loan_amount

end NUMINAMATH_CALUDE_loan_amount_l1521_152110


namespace NUMINAMATH_CALUDE_sine_of_alpha_l1521_152114

-- Define the angle α
variable (α : Real)

-- Define the point on the terminal side of α
def point : ℝ × ℝ := (3, 4)

-- Define sine function
noncomputable def sine (θ : Real) : Real :=
  point.2 / Real.sqrt (point.1^2 + point.2^2)

-- Theorem statement
theorem sine_of_alpha : sine α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_alpha_l1521_152114


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l1521_152163

theorem gcd_digits_bound (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 ∧
  1000000 ≤ b ∧ b < 10000000 ∧
  1000000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10000000000000 →
  Nat.gcd a b < 100 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l1521_152163


namespace NUMINAMATH_CALUDE_digit_101_of_7_26_l1521_152174

def decimal_expansion (n d : ℕ) : ℕ → ℕ
  | 0 => (10 * n / d) % 10
  | k + 1 => decimal_expansion (10 * (n % d)) d k

theorem digit_101_of_7_26 : decimal_expansion 7 26 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_digit_101_of_7_26_l1521_152174


namespace NUMINAMATH_CALUDE_circle_equation_from_line_l1521_152160

/-- Given a line in polar coordinates that intersects the polar axis, 
    find the polar equation of the circle with the intersection point's diameter --/
theorem circle_equation_from_line (θ : Real) (ρ p : Real → Real) :
  (∀ θ, p θ * Real.cos θ - 2 = 0) →  -- Line equation
  (∃ M : Real × Real, M.1 = 2 ∧ M.2 = 0) →  -- Intersection point
  (∀ θ, ρ θ = 2 * Real.cos θ) :=  -- Circle equation
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_line_l1521_152160


namespace NUMINAMATH_CALUDE_turtleneck_discount_l1521_152194

theorem turtleneck_discount (C : ℝ) (C_pos : C > 0) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_profit := 0.41
  let initial_price := C * (1 + initial_markup)
  let new_year_price := initial_price * (1 + new_year_markup)
  let february_price := C * (1 + february_profit)
  let discount := 1 - (february_price / new_year_price)
  discount = 0.06 := by
sorry

end NUMINAMATH_CALUDE_turtleneck_discount_l1521_152194


namespace NUMINAMATH_CALUDE_unique_solution_l1521_152193

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := lg (x + 1) = (1 / 2) * (Real.log x / Real.log 3)

-- Theorem statement
theorem unique_solution : ∃! x : ℝ, x > 0 ∧ equation x ∧ x = 9 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1521_152193


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achieved_l1521_152165

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  (a + 3*b + 5*c) * (a + b/3 + c/5) ≤ 9/5 := by
  sorry

theorem max_value_achieved (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀ + b₀ + c₀ = 1 ∧
    (a₀ + 3*b₀ + 5*c₀) * (a₀ + b₀/3 + c₀/5) = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achieved_l1521_152165


namespace NUMINAMATH_CALUDE_number_in_interval_l1521_152155

theorem number_in_interval (x : ℝ) (h : x = (1/x) * (-x) + 2) :
  x = 1 ∧ 0 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_number_in_interval_l1521_152155


namespace NUMINAMATH_CALUDE_population_growth_proof_l1521_152173

/-- The percentage increase in population during the first year -/
def first_year_increase : ℝ := 25

/-- The initial population two years ago -/
def initial_population : ℝ := 1200

/-- The population after two years of growth -/
def final_population : ℝ := 1950

/-- The percentage increase in population during the second year -/
def second_year_increase : ℝ := 30

theorem population_growth_proof :
  initial_population * (1 + first_year_increase / 100) * (1 + second_year_increase / 100) = final_population :=
sorry

end NUMINAMATH_CALUDE_population_growth_proof_l1521_152173


namespace NUMINAMATH_CALUDE_odd_digits_base4_157_l1521_152197

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of odd digits in the base-4 representation of 157 is 3 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 3 :=
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_157_l1521_152197


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1521_152199

theorem smallest_multiple_of_6_and_15 : 
  ∃ (a : ℕ), a > 0 ∧ 6 ∣ a ∧ 15 ∣ a ∧ ∀ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b → a ≤ b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1521_152199


namespace NUMINAMATH_CALUDE_abc_product_l1521_152131

theorem abc_product (a b c : ℝ) 
  (h1 : 1/a + 1/b + 1/c = 4)
  (h2 : 4 * (1/(a+b) + 1/(b+c) + 1/(c+a)) = 4)
  (h3 : c/(a+b) + a/(b+c) + b/(c+a) = 4) :
  a * b * c = 49/23 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1521_152131


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1521_152107

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 635 * n ≡ 1251 * n [ZMOD 30] ∧ 
  ∀ (m : ℕ), m > 0 → 635 * m ≡ 1251 * m [ZMOD 30] → n ≤ m :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1521_152107


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1521_152129

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1521_152129


namespace NUMINAMATH_CALUDE_area_ratio_of_triangles_l1521_152179

theorem area_ratio_of_triangles : 
  let mnp_sides : Fin 3 → ℝ := ![7, 24, 25]
  let qrs_sides : Fin 3 → ℝ := ![9, 12, 15]
  let mnp_area := (mnp_sides 0 * mnp_sides 1) / 2
  let qrs_area := (qrs_sides 0 * qrs_sides 1) / 2
  mnp_area / qrs_area = 14 / 9 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_of_triangles_l1521_152179


namespace NUMINAMATH_CALUDE_fourth_group_draw_l1521_152145

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_items : ℕ
  num_groups : ℕ
  first_draw : ℕ
  items_per_group : ℕ

/-- Calculates the number drawn in a given group for a systematic sampling -/
def draw_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_draw + s.items_per_group * (group - 1)

/-- Theorem: In the given systematic sampling, the number drawn in the fourth group is 22 -/
theorem fourth_group_draw (s : SystematicSampling) 
  (h1 : s.total_items = 30)
  (h2 : s.num_groups = 5)
  (h3 : s.first_draw = 4)
  (h4 : s.items_per_group = 6) :
  draw_in_group s 4 = 22 := by
  sorry


end NUMINAMATH_CALUDE_fourth_group_draw_l1521_152145


namespace NUMINAMATH_CALUDE_tangency_point_proof_l1521_152118

-- Define the two parabolas
def parabola1 (x y : ℚ) : Prop := y = x^2 + 20*x + 70
def parabola2 (x y : ℚ) : Prop := x = y^2 + 70*y + 1225

-- Define the point of tangency
def point_of_tangency : ℚ × ℚ := (-19/2, -69/2)

-- Theorem statement
theorem tangency_point_proof :
  let (x, y) := point_of_tangency
  parabola1 x y ∧ parabola2 x y ∧
  ∀ (x' y' : ℚ), x' ≠ x ∨ y' ≠ y →
    ¬(parabola1 x' y' ∧ parabola2 x' y') :=
by sorry

end NUMINAMATH_CALUDE_tangency_point_proof_l1521_152118


namespace NUMINAMATH_CALUDE_travel_time_proof_l1521_152191

def speed1 : ℝ := 6
def speed2 : ℝ := 12
def speed3 : ℝ := 18
def total_distance : ℝ := 1.8 -- 1800 meters converted to kilometers

theorem travel_time_proof :
  let d := total_distance / 3
  let time1 := d / speed1
  let time2 := d / speed2
  let time3 := d / speed3
  let total_time := (time1 + time2 + time3) * 60
  total_time = 11 := by
sorry

end NUMINAMATH_CALUDE_travel_time_proof_l1521_152191


namespace NUMINAMATH_CALUDE_parallel_to_a_l1521_152196

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- The vector a is defined as (-5, 4) -/
def a : ℝ × ℝ := (-5, 4)

/-- Theorem: A vector (x, y) is parallel to a = (-5, 4) if and only if
    there exists a real number k such that (x, y) = (-5k, 4k) -/
theorem parallel_to_a (x y : ℝ) :
  parallel (x, y) a ↔ ∃ k : ℝ, (x, y) = (-5 * k, 4 * k) :=
sorry

end NUMINAMATH_CALUDE_parallel_to_a_l1521_152196


namespace NUMINAMATH_CALUDE_seating_theorem_l1521_152164

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  groups : Nat
  seats_per_group : Nat
  extra_pair : Nat
  total_seats : Nat
  max_customers : Nat

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  arr.groups * arr.seats_per_group + arr.extra_pair = arr.total_seats ∧
  arr.max_customers ≤ arr.total_seats

/-- Checks if pairs can always be seated adjacently --/
def can_seat_pairs (arr : SeatingArrangement) : Prop :=
  ∀ n : Nat, n ≤ arr.max_customers → 
    (n / 2) * 2 ≤ arr.groups * 2 + arr.extra_pair

theorem seating_theorem (arr : SeatingArrangement) 
  (h1 : arr.groups = 7)
  (h2 : arr.seats_per_group = 3)
  (h3 : arr.extra_pair = 2)
  (h4 : arr.total_seats = 23)
  (h5 : arr.max_customers = 16)
  : is_valid_arrangement arr ∧ can_seat_pairs arr := by
  sorry

#check seating_theorem

end NUMINAMATH_CALUDE_seating_theorem_l1521_152164


namespace NUMINAMATH_CALUDE_sin_80_in_terms_of_tan_100_l1521_152170

theorem sin_80_in_terms_of_tan_100 (k : ℝ) (h : Real.tan (100 * π / 180) = k) :
  Real.sin (80 * π / 180) = -k / Real.sqrt (1 + k^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_80_in_terms_of_tan_100_l1521_152170


namespace NUMINAMATH_CALUDE_special_function_inequality_l1521_152111

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  greater_than_derivative : ∀ x, f x > deriv f x
  initial_value : f 0 = 1

/-- The main theorem -/
theorem special_function_inequality (F : SpecialFunction) :
  ∀ x, (F.f x / Real.exp x < 1) ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l1521_152111


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l1521_152140

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  b : ℝ
  s : ℝ
  t : ℝ
  b_nonzero : b ≠ 0
  line1_equation : 0 = 8 * s + b
  line2_equation : 0 = 4 * t + b

/-- The ratio of x-intercepts is 1/2 -/
theorem x_intercept_ratio (lines : TwoLines) : lines.s / lines.t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l1521_152140


namespace NUMINAMATH_CALUDE_count_maximal_arithmetic_sequences_correct_l1521_152184

/-- 
Given a positive integer n, count_maximal_arithmetic_sequences returns the number of 
maximal arithmetic sequences that can be formed from the set {1, 2, ..., n}.
A maximal arithmetic sequence is defined as an arithmetic sequence with a positive 
difference, containing at least two terms from the set, and to which no other element 
from the set can be added while maintaining the arithmetic progression.
-/
def count_maximal_arithmetic_sequences (n : ℕ) : ℕ :=
  (n^2) / 4

theorem count_maximal_arithmetic_sequences_correct (n : ℕ) :
  count_maximal_arithmetic_sequences n = ⌊(n^2 : ℚ) / 4⌋ := by
  sorry

#eval count_maximal_arithmetic_sequences 10  -- Expected output: 25

end NUMINAMATH_CALUDE_count_maximal_arithmetic_sequences_correct_l1521_152184


namespace NUMINAMATH_CALUDE_average_beef_sold_example_l1521_152133

/-- Calculates the average amount of beef sold per day over three days -/
def average_beef_sold (day1 : ℕ) (day2_multiplier : ℕ) (day3 : ℕ) : ℚ :=
  (day1 + day1 * day2_multiplier + day3) / 3

theorem average_beef_sold_example :
  average_beef_sold 210 2 150 = 260 := by
  sorry

end NUMINAMATH_CALUDE_average_beef_sold_example_l1521_152133


namespace NUMINAMATH_CALUDE_harrys_father_age_difference_l1521_152152

/-- Proves that Harry's father is 24 years older than Harry given the problem conditions -/
theorem harrys_father_age_difference : 
  ∀ (harry_age father_age mother_age : ℕ),
    harry_age = 50 →
    father_age > harry_age →
    mother_age = harry_age + 22 →
    father_age = mother_age + harry_age / 25 →
    father_age - harry_age = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_harrys_father_age_difference_l1521_152152


namespace NUMINAMATH_CALUDE_b_is_composite_greatest_number_of_factors_l1521_152132

/-- The greatest number of positive factors for b^m -/
def max_factors : ℕ := 81

/-- b is a positive integer less than or equal to 20 -/
def b : ℕ := 16

/-- m is a positive integer less than or equal to 20 -/
def m : ℕ := 20

/-- b is composite -/
theorem b_is_composite : ¬ Nat.Prime b := by sorry

theorem greatest_number_of_factors :
  ∀ b' m' : ℕ,
  b' ≤ 20 → m' ≤ 20 → b' > 1 → ¬ Nat.Prime b' →
  (Nat.divisors (b' ^ m')).card ≤ max_factors := by sorry

end NUMINAMATH_CALUDE_b_is_composite_greatest_number_of_factors_l1521_152132


namespace NUMINAMATH_CALUDE_triangle_side_squares_sum_l1521_152154

theorem triangle_side_squares_sum (a b c : ℝ) (h : a + b + c = 4) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^2 + b^2 + c^2 > 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_squares_sum_l1521_152154


namespace NUMINAMATH_CALUDE_f_difference_180_90_l1521_152126

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ+) : ℕ := sorry

-- Define the function f
def f (n : ℕ+) : ℚ := (sum_of_divisors n : ℚ) / n

-- Theorem statement
theorem f_difference_180_90 : f 180 - f 90 = 13 / 30 := by sorry

end NUMINAMATH_CALUDE_f_difference_180_90_l1521_152126


namespace NUMINAMATH_CALUDE_triangle_properties_l1521_152150

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π/2) →
  (a * c * Real.cos B - b * c * Real.cos A = 3 * b^2) →
  (c = Real.sqrt 11) →
  (Real.sin C = 2 * Real.sqrt 2 / 3) →
  (Real.sin A / Real.sin B = Real.sqrt 7) ∧
  (1/2 * a * b * Real.sin C = Real.sqrt 14) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1521_152150


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1521_152116

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem a_plus_b_value (a b : ℝ) : 
  A ∪ B a b = Set.univ ∧ A ∩ B a b = Set.Ioc 3 4 → a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1521_152116


namespace NUMINAMATH_CALUDE_ponce_lighter_than_jalen_l1521_152188

/-- Represents the weights of three people and their relationships. -/
structure WeightProblem where
  ishmael : ℝ
  ponce : ℝ
  jalen : ℝ
  ishmael_heavier : ishmael = ponce + 20
  jalen_weight : jalen = 160
  average_weight : (ishmael + ponce + jalen) / 3 = 160

/-- Theorem stating that Ponce is 10 pounds lighter than Jalen. -/
theorem ponce_lighter_than_jalen (w : WeightProblem) : w.jalen - w.ponce = 10 := by
  sorry

#check ponce_lighter_than_jalen

end NUMINAMATH_CALUDE_ponce_lighter_than_jalen_l1521_152188


namespace NUMINAMATH_CALUDE_prism_cone_properties_l1521_152105

/-- Regular triangular prism with a point T on edge BB₁ forming a cone --/
structure PrismWithCone where
  -- Base edge length of the prism
  a : ℝ
  -- Height of the prism
  h : ℝ
  -- Distance BT
  bt : ℝ
  -- Distance B₁T
  b₁t : ℝ
  -- Constraint on BT:B₁T ratio
  h_ratio : bt / b₁t = 2 / 3
  -- Constraint on prism height
  h_height : h = 5

/-- Theorem about the ratio of prism height to base edge and cone volume --/
theorem prism_cone_properties (p : PrismWithCone) :
  -- 1. Ratio of prism height to base edge is √5
  p.h / p.a = Real.sqrt 5 ∧
  -- 2. Volume of the cone
  ∃ (v : ℝ), v = (180 * Real.pi * Real.sqrt 3) / (23 * Real.sqrt 23) := by
  sorry

end NUMINAMATH_CALUDE_prism_cone_properties_l1521_152105


namespace NUMINAMATH_CALUDE_isosceles_triangles_count_l1521_152149

/-- A right hexagonal prism with height 2 and regular hexagonal bases of side length 1 -/
structure HexagonalPrism where
  height : ℝ
  base_side_length : ℝ
  height_eq : height = 2
  side_eq : base_side_length = 1

/-- A triangle formed by three vertices of the hexagonal prism -/
structure PrismTriangle where
  prism : HexagonalPrism
  v1 : Fin 12
  v2 : Fin 12
  v3 : Fin 12
  distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- Predicate to determine if a triangle is isosceles -/
def is_isosceles (t : PrismTriangle) : Prop :=
  sorry

/-- The number of isosceles triangles in the hexagonal prism -/
def num_isosceles_triangles (p : HexagonalPrism) : ℕ :=
  sorry

/-- Theorem stating that the number of isosceles triangles is 24 -/
theorem isosceles_triangles_count (p : HexagonalPrism) :
  num_isosceles_triangles p = 24 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_count_l1521_152149


namespace NUMINAMATH_CALUDE_minimum_cookies_cookies_exist_l1521_152192

theorem minimum_cookies (b : ℕ) : b ≡ 5 [ZMOD 6] ∧ b ≡ 7 [ZMOD 8] ∧ b ≡ 8 [ZMOD 9] → b ≥ 239 := by
  sorry

theorem cookies_exist : ∃ b : ℕ, b ≡ 5 [ZMOD 6] ∧ b ≡ 7 [ZMOD 8] ∧ b ≡ 8 [ZMOD 9] ∧ b = 239 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cookies_cookies_exist_l1521_152192


namespace NUMINAMATH_CALUDE_B_2_1_eq_12_l1521_152136

def B : ℕ → ℕ → ℕ
  | 0, n => n + 2
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_1_eq_12 : B 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_B_2_1_eq_12_l1521_152136


namespace NUMINAMATH_CALUDE_tan_over_cos_squared_l1521_152169

theorem tan_over_cos_squared (α : Real) (P : ℝ × ℝ) :
  P = (-1, 2) →
  (∃ r : ℝ, r > 0 ∧ P = (r * Real.cos α, r * Real.sin α)) →
  Real.tan α / (Real.cos α)^2 = -10 :=
by sorry

end NUMINAMATH_CALUDE_tan_over_cos_squared_l1521_152169


namespace NUMINAMATH_CALUDE_light_2004_is_yellow_l1521_152127

def light_sequence : ℕ → Fin 4
  | n => match n % 7 with
    | 0 => 0  -- green
    | 1 => 1  -- yellow
    | 2 => 1  -- yellow
    | 3 => 2  -- red
    | 4 => 3  -- blue
    | 5 => 2  -- red
    | _ => 2  -- red

theorem light_2004_is_yellow : light_sequence 2003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_light_2004_is_yellow_l1521_152127


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1521_152159

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 5, 6}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1521_152159


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l1521_152187

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l1521_152187


namespace NUMINAMATH_CALUDE_square_of_1003_l1521_152185

theorem square_of_1003 : (1003 : ℕ)^2 = 1006009 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1003_l1521_152185
