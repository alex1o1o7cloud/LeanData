import Mathlib

namespace NUMINAMATH_CALUDE_estimate_fish_population_l1607_160731

/-- Estimates the number of fish in a lake using the mark-recapture method. -/
theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) (h5 : k ≤ n) :
  (n * m : ℚ) / k = (m : ℚ) / (k : ℚ) * n :=
by sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l1607_160731


namespace NUMINAMATH_CALUDE_f_nonnegative_implies_a_range_l1607_160733

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- State the theorem
theorem f_nonnegative_implies_a_range (a b : ℝ) :
  (∀ x ≥ 2, f a b x ≥ 0) → a ∈ Set.Ioo (-9 : ℝ) (-3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_implies_a_range_l1607_160733


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1607_160757

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1607_160757


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l1607_160721

theorem sqrt_difference_approximation : |Real.sqrt 144 - Real.sqrt 140 - 0.17| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l1607_160721


namespace NUMINAMATH_CALUDE_unique_special_polynomial_l1607_160728

/-- A polynomial function that satisfies the given conditions -/
structure SpecialPolynomial where
  f : ℝ → ℝ
  is_polynomial : Polynomial ℝ
  degree_ge_one : (Polynomial.degree is_polynomial) ≥ 1
  cond_square : ∀ x, f (x^2) = (f x)^2
  cond_compose : ∀ x, f (x^2) = f (f x)

/-- Theorem stating that there exists exactly one special polynomial -/
theorem unique_special_polynomial :
  ∃! (p : SpecialPolynomial), True :=
sorry

end NUMINAMATH_CALUDE_unique_special_polynomial_l1607_160728


namespace NUMINAMATH_CALUDE_trig_identity_l1607_160777

theorem trig_identity (α : ℝ) (h : Real.sin α + 3 * Real.cos α = 0) : 
  2 * Real.sin (2 * α) - (Real.cos α)^2 = -13/10 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1607_160777


namespace NUMINAMATH_CALUDE_tuesday_kids_l1607_160744

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 24

/-- The difference in the number of kids Julia played with between Monday and Tuesday -/
def difference : ℕ := 18

/-- Theorem: The number of kids Julia played with on Tuesday is 6 -/
theorem tuesday_kids : monday_kids - difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_kids_l1607_160744


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1607_160764

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1607_160764


namespace NUMINAMATH_CALUDE_cost_calculation_l1607_160703

theorem cost_calculation (pencil_cost pen_cost eraser_cost : ℝ) 
  (eq1 : 8 * pencil_cost + 2 * pen_cost + eraser_cost = 4.60)
  (eq2 : 2 * pencil_cost + 5 * pen_cost + eraser_cost = 3.90)
  (eq3 : pencil_cost + pen_cost + 3 * eraser_cost = 2.75) :
  4 * pencil_cost + 3 * pen_cost + 2 * eraser_cost = 7.4135 := by
sorry

end NUMINAMATH_CALUDE_cost_calculation_l1607_160703


namespace NUMINAMATH_CALUDE_equation_solution_l1607_160717

theorem equation_solution (x : ℝ) :
  (|Real.cos x| - Real.cos (3 * x)) / (Real.cos x * Real.sin (2 * x)) = 2 / Real.sqrt 3 ↔
  (∃ k : ℤ, x = π / 6 + 2 * k * π ∨ x = 5 * π / 6 + 2 * k * π ∨ x = 4 * π / 3 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1607_160717


namespace NUMINAMATH_CALUDE_total_miles_equals_484_l1607_160727

/-- The number of ladies in the walking group -/
def num_ladies : ℕ := 5

/-- The number of miles walked together by the group per day -/
def group_miles_per_day : ℕ := 3

/-- The number of days per week the group walks together -/
def group_days_per_week : ℕ := 6

/-- Jamie's additional miles walked per day -/
def jamie_additional_miles : ℕ := 2

/-- Sue's additional miles walked per day (half of Jamie's) -/
def sue_additional_miles : ℕ := jamie_additional_miles / 2

/-- Laura's additional miles walked every two days -/
def laura_additional_miles : ℕ := 1

/-- Melissa's additional miles walked every three days -/
def melissa_additional_miles : ℕ := 2

/-- Katie's additional miles walked per day -/
def katie_additional_miles : ℕ := 1

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Calculate the total miles walked by all ladies in the group during a month -/
def total_miles_per_month : ℕ :=
  let jamie_miles := (group_miles_per_day * group_days_per_week + jamie_additional_miles * group_days_per_week) * weeks_per_month
  let sue_miles := (group_miles_per_day * group_days_per_week + sue_additional_miles * group_days_per_week) * weeks_per_month
  let laura_miles := (group_miles_per_day * group_days_per_week + laura_additional_miles * 3) * weeks_per_month
  let melissa_miles := (group_miles_per_day * group_days_per_week + melissa_additional_miles * 2) * weeks_per_month
  let katie_miles := (group_miles_per_day * group_days_per_week + katie_additional_miles * group_days_per_week) * weeks_per_month
  jamie_miles + sue_miles + laura_miles + melissa_miles + katie_miles

theorem total_miles_equals_484 : total_miles_per_month = 484 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_equals_484_l1607_160727


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1607_160735

theorem arithmetic_calculation : (-3 + 2) * 3 - (-4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1607_160735


namespace NUMINAMATH_CALUDE_mikes_tire_changes_l1607_160743

/-- The number of tires changed by Mike in a day -/
def total_tires_changed (
  motorcycles cars bicycles trucks atvs : ℕ)
  (motorcycle_wheels car_wheels bicycle_wheels truck_wheels atv_wheels : ℕ) : ℕ :=
  motorcycles * motorcycle_wheels +
  cars * car_wheels +
  bicycles * bicycle_wheels +
  trucks * truck_wheels +
  atvs * atv_wheels

/-- Theorem stating the total number of tires changed by Mike in a day -/
theorem mikes_tire_changes :
  total_tires_changed 12 10 8 5 7 2 4 2 18 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_mikes_tire_changes_l1607_160743


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l1607_160767

/-- Represents a hyperbola with equation x²/4 - y²/9 = 1 -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  equation_def : equation = fun x y => x^2 / 4 - y^2 / 9 = 1

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Angle between three points in degrees -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- A point is on the hyperbola if it satisfies the equation -/
def onHyperbola (h : Hyperbola) (p : Point) : Prop :=
  h.equation p.x p.y

/-- Foci of the hyperbola -/
def areFoci (h : Hyperbola) (f1 f2 : Point) : Prop := sorry

theorem hyperbola_triangle_area 
  (h : Hyperbola) 
  (a b m : Point) 
  (h_foci : areFoci h a b)
  (h_on_hyperbola : onHyperbola h m)
  (h_angle : angle a m b = 120) :
  triangleArea a m b = 2 * Real.sqrt 3 := sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l1607_160767


namespace NUMINAMATH_CALUDE_maxwell_current_age_l1607_160725

/-- Maxwell's current age -/
def maxwell_age : ℕ := sorry

/-- Maxwell's sister's current age -/
def sister_age : ℕ := 2

/-- In 2 years, Maxwell will be twice his sister's age -/
axiom maxwell_twice_sister : maxwell_age + 2 = 2 * (sister_age + 2)

theorem maxwell_current_age : maxwell_age = 6 := by sorry

end NUMINAMATH_CALUDE_maxwell_current_age_l1607_160725


namespace NUMINAMATH_CALUDE_rose_flowers_l1607_160749

/-- The number of flowers Rose bought -/
def total_flowers : ℕ := 12

/-- The number of daisies -/
def daisies : ℕ := 2

/-- The number of sunflowers -/
def sunflowers : ℕ := 4

/-- The number of tulips -/
def tulips : ℕ := (3 * (total_flowers - daisies)) / 5

theorem rose_flowers :
  total_flowers = daisies + tulips + sunflowers ∧
  tulips = (3 * (total_flowers - daisies)) / 5 ∧
  sunflowers = (2 * (total_flowers - daisies)) / 5 :=
by sorry

end NUMINAMATH_CALUDE_rose_flowers_l1607_160749


namespace NUMINAMATH_CALUDE_only_345_right_triangle_l1607_160704

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that only (3, 4, 5) forms a right-angled triangle among the given sets --/
theorem only_345_right_triangle :
  ¬ is_right_triangle 2 4 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) 2 2 ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 5 12 14 :=
by sorry

end NUMINAMATH_CALUDE_only_345_right_triangle_l1607_160704


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1607_160706

/-- The length of the major axis of the ellipse x^2/49 + y^2/81 = 1 is 18 -/
theorem ellipse_major_axis_length : 
  let a := Real.sqrt (max 49 81)
  2 * a = 18 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1607_160706


namespace NUMINAMATH_CALUDE_digit_sum_is_seventeen_l1607_160719

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

/-- The equation (AB) * (CD) = GGG -/
def satisfiesEquation (A B C D G : Digit) : Prop :=
  ∃ (AB CD : TwoDigitNumber) (GGG : ThreeDigitNumber),
    AB.val = 10 * A.val + B.val ∧
    CD.val = 10 * C.val + D.val ∧
    GGG.val = 100 * G.val + 10 * G.val + G.val ∧
    AB.val * CD.val = GGG.val

/-- All digits are distinct -/
def allDistinct (A B C D G : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ G ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ G ∧
  C ≠ D ∧ C ≠ G ∧
  D ≠ G

theorem digit_sum_is_seventeen :
  ∃ (A B C D G : Digit),
    satisfiesEquation A B C D G ∧
    allDistinct A B C D G ∧
    A.val + B.val + C.val + D.val + G.val = 17 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_is_seventeen_l1607_160719


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l1607_160797

theorem sequence_is_arithmetic (a : ℕ+ → ℝ)
  (h : ∀ p q : ℕ+, a p = a q + 2003 * (p - q)) :
  ∃ d : ℝ, ∀ n m : ℕ+, a n = a m + d * (n - m) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l1607_160797


namespace NUMINAMATH_CALUDE_sophie_bought_four_boxes_l1607_160732

/-- The number of boxes of donuts Sophie bought -/
def boxes_bought : ℕ := sorry

/-- The number of donuts in each box -/
def donuts_per_box : ℕ := 12

/-- The number of boxes Sophie gave to her mom -/
def boxes_to_mom : ℕ := 1

/-- The number of donuts Sophie gave to her sister -/
def donuts_to_sister : ℕ := 6

/-- The number of donuts Sophie had left for herself -/
def donuts_left : ℕ := 30

theorem sophie_bought_four_boxes : boxes_bought = 4 := by sorry

end NUMINAMATH_CALUDE_sophie_bought_four_boxes_l1607_160732


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1607_160774

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) is on a line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point (x₀ y₀ : ℝ) :
  ∃ (l : Line), l.isParallel ⟨1, -2, -2⟩ ∧ l.containsPoint 1 0 ∧ l = ⟨1, -2, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1607_160774


namespace NUMINAMATH_CALUDE_fraction_simplification_l1607_160779

theorem fraction_simplification (a x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 + a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1607_160779


namespace NUMINAMATH_CALUDE_distinct_integers_count_l1607_160769

def odd_squares_list : List ℤ :=
  (List.range 500).map (fun k => ⌊((2*k + 1)^2 : ℚ) / 500⌋)

theorem distinct_integers_count : (odd_squares_list.eraseDups).length = 469 := by
  sorry

end NUMINAMATH_CALUDE_distinct_integers_count_l1607_160769


namespace NUMINAMATH_CALUDE_product_cde_value_l1607_160718

theorem product_cde_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.6666666666666666) :
  c * d * e = 750 := by
  sorry

end NUMINAMATH_CALUDE_product_cde_value_l1607_160718


namespace NUMINAMATH_CALUDE_flour_bag_weight_l1607_160707

/-- Calculates the weight of each bag of flour given the problem conditions --/
theorem flour_bag_weight 
  (flour_needed : ℕ) 
  (bag_cost : ℚ) 
  (salt_needed : ℕ) 
  (salt_cost_per_pound : ℚ) 
  (promotion_cost : ℕ) 
  (ticket_price : ℕ) 
  (tickets_sold : ℕ) 
  (total_profit : ℚ) 
  (h1 : flour_needed = 500) 
  (h2 : bag_cost = 20) 
  (h3 : salt_needed = 10) 
  (h4 : salt_cost_per_pound = 0.2) 
  (h5 : promotion_cost = 1000) 
  (h6 : ticket_price = 20) 
  (h7 : tickets_sold = 500) 
  (h8 : total_profit = 8798) : 
  ℕ := by
  sorry

#check flour_bag_weight

end NUMINAMATH_CALUDE_flour_bag_weight_l1607_160707


namespace NUMINAMATH_CALUDE_total_distance_driven_l1607_160771

def miles_per_gallon : ℝ := 25
def tank_capacity : ℝ := 18
def initial_gas : ℝ := 12
def first_leg_distance : ℝ := 250
def gas_purchased : ℝ := 10
def final_gas : ℝ := 3

theorem total_distance_driven : ℝ := by
  -- The total distance driven is 475 miles
  sorry

#check total_distance_driven

end NUMINAMATH_CALUDE_total_distance_driven_l1607_160771


namespace NUMINAMATH_CALUDE_sin_eq_sin_sin_unique_solution_l1607_160724

noncomputable def arcsin_099 : ℝ := Real.arcsin 0.99

theorem sin_eq_sin_sin_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ arcsin_099 ∧ Real.sin x = Real.sin (Real.sin x) :=
sorry

end NUMINAMATH_CALUDE_sin_eq_sin_sin_unique_solution_l1607_160724


namespace NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_ninety_percent_l1607_160737

theorem increase_by_percentage (x : ℝ) (p : ℝ) : 
  x * (1 + p / 100) = x + x * (p / 100) := by sorry

theorem seventy_five_increased_by_ninety_percent : 
  75 * (1 + 90 / 100) = 142.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_ninety_percent_l1607_160737


namespace NUMINAMATH_CALUDE_angle_D_measure_l1607_160782

/-- Prove that given the specified angle conditions, angle D measures 25 degrees. -/
theorem angle_D_measure (A B C D : ℝ) : 
  A + B = 180 →
  C = D →
  A = 50 →
  D = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l1607_160782


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1607_160746

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure CoinSet :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (half_dollar : CoinFlip)
  (dollar : CoinFlip)

/-- The condition for a successful outcome -/
def successful_outcome (cs : CoinSet) : Prop :=
  (cs.penny = cs.nickel) ∧
  (cs.dime = cs.quarter) ∧ (cs.quarter = cs.half_dollar)

/-- The total number of possible outcomes -/
def total_outcomes : Nat := 64

/-- The number of successful outcomes -/
def successful_outcomes : Nat := 16

/-- The theorem to be proved -/
theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1607_160746


namespace NUMINAMATH_CALUDE_total_nails_needed_l1607_160788

/-- The total number of nails needed is equal to the sum of initial nails, 
    found nails, and nails to buy. -/
theorem total_nails_needed 
  (initial_nails : ℕ) 
  (found_nails : ℕ) 
  (nails_to_buy : ℕ) : 
  initial_nails + found_nails + nails_to_buy = 
  initial_nails + found_nails + nails_to_buy := by
  sorry

#eval 247 + 144 + 109

end NUMINAMATH_CALUDE_total_nails_needed_l1607_160788


namespace NUMINAMATH_CALUDE_cone_volume_from_triangle_rotation_l1607_160775

/-- The volume of a cone formed by rotating a right triangle -/
def cone_volume (S L : ℝ) : ℝ :=
  S * L

/-- Theorem: The volume of a cone formed by rotating a right triangle with area S
    around one of its legs is equal to SL, where L is the length of the circumference
    described by the intersection point of the medians during rotation -/
theorem cone_volume_from_triangle_rotation (S L : ℝ) (h1 : S > 0) (h2 : L > 0) :
  cone_volume S L = S * L :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_triangle_rotation_l1607_160775


namespace NUMINAMATH_CALUDE_negation_equivalence_l1607_160740

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 1, x^2 + 2*x ≤ 1) ↔ 
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, x^2 + 2*x > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1607_160740


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1607_160708

theorem rectangle_dimension_change (w l : ℝ) (h_w_pos : w > 0) (h_l_pos : l > 0) :
  let new_w := 1.4 * w
  let new_l := l / 1.4
  let area := w * l
  let new_area := new_w * new_l
  new_area = area ∧ (1 - new_l / l) * 100 = 100 * (1 - 1 / 1.4) := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1607_160708


namespace NUMINAMATH_CALUDE_no_integer_square_root_l1607_160784

theorem no_integer_square_root : ¬ ∃ (y : ℤ) (b : ℤ), y^4 + 8*y^3 + 18*y^2 + 10*y + 41 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l1607_160784


namespace NUMINAMATH_CALUDE_solve_for_y_l1607_160715

theorem solve_for_y (x y z : ℤ) 
  (eq1 : x + y + z = 355)
  (eq2 : x - y = 200)
  (eq3 : x + z = 500) :
  y = -145 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l1607_160715


namespace NUMINAMATH_CALUDE_dice_trick_existence_l1607_160765

def DicePair : Type := { p : ℕ × ℕ // p.1 ≤ p.2 ∧ p.1 ≥ 1 ∧ p.2 ≤ 6 }

theorem dice_trick_existence :
  ∃ f : DicePair → ℕ,
    Function.Bijective f ∧
    (∀ p : DicePair, 3 ≤ f p ∧ f p ≤ 21) :=
sorry

end NUMINAMATH_CALUDE_dice_trick_existence_l1607_160765


namespace NUMINAMATH_CALUDE_no_common_terms_except_first_l1607_160789

def X : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => X (n + 1) + 2 * X n

def Y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * Y (n + 1) + 3 * Y n

theorem no_common_terms_except_first : ∀ n m : ℕ, X n = Y m → n = 0 ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_except_first_l1607_160789


namespace NUMINAMATH_CALUDE_no_integer_solution_l1607_160714

theorem no_integer_solution : ¬∃ (x y z : ℤ), x^3 + y^3 = z^3 + 4 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1607_160714


namespace NUMINAMATH_CALUDE_line_relations_l1607_160780

def l1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

def l2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

theorem line_relations (m : ℝ) :
  (∀ x y, l1 m x y → l2 m x y → (m - 2 + 3 * m = 0) ↔ m = 1/2) ∧
  (∀ x y, l1 m x y → l2 m x y → ((m - 2) / 1 = 3 / m ∧ m ≠ 3 ∧ m ≠ -3) ↔ m = -1) ∧
  (∀ x y, l1 m x y → l2 m x y → ((m - 2) / 1 = 3 / m ∧ 3 / m = 2 * m / 6) ↔ m = 3) ∧
  (∀ x y, l1 m x y → l2 m x y → (m ≠ 3 ∧ m ≠ -1) ↔ (m ≠ 3 ∧ m ≠ -1)) :=
by sorry

end NUMINAMATH_CALUDE_line_relations_l1607_160780


namespace NUMINAMATH_CALUDE_pencil_gain_percentage_l1607_160747

/-- Represents the cost price of a single pencil in rupees -/
def cost_price_per_pencil : ℚ := 1 / 12

/-- Represents the selling price of 15 pencils in rupees -/
def selling_price_15 : ℚ := 1

/-- Represents the selling price of 10 pencils in rupees -/
def selling_price_10 : ℚ := 1

/-- The loss percentage when selling 15 pencils for a rupee -/
def loss_percentage : ℚ := 20 / 100

theorem pencil_gain_percentage :
  let cost_15 := 15 * cost_price_per_pencil
  let cost_10 := 10 * cost_price_per_pencil
  selling_price_15 = (1 - loss_percentage) * cost_15 →
  (selling_price_10 - cost_10) / cost_10 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_gain_percentage_l1607_160747


namespace NUMINAMATH_CALUDE_largest_number_l1607_160783

theorem largest_number (a b c d e : ℝ) : 
  a = 15679 + 1/3579 → 
  b = 15679 - 1/3579 → 
  c = 15679 * (1/3579) → 
  d = 15679 / (1/3579) → 
  e = 15679 * 1.03 → 
  d > a ∧ d > b ∧ d > c ∧ d > e := by
sorry

end NUMINAMATH_CALUDE_largest_number_l1607_160783


namespace NUMINAMATH_CALUDE_xyz_equation_solutions_l1607_160761

theorem xyz_equation_solutions :
  ∀ (x y z : ℕ), x * y * z = x + y → ((x = 2 ∧ y = 2 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 2)) :=
by sorry

end NUMINAMATH_CALUDE_xyz_equation_solutions_l1607_160761


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1607_160726

-- Problem 1
theorem problem_one : (Real.sqrt 48 + Real.sqrt 20) - (Real.sqrt 12 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_two : |2 - Real.sqrt 2| - Real.sqrt (1/12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1607_160726


namespace NUMINAMATH_CALUDE_shirts_arrangement_l1607_160786

/-- The number of ways to arrange shirts -/
def arrange_shirts (red : Nat) (green : Nat) : Nat :=
  Nat.factorial (red + green) / (Nat.factorial red * Nat.factorial green)

/-- The number of ways to arrange shirts with green shirts together -/
def arrange_shirts_green_together (red : Nat) (green : Nat) : Nat :=
  arrange_shirts red 1

theorem shirts_arrangement :
  arrange_shirts 3 2 - arrange_shirts_green_together 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirts_arrangement_l1607_160786


namespace NUMINAMATH_CALUDE_earliest_year_500_mismatched_l1607_160796

/-- Number of shoe pairs in Moor's room in a given year -/
def shoe_pairs (year : ℕ) : ℕ := 2^(year - 2013)

/-- Number of mismatched shoe pairs possible with a given number of shoe pairs -/
def mismatched_pairs (pairs : ℕ) : ℕ := pairs * (pairs - 1)

/-- Predicate for whether a year allows at least 500 mismatched pairs -/
def can_wear_500_mismatched (year : ℕ) : Prop :=
  mismatched_pairs (shoe_pairs year) ≥ 500

theorem earliest_year_500_mismatched :
  (∀ y < 2018, ¬ can_wear_500_mismatched y) ∧ can_wear_500_mismatched 2018 := by
  sorry

end NUMINAMATH_CALUDE_earliest_year_500_mismatched_l1607_160796


namespace NUMINAMATH_CALUDE_max_xyz_value_l1607_160793

theorem max_xyz_value (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 2) (h_sq_sum : x^2 + y^2 + z^2 = x*z + y*z + x*y) :
  x*y*z ≤ 8/27 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 2 ∧ a^2 + b^2 + c^2 = a*c + b*c + a*b ∧ a*b*c = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_max_xyz_value_l1607_160793


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1607_160713

theorem angle_measure_proof (x : ℝ) : 
  (90 - x + 40 = (180 - x) / 2) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1607_160713


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1607_160720

theorem fourth_power_sum (α β γ : ℂ) 
  (h1 : α + β + γ = 1)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 9) :
  α^4 + β^4 + γ^4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1607_160720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l1607_160753

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 1 + a 5 + a 7 + a 9 + a 13 = 100) ∧
  (a 6 - a 2 = 12)

/-- The theorem stating that a_1 = 2 for the given arithmetic sequence -/
theorem arithmetic_sequence_a1_value (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l1607_160753


namespace NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l1607_160716

theorem coefficient_m5n5_in_expansion : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l1607_160716


namespace NUMINAMATH_CALUDE_subset_condition_implies_a_geq_three_l1607_160791

/-- Given a > 0, if the set A is a subset of set B, then a ≥ 3 -/
theorem subset_condition_implies_a_geq_three (a : ℝ) (h : a > 0) :
  ({x : ℝ | (x - 2) * (x - 3 * a - 2) < 0} ⊆ {x : ℝ | (x - 1) * (x - a^2 - 2) < 0}) →
  a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_implies_a_geq_three_l1607_160791


namespace NUMINAMATH_CALUDE_intersection_implies_m_eq_neg_two_l1607_160799

-- Define the sets M and N
def M (m : ℝ) : Set ℂ := {1, 2, (m^2 - 2*m - 5 : ℂ) + (m^2 + 5*m + 6 : ℂ)*Complex.I}
def N : Set ℂ := {3}

-- State the theorem
theorem intersection_implies_m_eq_neg_two (m : ℝ) : 
  (M m ∩ N).Nonempty → m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_eq_neg_two_l1607_160799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l1607_160773

theorem arithmetic_sequence_sum_mod_15 : 
  let first_term := 1
  let last_term := 101
  let common_diff := 5
  let num_terms := (last_term - first_term) / common_diff + 1
  ∃ (sum : ℕ), sum = (num_terms * (first_term + last_term)) / 2 ∧ sum ≡ 6 [MOD 15] :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l1607_160773


namespace NUMINAMATH_CALUDE_real_roots_quadratic_l1607_160710

theorem real_roots_quadratic (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ k ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_l1607_160710


namespace NUMINAMATH_CALUDE_expected_total_audience_l1607_160795

theorem expected_total_audience (saturday_attendance : ℕ) 
  (monday_attendance : ℕ) (wednesday_attendance : ℕ) (friday_attendance : ℕ) 
  (actual_total : ℕ) (expected_total : ℕ) : 
  saturday_attendance = 80 →
  monday_attendance = saturday_attendance - 20 →
  wednesday_attendance = monday_attendance + 50 →
  friday_attendance = saturday_attendance + monday_attendance →
  actual_total = saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance →
  actual_total = expected_total + 40 →
  expected_total = 350 := by
sorry

end NUMINAMATH_CALUDE_expected_total_audience_l1607_160795


namespace NUMINAMATH_CALUDE_four_customers_no_change_l1607_160758

/-- Represents the auto shop scenario -/
structure AutoShop where
  initial_cars : ℕ
  new_customers : ℕ
  tires_per_car : ℕ
  half_change_customers : ℕ
  tires_left : ℕ

/-- Calculates the number of customers who didn't want their tires changed -/
def customers_no_change (shop : AutoShop) : ℕ :=
  let total_cars := shop.initial_cars + shop.new_customers
  let total_tires_bought := total_cars * shop.tires_per_car
  let half_change_tires := shop.half_change_customers * (shop.tires_per_car / 2)
  let unused_tires := shop.tires_left - half_change_tires
  unused_tires / shop.tires_per_car

/-- Theorem stating that given the conditions, 4 customers decided not to change their tires -/
theorem four_customers_no_change (shop : AutoShop) 
  (h1 : shop.initial_cars = 4)
  (h2 : shop.new_customers = 6)
  (h3 : shop.tires_per_car = 4)
  (h4 : shop.half_change_customers = 2)
  (h5 : shop.tires_left = 20) :
  customers_no_change shop = 4 := by
  sorry

#eval customers_no_change { initial_cars := 4, new_customers := 6, tires_per_car := 4, half_change_customers := 2, tires_left := 20 }

end NUMINAMATH_CALUDE_four_customers_no_change_l1607_160758


namespace NUMINAMATH_CALUDE_drone_image_trees_l1607_160754

theorem drone_image_trees (T : ℕ) (h1 : T ≥ 100) (h2 : T ≥ 90) (h3 : T ≥ 82) : 
  (T - 82) + (T - 82) = 26 := by
sorry

end NUMINAMATH_CALUDE_drone_image_trees_l1607_160754


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_divisible_by_360_l1607_160705

theorem smallest_k_for_sum_of_squares_divisible_by_360 :
  ∀ k : ℕ, k > 0 → (k * (k + 1) * (2 * k + 1)) % 2160 = 0 → k ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_divisible_by_360_l1607_160705


namespace NUMINAMATH_CALUDE_sum_is_composite_l1607_160742

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 - a*b + b^2 = c^2 - c*d + d^2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l1607_160742


namespace NUMINAMATH_CALUDE_max_gold_coins_l1607_160745

theorem max_gold_coins (n : ℕ) : n < 150 ∧ ∃ k : ℕ, n = 13 * k + 3 → n ≤ 146 :=
by
  sorry

end NUMINAMATH_CALUDE_max_gold_coins_l1607_160745


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1607_160768

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

theorem sufficient_but_not_necessary 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : subset m β) :
  (∃ (h : parallel α β), ∀ (l m : Line), perpendicular l α → subset m β → perpendicularLines l m) ∧
  (∃ (l m : Line) (α β : Plane), perpendicular l α ∧ subset m β ∧ perpendicularLines l m ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1607_160768


namespace NUMINAMATH_CALUDE_problem_statement_l1607_160763

theorem problem_statement (x y : ℝ) : 
  x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) →
  y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) →
  x^4 + y^4 + (x + y)^4 = 1152 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1607_160763


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l1607_160700

theorem product_from_gcd_lcm (a b : ℤ) : 
  Int.gcd a b = 8 → Int.lcm a b = 48 → a * b = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l1607_160700


namespace NUMINAMATH_CALUDE_real_part_of_x_l1607_160712

-- Define the variables and their types
variable (x : ℂ) -- x is a complex number
variable (y z : ℝ) -- y and z are real numbers
variable (p q : ℕ) -- p and q are natural numbers (we'll define them as prime later)
variable (n m : ℕ) -- n and m are non-negative integers
variable (k : ℕ) -- k is a natural number (we'll define it as odd prime later)

-- Define the conditions
axiom p_prime : Nat.Prime p
axiom q_prime : Nat.Prime q
axiom p_ne_q : p ≠ q
axiom k_odd_prime : Nat.Prime k ∧ k % 2 = 1
axiom least_p_q : ∀ p' q', Nat.Prime p' → Nat.Prime q' → p' ≠ q' → (p < p' ∨ q < q')

-- Define the specific values
axiom n_val : n = 2
axiom m_val : m = 3
axiom y_val : y = 5
axiom z_val : z = 10

-- Define the system of equations
axiom eq1 : x^n / (12 * ↑p * ↑q) = ↑k
axiom eq2 : x^m + y = z

-- Theorem to prove
theorem real_part_of_x :
  ∃ r : ℝ, (r = 6 * Real.sqrt 6 ∨ r = -6 * Real.sqrt 6) ∧ x.re = r :=
sorry

end NUMINAMATH_CALUDE_real_part_of_x_l1607_160712


namespace NUMINAMATH_CALUDE_exponential_inverse_existence_uniqueness_l1607_160748

theorem exponential_inverse_existence_uniqueness (a x : ℝ) (ha : 0 < a) (ha_neq : a ≠ 1) (hx : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by sorry

end NUMINAMATH_CALUDE_exponential_inverse_existence_uniqueness_l1607_160748


namespace NUMINAMATH_CALUDE_rice_price_reduction_l1607_160734

theorem rice_price_reduction (x : ℝ) (h : x > 0) :
  let original_amount := 30
  let price_reduction_factor := 0.75
  let new_amount := original_amount / price_reduction_factor
  new_amount = 40 := by
sorry

end NUMINAMATH_CALUDE_rice_price_reduction_l1607_160734


namespace NUMINAMATH_CALUDE_unique_equilateral_hyperbola_l1607_160736

/-- An equilateral hyperbola passing through (3, -1) with axes of symmetry on coordinate axes -/
def equilateral_hyperbola (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 = a ∧ (x = 3 ∧ y = -1)

/-- The unique value of 'a' for which the hyperbola is equilateral and passes through (3, -1) -/
theorem unique_equilateral_hyperbola :
  ∃! a : ℝ, equilateral_hyperbola a ∧ a = 8 := by sorry

end NUMINAMATH_CALUDE_unique_equilateral_hyperbola_l1607_160736


namespace NUMINAMATH_CALUDE_jasons_punch_problem_l1607_160709

/-- Represents the recipe for Jason's punch -/
structure PunchRecipe where
  water : ℝ
  lemon_juice : ℝ
  sugar : ℝ

/-- Represents the actual amounts used in Jason's punch -/
structure PunchIngredients where
  water : ℝ
  lemon_juice : ℝ
  sugar : ℝ

/-- The recipe ratios are correct -/
def recipe_ratios_correct (recipe : PunchRecipe) : Prop :=
  recipe.water = 5 * recipe.lemon_juice ∧ 
  recipe.lemon_juice = 3 * recipe.sugar

/-- The actual ingredients follow the recipe ratios -/
def ingredients_follow_recipe (recipe : PunchRecipe) (ingredients : PunchIngredients) : Prop :=
  ingredients.water / ingredients.lemon_juice = recipe.water / recipe.lemon_juice ∧
  ingredients.lemon_juice / ingredients.sugar = recipe.lemon_juice / recipe.sugar

/-- Jason's punch problem -/
theorem jasons_punch_problem (recipe : PunchRecipe) (ingredients : PunchIngredients) :
  recipe_ratios_correct recipe →
  ingredients_follow_recipe recipe ingredients →
  ingredients.lemon_juice = 5 →
  ingredients.water = 25 := by
  sorry

end NUMINAMATH_CALUDE_jasons_punch_problem_l1607_160709


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1607_160762

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (5 * x + 2) → y = 2 → x = -3/10 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1607_160762


namespace NUMINAMATH_CALUDE_graph6_triangle_or_independent_set_l1607_160781

/-- A simple graph with 6 vertices -/
structure Graph6 where
  vertices : Finset (Fin 6)
  edges : Set (Fin 6 × Fin 6)
  symmetry : ∀ (a b : Fin 6), (a, b) ∈ edges → (b, a) ∈ edges
  irreflexive : ∀ (a : Fin 6), (a, a) ∉ edges

/-- A triangle in a graph -/
def HasTriangle (G : Graph6) : Prop :=
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ G.edges ∧ (b, c) ∈ G.edges ∧ (c, a) ∈ G.edges

/-- An independent set of size 3 in a graph -/
def HasIndependentSet3 (G : Graph6) : Prop :=
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∉ G.edges ∧ (b, c) ∉ G.edges ∧ (c, a) ∉ G.edges

/-- The main theorem -/
theorem graph6_triangle_or_independent_set (G : Graph6) :
  HasTriangle G ∨ HasIndependentSet3 G :=
sorry

end NUMINAMATH_CALUDE_graph6_triangle_or_independent_set_l1607_160781


namespace NUMINAMATH_CALUDE_zero_not_in_empty_set_l1607_160739

theorem zero_not_in_empty_set : 0 ∉ (∅ : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_empty_set_l1607_160739


namespace NUMINAMATH_CALUDE_max_b_minus_a_l1607_160766

theorem max_b_minus_a (a b : ℝ) (ha : a < 0)
  (h : ∀ x : ℝ, (3 * x^2 + a) * (2 * x + b) ≥ 0) :
  ∃ (max : ℝ), max = 1/3 ∧ b - a ≤ max ∧
  ∀ (a' b' : ℝ), a' < 0 → (∀ x : ℝ, (3 * x^2 + a') * (2 * x + b') ≥ 0) →
  b' - a' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_b_minus_a_l1607_160766


namespace NUMINAMATH_CALUDE_simplify_expression_range_of_values_find_values_l1607_160702

-- Question 1
theorem simplify_expression (a : ℝ) (h : 3 ≤ a ∧ a ≤ 7) :
  Real.sqrt ((3 - a)^2) + Real.sqrt ((a - 7)^2) = 4 :=
sorry

-- Question 2
theorem range_of_values (a : ℝ) :
  Real.sqrt ((a - 1)^2) + Real.sqrt ((a - 6)^2) = 5 ↔ 1 ≤ a ∧ a ≤ 6 :=
sorry

-- Question 3
theorem find_values (a : ℝ) :
  Real.sqrt ((a + 1)^2) + Real.sqrt ((a - 3)^2) = 6 ↔ a = -2 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_range_of_values_find_values_l1607_160702


namespace NUMINAMATH_CALUDE_correct_operation_l1607_160738

theorem correct_operation (a b : ℝ) : 3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1607_160738


namespace NUMINAMATH_CALUDE_prob_objects_meet_l1607_160794

/-- The number of steps required for objects to meet -/
def stepsToMeet : ℕ := 9

/-- The possible x-coordinates of meeting points -/
def meetingPoints : List ℕ := [0, 2, 4, 6, 8]

/-- Calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculate the number of paths to a meeting point for each object -/
def pathsToPoint (i : ℕ) : ℕ × ℕ :=
  (binomial stepsToMeet i, binomial stepsToMeet (i + 1))

/-- Calculate the probability of meeting at a specific point -/
def probMeetAtPoint (i : ℕ) : ℚ :=
  let (a, b) := pathsToPoint i
  (a * b : ℚ) / (2^(2 * stepsToMeet) : ℚ)

/-- The main theorem: probability of objects meeting -/
theorem prob_objects_meet :
  (meetingPoints.map probMeetAtPoint).sum = 207 / 262144 := by sorry

end NUMINAMATH_CALUDE_prob_objects_meet_l1607_160794


namespace NUMINAMATH_CALUDE_distinct_power_tower_values_l1607_160756

def power_tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (power_tower base n)

def parenthesized_expressions (base : ℕ) (height : ℕ) : Finset ℕ :=
  sorry

theorem distinct_power_tower_values :
  (parenthesized_expressions 3 4).card = 5 :=
sorry

end NUMINAMATH_CALUDE_distinct_power_tower_values_l1607_160756


namespace NUMINAMATH_CALUDE_ap_terms_count_l1607_160770

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  n % 2 = 0 ∧ 
  (n / 2 : ℚ) * (2 * a + (n - 2) * d) = 32 ∧ 
  (n / 2 : ℚ) * (2 * a + 2 * d + (n - 2) * d) = 40 ∧ 
  a + (n - 1) * d - a = 8 → 
  n = 16 := by sorry

end NUMINAMATH_CALUDE_ap_terms_count_l1607_160770


namespace NUMINAMATH_CALUDE_bug_crawl_theorem_l1607_160759

def bug_movements : List Int := [5, -3, 10, -8, -6, 12, -10]

theorem bug_crawl_theorem :
  (List.sum bug_movements = 0) ∧
  (List.sum (List.map Int.natAbs bug_movements) = 54) := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_theorem_l1607_160759


namespace NUMINAMATH_CALUDE_unique_natural_number_solution_l1607_160752

theorem unique_natural_number_solution (n p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → (1 : ℚ) / n = 1 / p + 1 / q + 1 / (p * q) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_solution_l1607_160752


namespace NUMINAMATH_CALUDE_sum_of_relatively_prime_integers_l1607_160785

theorem sum_of_relatively_prime_integers (n : ℤ) (h : n ≥ 7) :
  ∃ a b : ℤ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_relatively_prime_integers_l1607_160785


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1607_160730

/-- Given two-digit numbers EH, OY, AY, and OH, where EH is four times OY 
    and AY is four times OH, prove that their sum is 150. -/
theorem sum_of_four_numbers (EH OY AY OH : ℕ) : 
  (10 ≤ EH) ∧ (EH < 100) ∧
  (10 ≤ OY) ∧ (OY < 100) ∧
  (10 ≤ AY) ∧ (AY < 100) ∧
  (10 ≤ OH) ∧ (OH < 100) ∧
  (EH = 4 * OY) ∧
  (AY = 4 * OH) →
  EH + OY + AY + OH = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1607_160730


namespace NUMINAMATH_CALUDE_bean_ratio_l1607_160792

/-- Given a jar of beans with the following properties:
  - There are 572 beans in total
  - One-fourth of the beans are red
  - Half of the remaining beans after removing red are green
  - There are 143 green beans
  This theorem proves that the ratio of white beans to the remaining beans
  after removing red beans is 1:2. -/
theorem bean_ratio (total : ℕ) (red : ℕ) (green : ℕ) (white : ℕ) : 
  total = 572 →
  red = total / 4 →
  green = (total - red) / 2 →
  green = 143 →
  white = total - red - green →
  (white : ℚ) / (total - red - green : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bean_ratio_l1607_160792


namespace NUMINAMATH_CALUDE_simplify_expression_l1607_160760

theorem simplify_expression (x : ℝ) : 120*x - 72*x + 15*x - 9*x = 54*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1607_160760


namespace NUMINAMATH_CALUDE_hash_difference_l1607_160798

def hash (x y : ℝ) : ℝ := x * y - 3 * x

theorem hash_difference : (hash 6 4) - (hash 4 6) = -6 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l1607_160798


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l1607_160750

/-- Circle C3 centered at (8, 0) with radius 5 -/
def C3 (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 25

/-- Circle C4 centered at (-10, 0) with radius 7 -/
def C4 (x y : ℝ) : Prop := (x + 10)^2 + y^2 = 49

/-- Point R on circle C3 -/
def R : ℝ × ℝ := sorry

/-- Point S on circle C4 -/
def S : ℝ × ℝ := sorry

/-- The shortest line segment RS is tangent to C3 at R and C4 at S -/
theorem shortest_tangent_length : 
  C3 R.1 R.2 ∧ C4 S.1 S.2 → 
  ∃ (R S : ℝ × ℝ), C3 R.1 R.2 ∧ C4 S.1 S.2 ∧ 
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l1607_160750


namespace NUMINAMATH_CALUDE_mix_alcohol_solutions_l1607_160723

/-- Represents an alcohol solution with a given volume and concentration -/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Proves that mixing two alcohol solutions results in the desired solution -/
theorem mix_alcohol_solutions
  (solution_a : AlcoholSolution)
  (solution_b : AlcoholSolution)
  (mixed_solution : AlcoholSolution)
  (h1 : solution_a.volume = 10.5)
  (h2 : solution_a.concentration = 0.75)
  (h3 : solution_b.volume = 7.5)
  (h4 : solution_b.concentration = 0.15)
  (h5 : mixed_solution.volume = 18)
  (h6 : mixed_solution.concentration = 0.5)
  : solution_a.volume * solution_a.concentration + solution_b.volume * solution_b.concentration
    = mixed_solution.volume * mixed_solution.concentration :=
by
  sorry

#check mix_alcohol_solutions

end NUMINAMATH_CALUDE_mix_alcohol_solutions_l1607_160723


namespace NUMINAMATH_CALUDE_clothing_store_gross_profit_l1607_160751

-- Define the purchase price
def purchase_price : ℚ := 81

-- Define the initial markup percentage
def markup_percentage : ℚ := 1/4

-- Define the price decrease percentage
def price_decrease_percentage : ℚ := 1/5

-- Define the function to calculate the initial selling price
def initial_selling_price (purchase_price : ℚ) (markup_percentage : ℚ) : ℚ :=
  purchase_price / (1 - markup_percentage)

-- Define the function to calculate the new selling price after discount
def new_selling_price (initial_price : ℚ) (decrease_percentage : ℚ) : ℚ :=
  initial_price * (1 - decrease_percentage)

-- Define the function to calculate the gross profit
def gross_profit (new_price : ℚ) (purchase_price : ℚ) : ℚ :=
  new_price - purchase_price

-- Theorem statement
theorem clothing_store_gross_profit :
  let initial_price := initial_selling_price purchase_price markup_percentage
  let new_price := new_selling_price initial_price price_decrease_percentage
  gross_profit new_price purchase_price = 27/5 := by sorry

end NUMINAMATH_CALUDE_clothing_store_gross_profit_l1607_160751


namespace NUMINAMATH_CALUDE_polynomial_equality_l1607_160790

theorem polynomial_equality (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1607_160790


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l1607_160701

/-- The number of zeros between the decimal point and the first non-zero digit when 7/8000 is written as a decimal -/
def zeros_before_first_nonzero : ℕ :=
  3

/-- The fraction we're considering -/
def fraction : ℚ :=
  7 / 8000

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 3 ∧ fraction = 7 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l1607_160701


namespace NUMINAMATH_CALUDE_max_value_product_l1607_160778

theorem max_value_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hsum : 5 * a + 3 * b < 90) :
  a * b * (90 - 5 * a - 3 * b) ≤ 1800 := by
sorry

end NUMINAMATH_CALUDE_max_value_product_l1607_160778


namespace NUMINAMATH_CALUDE_days_at_sisters_house_l1607_160776

/-- Calculates the number of days spent at the sister's house during a vacation --/
theorem days_at_sisters_house (total_vacation_days : ℕ) 
  (days_to_grandparents days_at_grandparents days_to_brother days_at_brother 
   days_to_sister days_from_sister : ℕ) : 
  total_vacation_days = 21 →
  days_to_grandparents = 1 →
  days_at_grandparents = 5 →
  days_to_brother = 1 →
  days_at_brother = 5 →
  days_to_sister = 2 →
  days_from_sister = 2 →
  total_vacation_days - (days_to_grandparents + days_at_grandparents + 
    days_to_brother + days_at_brother + days_to_sister + days_from_sister) = 5 := by
  sorry

end NUMINAMATH_CALUDE_days_at_sisters_house_l1607_160776


namespace NUMINAMATH_CALUDE_tenth_even_term_is_92_l1607_160722

def arithmetic_sequence (n : ℕ) : ℤ := 2 + (n - 1) * 5

def is_even (z : ℤ) : Prop := ∃ k : ℤ, z = 2 * k

def nth_even_term (n : ℕ) : ℕ := 2 * n - 1

theorem tenth_even_term_is_92 :
  arithmetic_sequence (nth_even_term 10) = 92 :=
sorry

end NUMINAMATH_CALUDE_tenth_even_term_is_92_l1607_160722


namespace NUMINAMATH_CALUDE_triangle_problem_l1607_160787

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2 →
  c = Real.sqrt 7 →
  a + b = 5 →
  C = π / 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1607_160787


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1607_160729

theorem modulus_of_complex_fraction :
  let z : ℂ := (-3 + Complex.I) / (2 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1607_160729


namespace NUMINAMATH_CALUDE_school_basketballs_l1607_160772

/-- The number of classes that received basketballs -/
def num_classes : ℕ := 7

/-- The number of basketballs each class received -/
def basketballs_per_class : ℕ := 7

/-- The total number of basketballs bought by the school -/
def total_basketballs : ℕ := num_classes * basketballs_per_class

theorem school_basketballs : total_basketballs = 49 := by
  sorry

end NUMINAMATH_CALUDE_school_basketballs_l1607_160772


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1607_160711

theorem complex_equation_solution (x : ℝ) :
  (1 - 2*Complex.I) * (x + Complex.I) = 4 - 3*Complex.I → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1607_160711


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1607_160741

theorem arithmetic_expression_equality : 3 + 15 / 3 - 2^2 + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1607_160741


namespace NUMINAMATH_CALUDE_map_scale_conversion_l1607_160755

/-- Given a map scale where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale_conversion (scale : ℝ → ℝ) (h1 : scale 15 = 90) : scale 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l1607_160755
