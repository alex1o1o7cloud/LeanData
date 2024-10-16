import Mathlib

namespace NUMINAMATH_CALUDE_japanese_students_fraction_l3156_315639

theorem japanese_students_fraction (J : ℕ) : 
  let S := 2 * J
  let seniors_japanese := (3 * S) / 8
  let juniors_japanese := J / 4
  let total_students := S + J
  let total_japanese := seniors_japanese + juniors_japanese
  (total_japanese : ℚ) / total_students = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l3156_315639


namespace NUMINAMATH_CALUDE_intersection_M_N_l3156_315632

-- Define the sets M and N
def M : Set ℝ := {x | x / (x - 1) > 0}
def N : Set ℝ := {x | ∃ y, y * y = x}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3156_315632


namespace NUMINAMATH_CALUDE_intersection_M_N_l3156_315689

def M : Set ℝ := {x | x^2 - x - 2 = 0}
def N : Set ℝ := {-1, 0}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3156_315689


namespace NUMINAMATH_CALUDE_jessie_friends_l3156_315619

/-- The number of friends Jessie invited -/
def num_friends (total_muffins : ℕ) (muffins_per_person : ℕ) : ℕ :=
  total_muffins / muffins_per_person - 1

/-- Theorem stating that Jessie invited 4 friends -/
theorem jessie_friends : num_friends 20 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jessie_friends_l3156_315619


namespace NUMINAMATH_CALUDE_train_length_l3156_315697

/-- Given a train with speed 108 km/hr passing a tree in 9 seconds, its length is 270 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 108 → time = 9 → length = speed * (1000 / 3600) * time → length = 270 := by sorry

end NUMINAMATH_CALUDE_train_length_l3156_315697


namespace NUMINAMATH_CALUDE_garage_sale_theorem_l3156_315630

def garage_sale_problem (treadmill_price : ℝ) (chest_price : ℝ) (tv_price : ℝ) (total_sales : ℝ) : Prop :=
  treadmill_price = 100 ∧
  chest_price = treadmill_price / 2 ∧
  tv_price = 3 * treadmill_price ∧
  (treadmill_price + chest_price + tv_price) / total_sales = 0.75 ∧
  total_sales = 600

theorem garage_sale_theorem :
  ∃ (treadmill_price chest_price tv_price total_sales : ℝ),
    garage_sale_problem treadmill_price chest_price tv_price total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_garage_sale_theorem_l3156_315630


namespace NUMINAMATH_CALUDE_inradius_properties_l3156_315621

/-- Properties of a triangle ABC --/
structure Triangle where
  /-- Inradius of the triangle --/
  r : ℝ
  /-- Circumradius of the triangle --/
  R : ℝ
  /-- Semiperimeter of the triangle --/
  s : ℝ
  /-- Angle A of the triangle --/
  A : ℝ
  /-- Angle B of the triangle --/
  B : ℝ
  /-- Angle C of the triangle --/
  C : ℝ
  /-- Exradius opposite to angle A --/
  r_a : ℝ
  /-- Exradius opposite to angle B --/
  r_b : ℝ
  /-- Exradius opposite to angle C --/
  r_c : ℝ

/-- Theorem: Properties of inradius in a triangle --/
theorem inradius_properties (t : Triangle) :
  (t.r = 4 * t.R * Real.sin (t.A / 2) * Real.sin (t.B / 2) * Real.sin (t.C / 2)) ∧
  (t.r = t.s * Real.tan (t.A / 2) * Real.tan (t.B / 2) * Real.tan (t.C / 2)) ∧
  (t.r = t.R * (Real.cos t.A + Real.cos t.B + Real.cos t.C - 1)) ∧
  (t.r = t.r_a + t.r_b + t.r_c - 4 * t.R) := by
  sorry

end NUMINAMATH_CALUDE_inradius_properties_l3156_315621


namespace NUMINAMATH_CALUDE_inequality_solution_set_min_perimeter_rectangle_min_perimeter_achieved_l3156_315628

-- Problem 1: Inequality solution set
theorem inequality_solution_set (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 ↔ x * (2 * x - 3) - 6 ≤ x := by sorry

-- Problem 2: Minimum perimeter of rectangle
theorem min_perimeter_rectangle (l w : ℝ) (h_area : l * w = 16) (h_positive : l > 0 ∧ w > 0) :
  2 * (l + w) ≥ 16 := by sorry

theorem min_perimeter_achieved (l w : ℝ) (h_area : l * w = 16) (h_positive : l > 0 ∧ w > 0) :
  2 * (l + w) = 16 ↔ l = 4 ∧ w = 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_min_perimeter_rectangle_min_perimeter_achieved_l3156_315628


namespace NUMINAMATH_CALUDE_negations_universal_and_true_l3156_315694

-- Define the propositions
def prop_A (x : ℝ) := x^2 - x + 1/4 < 0
def prop_C (x : ℝ) := x^2 + 2*x + 2 ≤ 0
def prop_D (x : ℝ) := x^3 + 1 = 0

-- Define the negations
def neg_A := ∀ x : ℝ, ¬(prop_A x)
def neg_C := ∀ x : ℝ, ¬(prop_C x)
def neg_D := ∀ x : ℝ, ¬(prop_D x)

-- Theorem statement
theorem negations_universal_and_true :
  (neg_A ∧ neg_C) ∧ 
  (∃ x : ℝ, prop_D x) :=
sorry

end NUMINAMATH_CALUDE_negations_universal_and_true_l3156_315694


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l3156_315655

/-- Given a cube of metal weighing 6 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 48 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight : ℝ) (h1 : weight = 6) :
  let new_weight := weight * (2^3)
  new_weight = 48 := by sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l3156_315655


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3156_315679

/-- Given a function f: ℝ → ℝ with a tangent line y=-x+8 at x=5, prove f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x, f 5 + (deriv f 5) * (x - 5) = -x + 8) : 
  f 5 + deriv f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3156_315679


namespace NUMINAMATH_CALUDE_houses_with_both_pets_l3156_315615

theorem houses_with_both_pets (total : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h_total : total = 60) 
  (h_dogs : dogs = 40) 
  (h_cats : cats = 30) : 
  dogs + cats - total = 10 := by
  sorry

end NUMINAMATH_CALUDE_houses_with_both_pets_l3156_315615


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3156_315650

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3156_315650


namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3156_315656

theorem fewer_bees_than_flowers (flowers : ℕ) (bees : ℕ) 
  (h1 : flowers = 5) (h2 : bees = 3) : flowers - bees = 2 := by
  sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3156_315656


namespace NUMINAMATH_CALUDE_calculate_expression_l3156_315653

theorem calculate_expression : |-3| - 2 * Real.tan (π / 4) + (-1) ^ 2023 - (Real.sqrt 3 - Real.pi) ^ 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3156_315653


namespace NUMINAMATH_CALUDE_total_volume_of_boxes_l3156_315614

-- Define the number of boxes
def num_boxes : ℕ := 4

-- Define the edge length of each box in feet
def edge_length : ℝ := 6

-- Define the volume of a single box
def single_box_volume : ℝ := edge_length ^ 3

-- Theorem stating the total volume of all boxes
theorem total_volume_of_boxes : single_box_volume * num_boxes = 864 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_boxes_l3156_315614


namespace NUMINAMATH_CALUDE_tv_screen_width_l3156_315605

/-- Given a rectangular TV screen with area 21 square feet and height 7 feet, its width is 3 feet. -/
theorem tv_screen_width (area : ℝ) (height : ℝ) (width : ℝ) : 
  area = 21 → height = 7 → area = width * height → width = 3 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_width_l3156_315605


namespace NUMINAMATH_CALUDE_total_cost_is_10_79_l3156_315686

/-- The total cost of peppers purchased by Dale's Vegetarian Restaurant -/
def total_cost_peppers : ℝ :=
  let green_pepper_weight : ℝ := 2.8333333333333335
  let green_pepper_price : ℝ := 1.20
  let red_pepper_weight : ℝ := 3.254
  let red_pepper_price : ℝ := 1.35
  let yellow_pepper_weight : ℝ := 1.375
  let yellow_pepper_price : ℝ := 1.50
  let orange_pepper_weight : ℝ := 0.567
  let orange_pepper_price : ℝ := 1.65
  green_pepper_weight * green_pepper_price +
  red_pepper_weight * red_pepper_price +
  yellow_pepper_weight * yellow_pepper_price +
  orange_pepper_weight * orange_pepper_price

/-- Theorem stating that the total cost of peppers is $10.79 -/
theorem total_cost_is_10_79 : total_cost_peppers = 10.79 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_10_79_l3156_315686


namespace NUMINAMATH_CALUDE_factorial_ratio_l3156_315608

theorem factorial_ratio (n : ℕ) (h : n > 0) : (n.factorial) / ((n-1).factorial) = n := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3156_315608


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l3156_315688

/-- Calculates the total cost of fruits with a discount -/
def totalCostWithDiscount (cherryPrice olivePrice : ℚ) (bagCount : ℕ) (discountPercentage : ℚ) : ℚ :=
  let discountFactor : ℚ := 1 - discountPercentage / 100
  let discountedCherryPrice : ℚ := cherryPrice * discountFactor
  let discountedOlivePrice : ℚ := olivePrice * discountFactor
  (discountedCherryPrice + discountedOlivePrice) * bagCount

/-- Proves that the total cost for 50 bags each of cherries and olives with a 10% discount is $540 -/
theorem fruit_purchase_cost : 
  totalCostWithDiscount 5 7 50 10 = 540 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_cost_l3156_315688


namespace NUMINAMATH_CALUDE_max_xy_on_line_AB_l3156_315652

/-- Given points A(3,0) and B(0,4), prove that the maximum value of xy for any point P(x,y) on the line AB is 3. -/
theorem max_xy_on_line_AB :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  let line_AB (x : ℝ) := -4/3 * x + 4
  ∀ x y : ℝ, y = line_AB x → x * y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_on_line_AB_l3156_315652


namespace NUMINAMATH_CALUDE_alex_lead_after_even_l3156_315643

/-- Represents the race between Alex and Max -/
structure Race where
  total_length : ℕ
  initial_even : ℕ
  max_ahead : ℕ
  alex_final_ahead : ℕ
  remaining : ℕ

/-- Calculates the distance Alex got ahead of Max after they were even -/
def alex_initial_lead (r : Race) : ℕ :=
  r.total_length - r.remaining - r.initial_even - (r.max_ahead + r.alex_final_ahead)

/-- The theorem stating that Alex got ahead of Max by 300 feet after they were even -/
theorem alex_lead_after_even (r : Race) 
  (h1 : r.total_length = 5000)
  (h2 : r.initial_even = 200)
  (h3 : r.max_ahead = 170)
  (h4 : r.alex_final_ahead = 440)
  (h5 : r.remaining = 3890) :
  alex_initial_lead r = 300 := by
  sorry

#eval alex_initial_lead { total_length := 5000, initial_even := 200, max_ahead := 170, alex_final_ahead := 440, remaining := 3890 }

end NUMINAMATH_CALUDE_alex_lead_after_even_l3156_315643


namespace NUMINAMATH_CALUDE_person_a_speed_l3156_315672

theorem person_a_speed (v_a v_b : ℝ) : 
  v_a > v_b →
  8 * (v_a + v_b) = 6 * (v_a + v_b + 4) →
  6 * ((v_a + 2) - (v_b + 2)) = 6 →
  v_a = 6.5 := by
sorry

end NUMINAMATH_CALUDE_person_a_speed_l3156_315672


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3156_315636

def polynomial (x : ℤ) : ℤ := x^4 + 4*x^3 - x^2 + 3*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = possible_roots := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3156_315636


namespace NUMINAMATH_CALUDE_fraction_division_l3156_315635

theorem fraction_division (x : ℚ) : 
  (37 + 1/2 : ℚ) = 450 * x → x = 1/12 ∧ (37 + 1/2 : ℚ) / x = 450 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3156_315635


namespace NUMINAMATH_CALUDE_trader_profit_equation_l3156_315691

/-- The trader's profit after a week of sales -/
def trader_profit : ℝ := 960

/-- The amount of donations received -/
def donations : ℝ := 310

/-- The trader's goal amount -/
def goal : ℝ := 610

/-- The amount above the goal -/
def above_goal : ℝ := 180

theorem trader_profit_equation :
  trader_profit / 2 + donations = goal + above_goal :=
by sorry

end NUMINAMATH_CALUDE_trader_profit_equation_l3156_315691


namespace NUMINAMATH_CALUDE_greatest_constant_right_triangle_l3156_315641

theorem greatest_constant_right_triangle (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_positive : c > 0) :
  ∀ N : ℝ, (∀ a b c : ℝ, c > 0 → a^2 + b^2 = c^2 → (a^2 + b^2 - c^2) / c^2 > N) → N ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_constant_right_triangle_l3156_315641


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l3156_315654

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (u v : ℝ × ℝ) : ℝ :=
  |u.1 * v.2 - u.2 * v.1|

theorem parallelogram_area_example : 
  let u : ℝ × ℝ := (4, 7)
  let z : ℝ × ℝ := (-6, 3)
  parallelogramArea u z = 54 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l3156_315654


namespace NUMINAMATH_CALUDE_trig_difference_equals_sqrt_three_l3156_315609

-- Define the problem
theorem trig_difference_equals_sqrt_three :
  (1 / Real.tan (20 * π / 180)) - (1 / Real.cos (10 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_difference_equals_sqrt_three_l3156_315609


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3156_315618

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3156_315618


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_990_l3156_315629

theorem sum_of_largest_and_smallest_prime_factors_of_990 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 990 ∧ largest ∣ 990 ∧
    (∀ p : ℕ, p.Prime → p ∣ 990 → smallest ≤ p ∧ p ≤ largest) ∧
    smallest + largest = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_990_l3156_315629


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3156_315669

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a3 : a 3 = 7) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3156_315669


namespace NUMINAMATH_CALUDE_area_with_holes_formula_l3156_315626

/-- The area of a rectangle with holes -/
def area_with_holes (x : ℝ) : ℝ :=
  let large_rectangle_area := (x + 8) * (x + 6)
  let hole_area := (2 * x - 4) * (x - 3)
  let total_hole_area := 2 * hole_area
  large_rectangle_area - total_hole_area

/-- Theorem: The area of the rectangle with holes is equal to -3x^2 + 34x + 24 -/
theorem area_with_holes_formula (x : ℝ) :
  area_with_holes x = -3 * x^2 + 34 * x + 24 := by
  sorry

#check area_with_holes_formula

end NUMINAMATH_CALUDE_area_with_holes_formula_l3156_315626


namespace NUMINAMATH_CALUDE_heather_walking_distance_l3156_315666

theorem heather_walking_distance (car_to_entrance : ℝ) (entrance_to_rides : ℝ) (rides_to_car : ℝ) 
  (h1 : car_to_entrance = 0.33)
  (h2 : entrance_to_rides = 0.33)
  (h3 : rides_to_car = 0.08) :
  car_to_entrance + entrance_to_rides + rides_to_car = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_heather_walking_distance_l3156_315666


namespace NUMINAMATH_CALUDE_new_average_is_44_l3156_315682

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  lastInningRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score after a given number of innings -/
def calculateAverage (performance : BatsmanPerformance) : Nat :=
  (performance.totalRuns + performance.lastInningRuns) / (performance.innings + 1)

/-- Theorem: Given the specific performance, prove the new average is 44 -/
theorem new_average_is_44 (performance : BatsmanPerformance)
  (h1 : performance.innings = 16)
  (h2 : performance.lastInningRuns = 92)
  (h3 : performance.averageIncrease = 3)
  (h4 : calculateAverage performance = calculateAverage performance - performance.averageIncrease + 3) :
  calculateAverage performance = 44 := by
  sorry

end NUMINAMATH_CALUDE_new_average_is_44_l3156_315682


namespace NUMINAMATH_CALUDE_expanded_polynomial_has_four_nonzero_terms_l3156_315683

/-- The polynomial obtained from expanding (x+5)(3x^2+2x+4)-4(x^3-x^2+3x) -/
def expanded_polynomial (x : ℝ) : ℝ := -x^3 + 21*x^2 + 2*x + 20

/-- The number of nonzero terms in the expanded polynomial -/
def nonzero_term_count : ℕ := 4

theorem expanded_polynomial_has_four_nonzero_terms :
  (∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    ∀ x : ℝ, expanded_polynomial x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ a b c d e : ℝ, ∀ i j k l m : ℕ,
    (i ≠ j ∨ a = 0) ∧ (i ≠ k ∨ a = 0) ∧ (i ≠ l ∨ a = 0) ∧ (i ≠ m ∨ a = 0) ∧
    (j ≠ k ∨ b = 0) ∧ (j ≠ l ∨ b = 0) ∧ (j ≠ m ∨ b = 0) ∧
    (k ≠ l ∨ c = 0) ∧ (k ≠ m ∨ c = 0) ∧
    (l ≠ m ∨ d = 0) →
    (∀ x : ℝ, expanded_polynomial x = a*x^i + b*x^j + c*x^k + d*x^l + e*x^m) →
    e = 0) :=
by sorry

#check expanded_polynomial_has_four_nonzero_terms

end NUMINAMATH_CALUDE_expanded_polynomial_has_four_nonzero_terms_l3156_315683


namespace NUMINAMATH_CALUDE_composition_equation_solution_l3156_315612

theorem composition_equation_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 2)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h : f (g a) = 4) :
  a = -1/2 := by sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l3156_315612


namespace NUMINAMATH_CALUDE_hannah_final_pay_l3156_315696

def calculate_final_pay (hourly_rate : ℚ) (hours_worked : ℕ) (late_penalty : ℚ) 
  (times_late : ℕ) (federal_tax_rate : ℚ) (state_tax_rate : ℚ) (bonus_per_review : ℚ) 
  (qualifying_reviews : ℕ) (total_reviews : ℕ) : ℚ :=
  let gross_pay := hourly_rate * hours_worked
  let total_late_penalty := late_penalty * times_late
  let total_bonus := bonus_per_review * qualifying_reviews
  let adjusted_gross_pay := gross_pay - total_late_penalty + total_bonus
  let federal_tax := adjusted_gross_pay * federal_tax_rate
  let state_tax := adjusted_gross_pay * state_tax_rate
  let total_taxes := federal_tax + state_tax
  adjusted_gross_pay - total_taxes

theorem hannah_final_pay : 
  calculate_final_pay 30 18 5 3 (1/10) (1/20) 15 4 6 = 497.25 := by
  sorry

end NUMINAMATH_CALUDE_hannah_final_pay_l3156_315696


namespace NUMINAMATH_CALUDE_may_savings_l3156_315693

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (month 0)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l3156_315693


namespace NUMINAMATH_CALUDE_min_ratio_of_intersections_l3156_315690

theorem min_ratio_of_intersections (a : ℝ) (ha : a > 0) :
  let f (x : ℝ) := |Real.log x / Real.log 4|
  let x_A := (4 : ℝ) ^ (-a)
  let x_B := (4 : ℝ) ^ a
  let x_C := (4 : ℝ) ^ (-18 / (2 * a + 1))
  let x_D := (4 : ℝ) ^ (18 / (2 * a + 1))
  let m := |x_A - x_C|
  let n := |x_B - x_D|
  ∃ (a_min : ℝ), ∀ (a : ℝ), a > 0 → n / m ≥ 2^11 ∧ n / m = 2^11 ↔ a = a_min :=
sorry

end NUMINAMATH_CALUDE_min_ratio_of_intersections_l3156_315690


namespace NUMINAMATH_CALUDE_reflection_result_l3156_315648

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (-3, 7)

/-- Theorem: Reflecting the point (-3, 7) about y = -x gives (-7, 3) -/
theorem reflection_result :
  reflect_about_y_eq_neg_x original_center = (-7, 3) := by
  sorry

end NUMINAMATH_CALUDE_reflection_result_l3156_315648


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3156_315646

/-- The line equation x*cos(θ) + y*sin(θ) + a = 0 intersects the circle x^2 + y^2 = a^2 at exactly one point -/
theorem line_circle_intersection (θ a : ℝ) :
  ∃! p : ℝ × ℝ, 
    (p.1 * Real.cos θ + p.2 * Real.sin θ + a = 0) ∧ 
    (p.1^2 + p.2^2 = a^2) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3156_315646


namespace NUMINAMATH_CALUDE_weather_report_totals_l3156_315661

/-- Represents the different weather conditions --/
inductive WeatherCondition
  | Rain
  | Overcast
  | Sunshine
  | Thunderstorm

/-- Represents a day's weather report --/
structure DayReport where
  hours : WeatherCondition → Nat
  total_hours : (hours Rain) + (hours Overcast) + (hours Sunshine) + (hours Thunderstorm) = 12

/-- The weather report for the three days --/
def three_day_report : Vector DayReport 3 :=
  ⟨[
    { hours := λ c => match c with
                      | WeatherCondition.Rain => 6
                      | WeatherCondition.Overcast => 6
                      | _ => 0,
      total_hours := sorry },
    { hours := λ c => match c with
                      | WeatherCondition.Sunshine => 6
                      | WeatherCondition.Overcast => 4
                      | WeatherCondition.Rain => 2
                      | _ => 0,
      total_hours := sorry },
    { hours := λ c => match c with
                      | WeatherCondition.Thunderstorm => 2
                      | WeatherCondition.Overcast => 4
                      | WeatherCondition.Sunshine => 6
                      | _ => 0,
      total_hours := sorry }
  ], sorry⟩

/-- The total hours for each weather condition over the three days --/
def total_hours (c : WeatherCondition) : Nat :=
  (three_day_report.get 0).hours c + (three_day_report.get 1).hours c + (three_day_report.get 2).hours c

/-- The main theorem to prove --/
theorem weather_report_totals :
  total_hours WeatherCondition.Rain = 8 ∧
  total_hours WeatherCondition.Overcast = 14 ∧
  total_hours WeatherCondition.Sunshine = 12 ∧
  total_hours WeatherCondition.Thunderstorm = 2 := by
  sorry

end NUMINAMATH_CALUDE_weather_report_totals_l3156_315661


namespace NUMINAMATH_CALUDE_basketball_win_rate_l3156_315658

theorem basketball_win_rate (initial_wins initial_games remaining_games : ℕ) 
  (h1 : initial_wins = 45)
  (h2 : initial_games = 60)
  (h3 : remaining_games = 50) :
  ∃ (remaining_wins : ℕ), 
    (initial_wins + remaining_wins : ℚ) / (initial_games + remaining_games) = 3/4 ∧ 
    remaining_wins = 38 := by
  sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l3156_315658


namespace NUMINAMATH_CALUDE_cakes_served_at_lunch_l3156_315671

theorem cakes_served_at_lunch (total : ℕ) (dinner : ℕ) (yesterday : ℕ) 
  (h1 : total = 14) 
  (h2 : dinner = 6) 
  (h3 : yesterday = 3) : 
  total - dinner - yesterday = 5 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_at_lunch_l3156_315671


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l3156_315651

/-- The correct quadratic equation given the conditions of the problem -/
theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 5 ∨ x = 3 ∨ x = -6 ∨ x = -4) →
  (5 + 3 = -(b)) →
  ((-6) * (-4) = c) →
  (∀ x : ℝ, x^2 - 8*x + 24 = 0 ↔ x^2 + b*x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l3156_315651


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3156_315687

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (- (1/2) * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3156_315687


namespace NUMINAMATH_CALUDE_sum_of_areas_is_1168_l3156_315660

/-- A square in the xy-plane with three vertices having x-coordinates 2, 0, and 18 -/
structure SquareXY where
  vertices : Finset (ℝ × ℝ)
  is_square : vertices.card = 4
  x_coords : {2, 0, 18} ⊆ vertices.image Prod.fst

/-- The sum of all possible areas of the square -/
def sum_of_possible_areas (s : SquareXY) : ℝ :=
  -- Definition of sum_of_possible_areas
  sorry

/-- Theorem: The sum of all possible areas of the square is 1168 -/
theorem sum_of_areas_is_1168 (s : SquareXY) : sum_of_possible_areas s = 1168 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_areas_is_1168_l3156_315660


namespace NUMINAMATH_CALUDE_cube_root_difference_l3156_315602

theorem cube_root_difference (a b : ℝ) 
  (h1 : (a ^ (1/3) : ℝ) - (b ^ (1/3) : ℝ) = 12)
  (h2 : a * b = ((a + b + 8) / 6) ^ 3) :
  a - b = 468 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_difference_l3156_315602


namespace NUMINAMATH_CALUDE_sasha_remainder_l3156_315663

theorem sasha_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  b < 102 ∧ 
  d < 103 ∧ 
  a + d = 20 → 
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_sasha_remainder_l3156_315663


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3156_315645

theorem profit_percentage_calculation (selling_price profit : ℝ) 
  (h1 : selling_price = 900)
  (h2 : profit = 150) :
  (profit / (selling_price - profit)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3156_315645


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l3156_315649

-- Define the function f(x)
def f (x : ℝ) : ℝ := x - x^2 - x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y, -1/3 < x ∧ x < y ∧ y < 1 → f y < f x :=
by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l3156_315649


namespace NUMINAMATH_CALUDE_bench_arrangements_l3156_315624

theorem bench_arrangements (n : Nat) (h : n = 9) : Nat.factorial n = 362880 := by
  sorry

end NUMINAMATH_CALUDE_bench_arrangements_l3156_315624


namespace NUMINAMATH_CALUDE_forty_percent_value_l3156_315692

theorem forty_percent_value (x : ℝ) (h : 0.5 * x = 200) : 0.4 * x = 160 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_value_l3156_315692


namespace NUMINAMATH_CALUDE_janet_needs_775_l3156_315680

/-- The amount of additional money Janet needs to rent an apartment -/
def additional_money_needed (savings : ℕ) (monthly_rent : ℕ) (months_advance : ℕ) (deposit : ℕ) : ℕ :=
  (monthly_rent * months_advance + deposit) - savings

/-- Proof that Janet needs $775 more to rent the apartment -/
theorem janet_needs_775 :
  additional_money_needed 2225 1250 2 500 = 775 :=
by sorry

end NUMINAMATH_CALUDE_janet_needs_775_l3156_315680


namespace NUMINAMATH_CALUDE_roque_walking_time_l3156_315627

/-- The time it takes Roque to walk to work -/
def walking_time : ℝ := sorry

/-- The time it takes Roque to bike to work -/
def biking_time : ℝ := 1

/-- Number of times Roque walks to and from work per week -/
def walks_per_week : ℕ := 3

/-- Number of times Roque bikes to and from work per week -/
def bikes_per_week : ℕ := 2

/-- Total commuting time in a week -/
def total_commute_time : ℝ := 16

theorem roque_walking_time :
  walking_time = 2 :=
by
  have h1 : (2 * walking_time * walks_per_week) + (2 * biking_time * bikes_per_week) = total_commute_time := by sorry
  sorry

end NUMINAMATH_CALUDE_roque_walking_time_l3156_315627


namespace NUMINAMATH_CALUDE_one_eighth_of_2_40_l3156_315613

theorem one_eighth_of_2_40 (x : ℕ) : (1 / 8 : ℝ) * 2^40 = 2^x → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_2_40_l3156_315613


namespace NUMINAMATH_CALUDE_coin_position_determinable_l3156_315685

-- Define the coin values
def left_coin : ℕ := 10
def right_coin : ℕ := 15

-- Define the possible multipliers
def left_multipliers : List ℕ := [4, 10, 12, 26]
def right_multipliers : List ℕ := [7, 13, 21, 35]

-- Define a function to check if a number is even
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the possible configurations
structure Configuration :=
  (left_value : ℕ)
  (right_value : ℕ)
  (left_multiplier : ℕ)
  (right_multiplier : ℕ)

-- Define the theorem
theorem coin_position_determinable :
  ∀ (c : Configuration),
  c.left_value ∈ [left_coin, right_coin] ∧
  c.right_value ∈ [left_coin, right_coin] ∧
  c.left_value ≠ c.right_value ∧
  c.left_multiplier ∈ left_multipliers ∧
  c.right_multiplier ∈ right_multipliers →
  (is_even (c.left_value * c.left_multiplier + c.right_value * c.right_multiplier) ↔
   c.right_value = left_coin) :=
by sorry

end NUMINAMATH_CALUDE_coin_position_determinable_l3156_315685


namespace NUMINAMATH_CALUDE_sugar_spilled_calculation_l3156_315617

/-- The amount of sugar Pamela bought, in ounces -/
def original_amount : ℝ := 9.8

/-- The amount of sugar Pamela has left, in ounces -/
def amount_left : ℝ := 4.6

/-- The amount of sugar Pamela spilled, in ounces -/
def amount_spilled : ℝ := original_amount - amount_left

theorem sugar_spilled_calculation :
  amount_spilled = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_spilled_calculation_l3156_315617


namespace NUMINAMATH_CALUDE_test_results_l3156_315664

theorem test_results (total_students : ℕ) (correct_q1 : ℕ) (correct_q2 : ℕ) (not_taken : ℕ)
  (h1 : total_students = 40)
  (h2 : correct_q1 = 30)
  (h3 : correct_q2 = 29)
  (h4 : not_taken = 10)
  : (total_students - not_taken) = correct_q1 ∧ correct_q1 - 1 = correct_q2 := by
  sorry

end NUMINAMATH_CALUDE_test_results_l3156_315664


namespace NUMINAMATH_CALUDE_percentage_problem_l3156_315665

theorem percentage_problem : ∃ X : ℝ, 
  (X / 100 * 100 = (0.6 * 80 + 22)) ∧ 
  (X = 70) := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3156_315665


namespace NUMINAMATH_CALUDE_problem_solution_l3156_315606

theorem problem_solution (a b c x : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) ≠ 0)
  (h2 : a^2 / (a + b) = a^2 / (a + c) + 20)
  (h3 : b^2 / (b + c) = b^2 / (b + a) + 14)
  (h4 : c^2 / (c + a) = c^2 / (c + b) + x) :
  x = -34 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3156_315606


namespace NUMINAMATH_CALUDE_distribute_10_3_1_l3156_315610

/-- The number of ways to distribute n identical objects into k identical containers,
    with each container having at least m objects. -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical coins into 3 identical bags,
    with each bag having at least 1 coin. -/
theorem distribute_10_3_1 : distribute 10 3 1 = 8 := sorry

end NUMINAMATH_CALUDE_distribute_10_3_1_l3156_315610


namespace NUMINAMATH_CALUDE_expression_equals_25_l3156_315633

theorem expression_equals_25 : 
  (5^1010)^2 - (5^1008)^2 / (5^1009)^2 - (5^1007)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_25_l3156_315633


namespace NUMINAMATH_CALUDE_compute_expression_l3156_315659

theorem compute_expression : 18 * (200 / 3 + 50 / 6 + 16 / 18 + 2) = 1402 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3156_315659


namespace NUMINAMATH_CALUDE_min_notebooks_correct_l3156_315600

/-- The minimum number of notebooks needed to get a discount -/
def min_notebooks : ℕ := 18

/-- The cost of a single pen in yuan -/
def pen_cost : ℕ := 10

/-- The cost of a single notebook in yuan -/
def notebook_cost : ℕ := 4

/-- The number of pens Xiao Wei plans to buy -/
def num_pens : ℕ := 3

/-- The minimum spending amount to get a discount in yuan -/
def discount_threshold : ℕ := 100

/-- Theorem stating that min_notebooks is the minimum number of notebooks
    needed to get the discount -/
theorem min_notebooks_correct : 
  (num_pens * pen_cost + min_notebooks * notebook_cost ≥ discount_threshold) ∧ 
  (∀ n : ℕ, n < min_notebooks → num_pens * pen_cost + n * notebook_cost < discount_threshold) :=
sorry

end NUMINAMATH_CALUDE_min_notebooks_correct_l3156_315600


namespace NUMINAMATH_CALUDE_no_valid_polygon_pairs_l3156_315675

theorem no_valid_polygon_pairs : ¬∃ (r k : ℕ), 
  r > 2 ∧ k > 2 ∧ 
  (180 * r - 360) / (180 * k - 360) = 7 / 5 ∧
  ∃ (c : ℚ), c * r = k := by
  sorry

end NUMINAMATH_CALUDE_no_valid_polygon_pairs_l3156_315675


namespace NUMINAMATH_CALUDE_corporate_event_handshakes_eq_430_l3156_315607

/-- Represents the number of handshakes at a corporate event --/
def corporate_event_handshakes : ℕ :=
  let total_people : ℕ := 40
  let group_a_size : ℕ := 15
  let group_b_size : ℕ := 20
  let group_c_size : ℕ := 5
  let group_b_knowing_a : ℕ := 5
  let group_b_knowing_none : ℕ := 15

  let handshakes_a_b : ℕ := group_a_size * group_b_knowing_none
  let handshakes_within_b : ℕ := (group_b_knowing_none * (group_b_knowing_none - 1)) / 2
  let handshakes_b_c : ℕ := group_b_size * group_c_size

  handshakes_a_b + handshakes_within_b + handshakes_b_c

/-- Theorem stating that the number of handshakes at the corporate event is 430 --/
theorem corporate_event_handshakes_eq_430 : corporate_event_handshakes = 430 := by
  sorry

end NUMINAMATH_CALUDE_corporate_event_handshakes_eq_430_l3156_315607


namespace NUMINAMATH_CALUDE_jake_hourly_rate_l3156_315638

-- Define the problem parameters
def initial_debt : ℚ := 100
def payment : ℚ := 40
def work_hours : ℚ := 4

-- Define the theorem
theorem jake_hourly_rate :
  (initial_debt - payment) / work_hours = 15 := by
  sorry

end NUMINAMATH_CALUDE_jake_hourly_rate_l3156_315638


namespace NUMINAMATH_CALUDE_haircut_cost_per_year_l3156_315644

/-- Calculates the total amount spent on haircuts in a year given the specified conditions. -/
theorem haircut_cost_per_year
  (growth_rate : ℝ)
  (initial_length : ℝ)
  (cut_length : ℝ)
  (haircut_cost : ℝ)
  (tip_percentage : ℝ)
  (months_per_year : ℕ)
  (h1 : growth_rate = 1.5)
  (h2 : initial_length = 9)
  (h3 : cut_length = 6)
  (h4 : haircut_cost = 45)
  (h5 : tip_percentage = 0.2)
  (h6 : months_per_year = 12) :
  (haircut_cost * (1 + tip_percentage) * (months_per_year / ((initial_length - cut_length) / growth_rate))) = 324 :=
by sorry

end NUMINAMATH_CALUDE_haircut_cost_per_year_l3156_315644


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l3156_315674

def chris_schedule : ℕ := 5
def alex_schedule : ℕ := 6
def jordan_schedule : ℕ := 8
def taylor_schedule : ℕ := 9

theorem tutors_next_meeting :
  lcm (lcm (lcm chris_schedule alex_schedule) jordan_schedule) taylor_schedule = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l3156_315674


namespace NUMINAMATH_CALUDE_even_function_implies_m_zero_l3156_315603

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Theorem statement
theorem even_function_implies_m_zero (m : ℝ) :
  is_even (f m) → m = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_m_zero_l3156_315603


namespace NUMINAMATH_CALUDE_brothers_initial_money_l3156_315631

theorem brothers_initial_money 
  (michael_initial : ℝ) 
  (brother_final : ℝ) 
  (candy_cost : ℝ) :
  michael_initial = 42 →
  brother_final = 35 →
  candy_cost = 3 →
  ∃ (brother_initial : ℝ),
    brother_initial + michael_initial / 2 - candy_cost = brother_final ∧
    brother_initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_brothers_initial_money_l3156_315631


namespace NUMINAMATH_CALUDE_daisy_rose_dogs_pool_l3156_315670

/-- The number of legs/paws in a pool with humans and dogs -/
def legs_paws_in_pool (num_humans : ℕ) (num_dogs : ℕ) : ℕ :=
  num_humans * 2 + num_dogs * 4

/-- Theorem: The number of legs/paws in the pool with Daisy, Rose, and their 5 dogs is 24 -/
theorem daisy_rose_dogs_pool : legs_paws_in_pool 2 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_daisy_rose_dogs_pool_l3156_315670


namespace NUMINAMATH_CALUDE_twentieth_number_in_base8_l3156_315673

/-- Converts a decimal number to its base 8 representation -/
def toBase8 (n : ℕ) : ℕ := sorry

/-- Represents the sequence of numbers in base 8 -/
def base8Sequence : ℕ → ℕ := sorry

theorem twentieth_number_in_base8 :
  base8Sequence 20 = toBase8 24 := by sorry

end NUMINAMATH_CALUDE_twentieth_number_in_base8_l3156_315673


namespace NUMINAMATH_CALUDE_max_sum_under_constraint_l3156_315622

theorem max_sum_under_constraint (m n : ℤ) (h : 205 * m^2 + 409 * n^4 ≤ 20736) :
  m + n ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraint_l3156_315622


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3156_315698

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := {1 - 1 / n, 1 + 1 / n^2} ∪ Finset.range (n - 2)
  (Finset.sum set id) / n = 1 - 1 / n^2 + 1 / n^3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3156_315698


namespace NUMINAMATH_CALUDE_initial_marbles_l3156_315604

theorem initial_marbles (lost_marbles current_marbles : ℕ) 
  (h1 : lost_marbles = 7)
  (h2 : current_marbles = 9) :
  lost_marbles + current_marbles = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l3156_315604


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3156_315695

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence property
  a 3 + a 5 = 2 →                                       -- given condition
  a 4 = 1 :=                                            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3156_315695


namespace NUMINAMATH_CALUDE_parking_cost_savings_l3156_315647

-- Define the cost per week
def cost_per_week : ℕ := 10

-- Define the cost per month
def cost_per_month : ℕ := 35

-- Define the number of weeks in a year
def weeks_per_year : ℕ := 52

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem parking_cost_savings : 
  (weeks_per_year * cost_per_week) - (months_per_year * cost_per_month) = 100 := by
  sorry


end NUMINAMATH_CALUDE_parking_cost_savings_l3156_315647


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l3156_315625

theorem binomial_coefficient_19_13 (h1 : Nat.choose 18 11 = 31824)
                                   (h2 : Nat.choose 18 12 = 18564)
                                   (h3 : Nat.choose 20 13 = 77520) :
  Nat.choose 19 13 = 27132 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l3156_315625


namespace NUMINAMATH_CALUDE_max_k_for_cube_sum_inequality_l3156_315616

theorem max_k_for_cube_sum_inequality (m n : ℕ+) (h : m^3 + n^3 > (m + n)^2) :
  (∃ k : ℕ+, ∀ j : ℕ+, (m^3 + n^3 ≥ (m + n)^2 + j) → j ≤ k) ∧
  (∀ k : ℕ+, (∀ m n : ℕ+, m^3 + n^3 > (m + n)^2 → m^3 + n^3 ≥ (m + n)^2 + k) → k ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_cube_sum_inequality_l3156_315616


namespace NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l3156_315611

theorem x_power_2187_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^2187 - 1/x^2187 = Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l3156_315611


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3156_315634

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ  -- side EF
  b : ℕ  -- side DE and DF
  h : a = 2 * b  -- EF is twice the length of DE and DF

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircle of a triangle opposite to side EF -/
def excircle_EF (t : Triangle) : Excircle := sorry

/-- The excircles of a triangle opposite to sides DE and DF -/
def excircles_DE_DF (t : Triangle) : Excircle × Excircle := sorry

/-- Checks if two circles are internally tangent -/
def internally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Checks if two circles are externally tangent -/
def externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- The main theorem -/
theorem min_perimeter_triangle (t : Triangle) :
  let χ : Circle := incircle t
  let exc_EF : Excircle := excircle_EF t
  let (exc_DE, exc_DF) := excircles_DE_DF t
  (internally_tangent ⟨exc_EF.center, exc_EF.radius⟩ χ) ∧
  (externally_tangent ⟨exc_DE.center, exc_DE.radius⟩ χ) ∧
  (externally_tangent ⟨exc_DF.center, exc_DF.radius⟩ χ) →
  t.a + 2 * t.b ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3156_315634


namespace NUMINAMATH_CALUDE_integral_exp_abs_plus_sqrt_l3156_315640

theorem integral_exp_abs_plus_sqrt : ∫ (x : ℝ) in (-1)..(1), (Real.exp (|x|) + Real.sqrt (1 - x^2)) = 2 * (Real.exp 1 - 1) + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_abs_plus_sqrt_l3156_315640


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l3156_315623

theorem original_sales_tax_percentage 
  (market_price : ℝ) 
  (new_tax_rate : ℝ) 
  (savings : ℝ) :
  market_price = 8400 ∧ 
  new_tax_rate = 10 / 3 ∧ 
  savings = 14 → 
  ∃ original_tax_rate : ℝ,
    original_tax_rate = 3.5 ∧
    market_price * (original_tax_rate / 100) = 
      market_price * (new_tax_rate / 100) + savings :=
by sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l3156_315623


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l3156_315668

/-- Given a circle with area A and circumference C, if A/C = 15, then the radius is 30 -/
theorem circle_radius_from_area_circumference_ratio (A C : ℝ) (h : A / C = 15) :
  ∃ (r : ℝ), A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l3156_315668


namespace NUMINAMATH_CALUDE_bills_final_money_l3156_315601

/-- Calculates Bill's final amount of money after Frank buys pizzas and gives him the rest. -/
theorem bills_final_money (total_initial : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (bills_initial : ℕ) : 
  total_initial = 42 →
  pizza_cost = 11 →
  num_pizzas = 3 →
  bills_initial = 30 →
  bills_initial + (total_initial - (pizza_cost * num_pizzas)) = 39 := by
  sorry

end NUMINAMATH_CALUDE_bills_final_money_l3156_315601


namespace NUMINAMATH_CALUDE_water_transfer_problem_l3156_315620

/-- Represents a rectangular pool with given dimensions -/
structure Pool where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents a valve with a specific flow rate -/
structure Valve where
  flow_rate : ℝ

/-- The main theorem to prove -/
theorem water_transfer_problem 
  (pool_A pool_B : Pool)
  (valve_1 valve_2 : Valve)
  (h1 : pool_A.length = 3 ∧ pool_A.width = 2 ∧ pool_A.depth = 1.2)
  (h2 : pool_B.length = 3 ∧ pool_B.width = 2 ∧ pool_B.depth = 1.2)
  (h3 : valve_1.flow_rate * 18 = pool_A.length * pool_A.width * pool_A.depth)
  (h4 : valve_2.flow_rate * 24 = pool_A.length * pool_A.width * pool_A.depth)
  (h5 : 0.4 * pool_A.length * pool_A.width = (valve_1.flow_rate - valve_2.flow_rate) * t)
  (h6 : t > 0) :
  valve_2.flow_rate * t = 7.2 := by sorry


end NUMINAMATH_CALUDE_water_transfer_problem_l3156_315620


namespace NUMINAMATH_CALUDE_expression_evaluation_l3156_315677

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (2 * c^c - (c + 1) * (c - 1)^c)^c = 131044201 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3156_315677


namespace NUMINAMATH_CALUDE_deck_size_l3156_315642

theorem deck_size (r b u : ℕ) : 
  r + b + u > 0 →
  r / (r + b + u : ℚ) = 1 / 5 →
  r / ((r + b + u + 3) : ℚ) = 1 / 6 →
  r + b + u = 15 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l3156_315642


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3156_315637

theorem quadratic_function_minimum (a b c m : ℝ) (ha : a > 0) :
  (2 * a * m + b = 0) →
  ¬(∀ x : ℝ, a * x^2 + b * x + c ≤ a * m^2 + b * m + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3156_315637


namespace NUMINAMATH_CALUDE_intersection_M_N_l3156_315699

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3156_315699


namespace NUMINAMATH_CALUDE_impossible_friendship_configuration_l3156_315684

/-- A graph representing friendships in a class -/
structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  sym : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges

/-- The degree of a vertex in the graph -/
def degree (G : FriendshipGraph) (v : Nat) : Nat :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- Theorem: It's impossible to have a friendship graph with 30 students where
    9 have 3 friends, 11 have 4 friends, and 10 have 5 friends -/
theorem impossible_friendship_configuration (G : FriendshipGraph) :
  G.vertices.card = 30 →
  (∃ S₁ S₂ S₃ : Finset Nat,
    S₁.card = 9 ∧ S₂.card = 11 ∧ S₃.card = 10 ∧
    S₁ ∪ S₂ ∪ S₃ = G.vertices ∧
    (∀ v ∈ S₁, degree G v = 3) ∧
    (∀ v ∈ S₂, degree G v = 4) ∧
    (∀ v ∈ S₃, degree G v = 5)) →
  False := by
  sorry

end NUMINAMATH_CALUDE_impossible_friendship_configuration_l3156_315684


namespace NUMINAMATH_CALUDE_sin_cos_225_degrees_l3156_315667

theorem sin_cos_225_degrees : 
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 ∧ 
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_225_degrees_l3156_315667


namespace NUMINAMATH_CALUDE_solve_equation_l3156_315662

theorem solve_equation : ∃ y : ℝ, (y - 3)^4 = (1/16)⁻¹ ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3156_315662


namespace NUMINAMATH_CALUDE_probability_one_from_each_l3156_315678

def total_cards : ℕ := 10
def amelia_cards : ℕ := 6
def lucas_cards : ℕ := 4

theorem probability_one_from_each : 
  let p := (amelia_cards * lucas_cards + lucas_cards * amelia_cards) / (total_cards * (total_cards - 1))
  p = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_one_from_each_l3156_315678


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l3156_315676

/-- Given a point (x, y) and a line y = mx + b, 
    returns the symmetric point with respect to the line -/
def symmetricPoint (x y m b : ℝ) : ℝ × ℝ := sorry

/-- The line of symmetry y = x - 1 -/
def lineOfSymmetry : ℝ → ℝ := fun x ↦ x - 1

theorem symmetric_point_correct : 
  symmetricPoint (-1) 2 1 (-1) = (3, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l3156_315676


namespace NUMINAMATH_CALUDE_equal_area_equal_intersection_l3156_315681

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A placement of a rectangle in the plane -/
structure PlacedRectangle where
  rect : Rectangle
  center : Point
  angle : ℝ  -- Rotation angle in radians

/-- A horizontal line in the plane -/
structure HorizontalLine where
  y : ℝ

/-- The intersection of a horizontal line with a placed rectangle -/
def intersection (line : HorizontalLine) (pr : PlacedRectangle) : Option ℝ :=
  sorry

theorem equal_area_equal_intersection 
  (r1 r2 : Rectangle) 
  (h : r1.area = r2.area) :
  ∃ (pr1 pr2 : PlacedRectangle),
    pr1.rect = r1 ∧ pr2.rect = r2 ∧
    ∀ (line : HorizontalLine),
      (intersection line pr1).isSome ∨ (intersection line pr2).isSome →
      (intersection line pr1).isSome ∧ (intersection line pr2).isSome ∧
      (intersection line pr1 = intersection line pr2) :=
by sorry

end NUMINAMATH_CALUDE_equal_area_equal_intersection_l3156_315681


namespace NUMINAMATH_CALUDE_second_quadrant_complex_l3156_315657

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem second_quadrant_complex :
  let z : ℂ := -1
  is_in_second_quadrant ((2 - Complex.I) * z) := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_complex_l3156_315657
