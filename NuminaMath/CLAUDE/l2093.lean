import Mathlib

namespace NUMINAMATH_CALUDE_sandwich_combinations_l2093_209378

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) (n_bread : Nat) :
  n_meat = 12 →
  n_cheese = 11 →
  n_bread = 5 →
  (n_meat.choose 2) * (n_cheese.choose 2) * (n_bread.choose 1) = 18150 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2093_209378


namespace NUMINAMATH_CALUDE_trapezoid_area_is_12_sqrt_5_l2093_209339

/-- Represents a trapezoid with given measurements -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of a trapezoid given its measurements -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with the given measurements has an area of 12√5 -/
theorem trapezoid_area_is_12_sqrt_5 :
  ∀ t : Trapezoid,
    t.base1 = 3 ∧
    t.base2 = 6 ∧
    t.diagonal1 = 7 ∧
    t.diagonal2 = 8 →
    area t = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_12_sqrt_5_l2093_209339


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_2_eval_0_problem_2_eval_3_l2093_209373

theorem problem_1 : -(-2)^2 + |(-Real.sqrt 3)| - 2 * Real.sin (π / 3) + (1 / 2)⁻¹ = -2 := by sorry

theorem problem_2 (m : ℝ) (h : m ≠ 2 ∧ m ≠ -2) : 
  (m / (m - 2) - 2 * m / (m^2 - 4)) + m / (m + 2) = (2 * m^2 - 2 * m) / (m^2 - 4) := by sorry

theorem problem_2_eval_0 : 
  (0 : ℝ) / (0 - 2) - 2 * 0 / (0^2 - 4) + 0 / (0 + 2) = 0 := by sorry

theorem problem_2_eval_3 : 
  (3 : ℝ) / (3 - 2) - 2 * 3 / (3^2 - 4) + 3 / (3 + 2) = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_2_eval_0_problem_2_eval_3_l2093_209373


namespace NUMINAMATH_CALUDE_jerrys_average_score_l2093_209344

theorem jerrys_average_score (current_average : ℝ) : 
  (3 * current_average + 97) / 4 = current_average + 3 →
  current_average = 85 := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_score_l2093_209344


namespace NUMINAMATH_CALUDE_quadratic_range_l2093_209374

theorem quadratic_range (a c : ℝ) (h1 : -4 ≤ a + c) (h2 : a + c ≤ -1)
  (h3 : -1 ≤ 4*a + c) (h4 : 4*a + c ≤ 5) : -1 ≤ 9*a + c ∧ 9*a + c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l2093_209374


namespace NUMINAMATH_CALUDE_parking_capacity_l2093_209332

/-- Represents a parking garage with four levels --/
structure ParkingGarage :=
  (level1 : ℕ)
  (level2 : ℕ)
  (level3 : ℕ)
  (level4 : ℕ)

/-- Calculates the total number of parking spaces in the garage --/
def total_spaces (g : ParkingGarage) : ℕ :=
  g.level1 + g.level2 + g.level3 + g.level4

/-- Theorem: Given the parking garage conditions, 299 more cars can be accommodated --/
theorem parking_capacity 
  (g : ParkingGarage)
  (h1 : g.level1 = 90)
  (h2 : g.level2 = g.level1 + 8)
  (h3 : g.level3 = g.level2 + 12)
  (h4 : g.level4 = g.level3 - 9)
  (h5 : total_spaces g - 100 = 299) : 
  ∃ (n : ℕ), n = 299 ∧ n = total_spaces g - 100 :=
by
  sorry

#check parking_capacity

end NUMINAMATH_CALUDE_parking_capacity_l2093_209332


namespace NUMINAMATH_CALUDE_problem_statement_l2093_209313

theorem problem_statement (a : ℝ) (h : a^2 + a - 1 = 0) :
  2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2093_209313


namespace NUMINAMATH_CALUDE_min_a_value_l2093_209302

-- Define the function representing the left side of the inequality
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∃ x : ℝ, f x a ≤ 8) → 
  (∀ b : ℝ, (∃ x : ℝ, f x b ≤ 8) → a ≤ b) → 
  a = -9 := by
sorry

end NUMINAMATH_CALUDE_min_a_value_l2093_209302


namespace NUMINAMATH_CALUDE_shreehari_pencils_l2093_209322

/-- Calculates the minimum number of pencils initially possessed given the number of students and pencils per student. -/
def min_initial_pencils (num_students : ℕ) (pencils_per_student : ℕ) : ℕ :=
  num_students * pencils_per_student

/-- Proves that given 25 students and 5 pencils per student, the minimum number of pencils initially possessed is 125. -/
theorem shreehari_pencils : min_initial_pencils 25 5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_shreehari_pencils_l2093_209322


namespace NUMINAMATH_CALUDE_number_and_square_average_l2093_209377

theorem number_and_square_average (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5*x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_average_l2093_209377


namespace NUMINAMATH_CALUDE_fabric_area_calculation_l2093_209370

/-- The area of a rectangular piece of fabric -/
def fabric_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of a rectangular piece of fabric with width 3 cm and length 8 cm is 24 square cm -/
theorem fabric_area_calculation :
  fabric_area 3 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_fabric_area_calculation_l2093_209370


namespace NUMINAMATH_CALUDE_distance_to_incenter_l2093_209334

/-- An isosceles right triangle with side length 6√2 -/
structure IsoscelesRightTriangle where
  /-- The length of the equal sides -/
  side_length : ℝ
  /-- The side length is 6√2 -/
  side_length_eq : side_length = 6 * Real.sqrt 2

/-- The incenter of a triangle -/
def incenter (t : IsoscelesRightTriangle) : ℝ × ℝ := sorry

/-- The distance from a vertex to a point -/
def distance_to_point (v : ℝ × ℝ) (p : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the distance from the right angle vertex to the incenter -/
theorem distance_to_incenter (t : IsoscelesRightTriangle) : 
  distance_to_point (0, t.side_length) (incenter t) = 6 - 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_incenter_l2093_209334


namespace NUMINAMATH_CALUDE_shaded_probability_is_half_l2093_209304

/-- Represents a game board with an equilateral triangle -/
structure GameBoard where
  /-- The number of regions the triangle is divided into -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Proof that the total number of regions is 6 -/
  total_is_six : total_regions = 6
  /-- Proof that the number of shaded regions is 3 -/
  shaded_is_three : shaded_regions = 3

/-- The probability of the spinner landing in a shaded region -/
def shaded_probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating that the probability of landing in a shaded region is 1/2 -/
theorem shaded_probability_is_half (board : GameBoard) :
  shaded_probability board = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_probability_is_half_l2093_209304


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2093_209338

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 49 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2093_209338


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l2093_209333

theorem greatest_integer_fraction (x : ℤ) :
  x ≠ 3 →
  (∃ y : ℤ, (x^2 + 2*x + 5) = (x - 3) * y) →
  x ≤ 23 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l2093_209333


namespace NUMINAMATH_CALUDE_savings_for_engagement_ring_l2093_209335

/-- Calculates the monthly savings required to accumulate two months' salary in a given time period. -/
def monthly_savings (annual_salary : ℚ) (months_to_save : ℕ) : ℚ :=
  (2 * annual_salary) / (12 * months_to_save)

/-- Proves that given an annual salary of $60,000 and 10 months to save,
    the amount to save per month to accumulate two months' salary is $1,000. -/
theorem savings_for_engagement_ring :
  monthly_savings 60000 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_savings_for_engagement_ring_l2093_209335


namespace NUMINAMATH_CALUDE_expand_binomial_product_l2093_209346

theorem expand_binomial_product (x : ℝ) : (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomial_product_l2093_209346


namespace NUMINAMATH_CALUDE_orthogonality_iff_k_eq_4_l2093_209312

/-- Two unit vectors with an angle of 60° between them -/
structure UnitVectorPair :=
  (e₁ e₂ : ℝ × ℝ)
  (unit_e₁ : e₁.1 ^ 2 + e₁.2 ^ 2 = 1)
  (unit_e₂ : e₂.1 ^ 2 + e₂.2 ^ 2 = 1)
  (angle_60 : e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2)

/-- The orthogonality condition -/
def orthogonality (v : UnitVectorPair) (k : ℝ) : Prop :=
  (2 * v.e₁.1 - k * v.e₂.1) * v.e₁.1 + (2 * v.e₁.2 - k * v.e₂.2) * v.e₁.2 = 0

/-- The main theorem -/
theorem orthogonality_iff_k_eq_4 (v : UnitVectorPair) :
  orthogonality v 4 ∧ (∀ k : ℝ, orthogonality v k → k = 4) :=
sorry

end NUMINAMATH_CALUDE_orthogonality_iff_k_eq_4_l2093_209312


namespace NUMINAMATH_CALUDE_bonus_sector_area_l2093_209324

/-- Given a circular spinner with radius 15 cm and a "Bonus" sector with a 
    probability of 1/3 of being landed on, the area of the "Bonus" sector 
    is 75π square centimeters. -/
theorem bonus_sector_area (radius : ℝ) (probability : ℝ) (bonus_area : ℝ) : 
  radius = 15 →
  probability = 1 / 3 →
  bonus_area = probability * π * radius^2 →
  bonus_area = 75 * π := by
  sorry


end NUMINAMATH_CALUDE_bonus_sector_area_l2093_209324


namespace NUMINAMATH_CALUDE_derivative_tan_and_exp_minus_sqrt_l2093_209385

open Real

theorem derivative_tan_and_exp_minus_sqrt (x : ℝ) : 
  (deriv tan x = 1 / (cos x)^2) ∧ 
  (deriv (fun x => exp x - sqrt x) x = exp x - 1 / (2 * sqrt x)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_tan_and_exp_minus_sqrt_l2093_209385


namespace NUMINAMATH_CALUDE_no_suitable_dishes_l2093_209307

theorem no_suitable_dishes (total_dishes : ℕ) (vegetarian_dishes : ℕ) 
  (gluten_dishes : ℕ) (nut_dishes : ℕ) 
  (h1 : vegetarian_dishes = 6)
  (h2 : vegetarian_dishes = total_dishes / 4)
  (h3 : gluten_dishes = 4)
  (h4 : nut_dishes = 2)
  (h5 : gluten_dishes + nut_dishes = vegetarian_dishes) :
  (vegetarian_dishes - gluten_dishes - nut_dishes : ℚ) / total_dishes = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_suitable_dishes_l2093_209307


namespace NUMINAMATH_CALUDE_bottle_caps_count_l2093_209318

/-- The number of bottle caps in the box after removing some and adding others. -/
def final_bottle_caps (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that given the initial conditions, the final number of bottle caps is 137. -/
theorem bottle_caps_count : final_bottle_caps 144 63 56 = 137 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_count_l2093_209318


namespace NUMINAMATH_CALUDE_surface_area_of_problem_lshape_l2093_209321

/-- Represents the L-shaped structure formed by unit cubes -/
structure LShape where
  base_length : Nat
  top_length : Nat
  top_start_position : Nat

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LShape) : Nat :=
  let base_visible_top := l.base_length - l.top_length
  let base_visible_sides := 2 * l.base_length
  let base_visible_ends := 2
  let top_visible_top := l.top_length
  let top_visible_sides := 2 * l.top_length
  let top_visible_ends := 2
  base_visible_top + base_visible_sides + base_visible_ends +
  top_visible_top + top_visible_sides + top_visible_ends

/-- The specific L-shaped structure described in the problem -/
def problem_lshape : LShape :=
  { base_length := 10
    top_length := 5
    top_start_position := 5 }

theorem surface_area_of_problem_lshape :
  surface_area problem_lshape = 45 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_lshape_l2093_209321


namespace NUMINAMATH_CALUDE_v_2002_equals_1_l2093_209355

/-- The function g as defined in the problem --/
def g : ℕ → ℕ
| 1 => 2
| 2 => 3
| 3 => 1
| 4 => 4
| 5 => 5
| _ => 0  -- default case for inputs not in the table

/-- The sequence v defined recursively --/
def v : ℕ → ℕ
| 0 => 2
| (n + 1) => g (v n)

/-- Theorem stating that the 2002nd term of the sequence is 1 --/
theorem v_2002_equals_1 : v 2002 = 1 := by
  sorry


end NUMINAMATH_CALUDE_v_2002_equals_1_l2093_209355


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2093_209367

theorem geometric_progression_fourth_term :
  ∀ x : ℝ,
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = r * x ∧ (3*x + 3) = r * (2*x + 2)) →
  ∃ fourth_term : ℝ, fourth_term = -13/2 ∧ (3*x + 3) * r = fourth_term :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2093_209367


namespace NUMINAMATH_CALUDE_inequality_solution_l2093_209398

theorem inequality_solution (x : ℝ) : 
  (x^2 + x^3 - x^4) / (x + x^2 - x^3) ≥ -1 ↔ 
  (x ∈ Set.Icc (-1) ((1 - Real.sqrt 5) / 2) ∪ 
   Set.Ioo ((1 - Real.sqrt 5) / 2) 0 ∪ 
   Set.Ioo 0 ((1 + Real.sqrt 5) / 2) ∪ 
   Set.Ioi ((1 + Real.sqrt 5) / 2)) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2093_209398


namespace NUMINAMATH_CALUDE_team_average_correct_l2093_209328

theorem team_average_correct (v w x y : ℝ) (h : v < w ∧ w < x ∧ x < y) : 
  ((v + w) / 2 + (x + y) / 2) / 2 = (v + w + x + y) / 4 := by
  sorry

end NUMINAMATH_CALUDE_team_average_correct_l2093_209328


namespace NUMINAMATH_CALUDE_no_solution_equation_l2093_209309

theorem no_solution_equation : ∀ (a b : ℤ), a^4 + 6 ≠ b^3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2093_209309


namespace NUMINAMATH_CALUDE_c_alone_time_l2093_209371

-- Define the work rates for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
def condition1 : Prop := rA + rB = 1/3
def condition2 : Prop := rB + rC = 1/3
def condition3 : Prop := rA + rC = 2/3

-- Theorem to prove
theorem c_alone_time (h1 : condition1 rA rB) (h2 : condition2 rB rC) (h3 : condition3 rA rC) :
  1 / rC = 3 := by
  sorry


end NUMINAMATH_CALUDE_c_alone_time_l2093_209371


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2093_209397

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line_equation x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2093_209397


namespace NUMINAMATH_CALUDE_nancy_chips_to_brother_l2093_209323

def tortilla_chips_problem (total_chips : ℕ) (kept_chips : ℕ) (sister_chips : ℕ) : ℕ :=
  total_chips - kept_chips - sister_chips

theorem nancy_chips_to_brother :
  tortilla_chips_problem 22 10 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_chips_to_brother_l2093_209323


namespace NUMINAMATH_CALUDE_andrei_apple_spending_l2093_209329

/-- Calculates Andrei's monthly spending on apples after price increase and discount --/
def andreiMonthlySpending (originalPrice : ℚ) (priceIncrease : ℚ) (discount : ℚ) (kgPerMonth : ℚ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease)
  let discountedPrice := newPrice * (1 - discount)
  discountedPrice * kgPerMonth

/-- Theorem stating that Andrei's monthly spending on apples is 99 rubles --/
theorem andrei_apple_spending :
  andreiMonthlySpending 50 (10/100) (10/100) 2 = 99 := by sorry

end NUMINAMATH_CALUDE_andrei_apple_spending_l2093_209329


namespace NUMINAMATH_CALUDE_octal_127_equals_87_l2093_209379

-- Define the octal number as a list of digits
def octal_127 : List Nat := [1, 2, 7]

-- Function to convert octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_127_equals_87 :
  octal_to_decimal octal_127 = 87 := by
  sorry

end NUMINAMATH_CALUDE_octal_127_equals_87_l2093_209379


namespace NUMINAMATH_CALUDE_points_collinear_l2093_209394

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

/-- The three given points are collinear. -/
theorem points_collinear : collinear (2, 5) (-6, -3) (-4, -1) := by
  sorry


end NUMINAMATH_CALUDE_points_collinear_l2093_209394


namespace NUMINAMATH_CALUDE_smallest_class_size_class_with_25_students_exists_l2093_209336

/-- Represents a class of students who took a history test -/
structure HistoryClass where
  /-- The total number of students in the class -/
  num_students : ℕ
  /-- The total score of all students -/
  total_score : ℕ
  /-- The number of students who scored 120 points -/
  perfect_scores : ℕ
  /-- The number of students who scored 115 points -/
  near_perfect_scores : ℕ

/-- The properties of the history class based on the given problem -/
def valid_history_class (c : HistoryClass) : Prop :=
  c.perfect_scores = 8 ∧
  c.near_perfect_scores = 3 ∧
  c.total_score = c.num_students * 92 ∧
  c.total_score ≥ c.perfect_scores * 120 + c.near_perfect_scores * 115 + (c.num_students - c.perfect_scores - c.near_perfect_scores) * 70

/-- The theorem stating that the smallest possible number of students in the class is 25 -/
theorem smallest_class_size (c : HistoryClass) (h : valid_history_class c) : c.num_students ≥ 25 := by
  sorry

/-- The theorem stating that a class with 25 students satisfying all conditions exists -/
theorem class_with_25_students_exists : ∃ c : HistoryClass, valid_history_class c ∧ c.num_students = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_class_with_25_students_exists_l2093_209336


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2093_209369

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) :
  Real.sqrt (((a.1 + 2 * b.1) ^ 2) + ((a.2 + 2 * b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2093_209369


namespace NUMINAMATH_CALUDE_expression_evaluation_l2093_209387

theorem expression_evaluation (a b c : ℚ) (ha : a = 14) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

#eval (14 : ℚ) + 19 + 23

end NUMINAMATH_CALUDE_expression_evaluation_l2093_209387


namespace NUMINAMATH_CALUDE_arun_lower_limit_l2093_209327

-- Define Arun's weight as a real number
variable (W : ℝ)

-- Define the conditions
def arun_upper_limit : Prop := W < 72
def brother_opinion : Prop := 60 < W ∧ W < 70
def mother_opinion : Prop := W ≤ 67
def average_weight : Prop := (W + 67) / 2 = 66

-- Define the theorem
theorem arun_lower_limit :
  arun_upper_limit W →
  brother_opinion W →
  mother_opinion W →
  average_weight W →
  W = 65 := by sorry

end NUMINAMATH_CALUDE_arun_lower_limit_l2093_209327


namespace NUMINAMATH_CALUDE_remainder_x2023_plus_1_l2093_209300

theorem remainder_x2023_plus_1 (x : ℂ) : 
  (x^2023 + 1) % (x^6 - x^4 + x^2 - 1) = -x^3 + 1 := by sorry

end NUMINAMATH_CALUDE_remainder_x2023_plus_1_l2093_209300


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2093_209326

theorem sum_of_x_and_y (x y : ℝ) : 
  ((x + Real.sqrt y) + (x - Real.sqrt y) = 8) →
  ((x + Real.sqrt y) * (x - Real.sqrt y) = 15) →
  (x + y = 5) := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2093_209326


namespace NUMINAMATH_CALUDE_cubic_expression_value_l2093_209308

theorem cubic_expression_value (p q : ℝ) : 
  (8 * p + 2 * q = 2022) → 
  ((-8) * p + (-2) * q + 1 = -2021) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l2093_209308


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2093_209305

theorem rectangular_field_area (length width area : ℝ) : 
  length = width + 10 →
  length = 19.13 →
  area = length * width →
  area = 174.6359 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2093_209305


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2093_209337

/-- Given complex numbers x and y, prove that 3x + 4y = 17 + 2i -/
theorem complex_expression_equality (x y : ℂ) (hx : x = 3 + 2*I) (hy : y = 2 - I) :
  3*x + 4*y = 17 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2093_209337


namespace NUMINAMATH_CALUDE_no_natural_solution_l2093_209368

theorem no_natural_solution :
  ¬∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2093_209368


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2093_209357

theorem ratio_sum_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 4) (h3 : a * b + b * c + c * a = 13) : b * c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2093_209357


namespace NUMINAMATH_CALUDE_root_property_l2093_209325

theorem root_property (x₀ : ℝ) (h : 2 * x₀^2 * Real.exp (2 * x₀) + Real.log x₀ = 0) :
  2 * x₀ + Real.log x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l2093_209325


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2093_209359

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℚ := 3 * n

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℚ := n * (3 + a n) / 2

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := 3 / (2 * S n)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_sequence_properties :
  (a 1 = 3) ∧
  (a 3 + S 3 = 27) ∧
  (∀ n : ℕ, a n = 3 * n) ∧
  (∀ n : ℕ, T n = n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2093_209359


namespace NUMINAMATH_CALUDE_cubic_minimum_at_negative_one_l2093_209345

/-- A cubic function with parameters p, q, and r -/
def cubic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  x^3 + p*x^2 + q*x + r

theorem cubic_minimum_at_negative_one (p q r : ℝ) :
  (∀ x, cubic_function p q r x ≥ 0) ∧
  (cubic_function p q r (-1) = 0) →
  r = p - 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_minimum_at_negative_one_l2093_209345


namespace NUMINAMATH_CALUDE_line_equation_with_x_intercept_and_slope_angle_l2093_209386

theorem line_equation_with_x_intercept_and_slope_angle 
  (x_intercept : ℝ) 
  (slope_angle : ℝ) 
  (h1 : x_intercept = 2) 
  (h2 : slope_angle = 135) :
  ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b ∧ 
    (x = x_intercept ∧ y = 0) ∧ 
    m = Real.tan (π - slope_angle * π / 180) ∧
    y = -x + 2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_with_x_intercept_and_slope_angle_l2093_209386


namespace NUMINAMATH_CALUDE_french_exam_vocabulary_l2093_209352

theorem french_exam_vocabulary (total_words : ℕ) (guess_rate : ℚ) (target_score : ℚ) : 
  total_words = 600 → 
  guess_rate = 5 / 100 → 
  target_score = 90 / 100 → 
  ∃ (words_to_learn : ℕ), 
    words_to_learn ≥ 537 ∧ 
    (words_to_learn : ℚ) / total_words + 
      guess_rate * ((total_words - words_to_learn) : ℚ) / total_words ≥ target_score ∧
    ∀ (x : ℕ), x < 537 → 
      (x : ℚ) / total_words + 
        guess_rate * ((total_words - x) : ℚ) / total_words < target_score :=
by sorry

end NUMINAMATH_CALUDE_french_exam_vocabulary_l2093_209352


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2093_209363

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 1991 is the only natural number n for which n + s(n) = 2011 -/
theorem unique_solution_for_equation : 
  ∃! n : ℕ, n + sum_of_digits n = 2011 ∧ n = 1991 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2093_209363


namespace NUMINAMATH_CALUDE_negation_equivalence_l2093_209360

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x = 0) ↔ (∀ x : ℝ, x^2 - 2*x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2093_209360


namespace NUMINAMATH_CALUDE_equal_area_line_equation_l2093_209390

/-- A circle in the coordinate plane -/
structure Circle where
  center : ℝ × ℝ
  diameter : ℝ

/-- The region S formed by the union of nine circular regions -/
def region_S : Set (ℝ × ℝ) :=
  sorry

/-- A line in the coordinate plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a line divides a region into two equal areas -/
def divides_equally (l : Line) (r : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The equation of a line in the form ax = by + c -/
structure LineEquation where
  a : ℕ
  b : ℕ
  c : ℕ
  gcd_one : Nat.gcd a (Nat.gcd b c) = 1

theorem equal_area_line_equation :
  ∃ (eq : LineEquation),
    let l : Line := { slope := 2, intercept := sorry }
    divides_equally l region_S ∧
    eq.a^2 + eq.b^2 + eq.c^2 = 69 :=
  sorry

end NUMINAMATH_CALUDE_equal_area_line_equation_l2093_209390


namespace NUMINAMATH_CALUDE_exists_m_for_second_quadrant_l2093_209320

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P -/
def y_coord : ℝ := 3

theorem exists_m_for_second_quadrant : 
  ∃ m : ℝ, m < 1 ∧ is_in_second_quadrant (x_coord m) y_coord :=
sorry

end NUMINAMATH_CALUDE_exists_m_for_second_quadrant_l2093_209320


namespace NUMINAMATH_CALUDE_i_minus_one_in_second_quadrant_l2093_209375

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem i_minus_one_in_second_quadrant :
  is_in_second_quadrant (Complex.I - 1) := by
  sorry

end NUMINAMATH_CALUDE_i_minus_one_in_second_quadrant_l2093_209375


namespace NUMINAMATH_CALUDE_beta_still_water_speed_l2093_209301

/-- Represents a boat with its speed in still water -/
structure Boat where
  speed : ℝ

/-- Represents the river with its current speed -/
structure River where
  currentSpeed : ℝ

/-- Represents a journey on the river -/
inductive Direction
  | Upstream
  | Downstream

def effectiveSpeed (b : Boat) (r : River) (d : Direction) : ℝ :=
  match d with
  | Direction.Upstream => b.speed + r.currentSpeed
  | Direction.Downstream => b.speed - r.currentSpeed

theorem beta_still_water_speed 
  (alpha : Boat)
  (beta : Boat)
  (river : River)
  (h1 : alpha.speed = 56)
  (h2 : beta.speed = 52)
  (h3 : river.currentSpeed = 4)
  (h4 : effectiveSpeed alpha river Direction.Upstream / effectiveSpeed beta river Direction.Downstream = 5 / 4)
  (h5 : effectiveSpeed alpha river Direction.Downstream / effectiveSpeed beta river Direction.Upstream = 4 / 5) :
  beta.speed = 61 := by
  sorry

end NUMINAMATH_CALUDE_beta_still_water_speed_l2093_209301


namespace NUMINAMATH_CALUDE_blue_balls_count_l2093_209330

theorem blue_balls_count (total : ℕ) (red : ℕ) (orange : ℕ) (pink : ℕ) 
  (h1 : total = 50)
  (h2 : red = 20)
  (h3 : orange = 5)
  (h4 : pink = 3 * orange)
  : total - (red + orange + pink) = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2093_209330


namespace NUMINAMATH_CALUDE_imaginary_part_zi_l2093_209395

def complex_coords (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem imaginary_part_zi (z : ℂ) (h : complex_coords z = (-2, 1)) : 
  (z * Complex.I).im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_zi_l2093_209395


namespace NUMINAMATH_CALUDE_problem_solution_l2093_209340

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
  (x + y) / (x - y) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2093_209340


namespace NUMINAMATH_CALUDE_remainder_of_n_mod_11_l2093_209383

def A : ℕ := (10^20069 - 1) / 9
def B : ℕ := 7 * (10^20066 - 1) / 9

def n : ℤ := A^2 - B

theorem remainder_of_n_mod_11 : n % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_mod_11_l2093_209383


namespace NUMINAMATH_CALUDE_shoes_sold_l2093_209362

theorem shoes_sold (shoes sandals : ℕ) 
  (ratio : shoes / sandals = 9 / 5)
  (sandals_count : sandals = 40) : 
  shoes = 72 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_l2093_209362


namespace NUMINAMATH_CALUDE_jessica_mark_earnings_l2093_209342

/-- Given the working hours and earnings of Jessica and Mark, prove that t = 5 --/
theorem jessica_mark_earnings (t : ℝ) : 
  (t + 2) * (4 * t + 1) = (4 * t - 7) * (t + 3) + 4 → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_mark_earnings_l2093_209342


namespace NUMINAMATH_CALUDE_banana_boxes_l2093_209348

def total_bananas : ℕ := 40
def bananas_per_box : ℕ := 4

theorem banana_boxes : total_bananas / bananas_per_box = 10 := by
  sorry

end NUMINAMATH_CALUDE_banana_boxes_l2093_209348


namespace NUMINAMATH_CALUDE_largest_and_smallest_subsequence_l2093_209391

def original_number : ℕ := 798056132

-- Define a function to check if a number is a valid 5-digit subsequence of the original number
def is_valid_subsequence (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  ∃ (a b c d e : ℕ), 
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    ∃ (i j k l m : ℕ), 
      i < j ∧ j < k ∧ k < l ∧ l < m ∧
      (original_number / 10 ^ (8 - i) % 10 = a) ∧
      (original_number / 10 ^ (8 - j) % 10 = b) ∧
      (original_number / 10 ^ (8 - k) % 10 = c) ∧
      (original_number / 10 ^ (8 - l) % 10 = d) ∧
      (original_number / 10 ^ (8 - m) % 10 = e)

theorem largest_and_smallest_subsequence :
  (∀ n : ℕ, is_valid_subsequence n → n ≤ 98632) ∧
  (∀ n : ℕ, is_valid_subsequence n → n ≥ 56132) ∧
  is_valid_subsequence 98632 ∧
  is_valid_subsequence 56132 := by sorry

end NUMINAMATH_CALUDE_largest_and_smallest_subsequence_l2093_209391


namespace NUMINAMATH_CALUDE_total_votes_l2093_209358

theorem total_votes (veggies : ℕ) (meat : ℕ) (dairy : ℕ) (plant_protein : ℕ)
  (h1 : veggies = 337)
  (h2 : meat = 335)
  (h3 : dairy = 274)
  (h4 : plant_protein = 212) :
  veggies + meat + dairy + plant_protein = 1158 :=
by sorry

end NUMINAMATH_CALUDE_total_votes_l2093_209358


namespace NUMINAMATH_CALUDE_coefficient_of_term_l2093_209399

theorem coefficient_of_term (x y : ℝ) : 
  ∃ (c : ℝ), -π * x * y^3 / 5 = c * x * y^3 ∧ c = -π / 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_term_l2093_209399


namespace NUMINAMATH_CALUDE_number_of_cars_on_road_prove_number_of_cars_l2093_209315

theorem number_of_cars_on_road (total_distance : ℝ) (car_spacing : ℝ) : ℝ :=
  let number_of_cars := (total_distance / car_spacing) + 1
  number_of_cars

theorem prove_number_of_cars :
  number_of_cars_on_road 242 5.5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cars_on_road_prove_number_of_cars_l2093_209315


namespace NUMINAMATH_CALUDE_tenth_pirate_share_l2093_209381

/-- Represents the number of coins each pirate takes -/
def pirate_share (i : Nat) (remaining : Nat) : Nat :=
  if i ≤ 5 then
    (i * remaining) / 10
  else if i < 10 then
    20
  else
    remaining

/-- Calculates the remaining coins after each pirate takes their share -/
def remaining_coins (i : Nat) (initial : Nat) : Nat :=
  if i = 0 then initial
  else
    remaining_coins (i - 1) initial - pirate_share (i - 1) (remaining_coins (i - 1) initial)

/-- The main theorem stating that the 10th pirate receives 376 coins -/
theorem tenth_pirate_share : pirate_share 10 (remaining_coins 10 3000) = 376 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pirate_share_l2093_209381


namespace NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l2093_209341

/-- Given a quadratic equation with complex coefficients, prove the maximum and minimum absolute values of a specific coefficient. -/
theorem quadratic_coefficient_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  z₁^2 - 4*z₂ = 16 + 20*I →
  x^2 + z₁*x + z₂ + m = 0 →
  α^2 + z₁*α + z₂ + m = 0 →
  β^2 + z₁*β + z₂ + m = 0 →
  Complex.abs (α - β) = 2 * Real.sqrt 7 →
  (Complex.abs m = Real.sqrt 41 + 7 ∨ Complex.abs m = Real.sqrt 41 - 7) ∧
  ∀ m' : ℂ, Complex.abs m' ≤ Real.sqrt 41 + 7 ∧ Complex.abs m' ≥ Real.sqrt 41 - 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l2093_209341


namespace NUMINAMATH_CALUDE_cos_48_degrees_l2093_209347

theorem cos_48_degrees : Real.cos (48 * π / 180) = Real.cos (48 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l2093_209347


namespace NUMINAMATH_CALUDE_group_dinner_cost_l2093_209350

/-- Calculate the total cost for a group dinner including service charge -/
theorem group_dinner_cost (num_people : ℕ) (meal_cost drink_cost dessert_cost : ℚ) 
  (service_charge_rate : ℚ) : 
  num_people = 5 →
  meal_cost = 12 →
  drink_cost = 3 →
  dessert_cost = 5 →
  service_charge_rate = 1/10 →
  (num_people * (meal_cost + drink_cost + dessert_cost)) * (1 + service_charge_rate) = 110 := by
  sorry

end NUMINAMATH_CALUDE_group_dinner_cost_l2093_209350


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2093_209380

theorem rational_equation_solution : 
  ∃! y : ℚ, (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5 ∧ y = -17/6 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2093_209380


namespace NUMINAMATH_CALUDE_prob_same_color_is_49_128_l2093_209361

-- Define the number of balls of each color
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ := 
  (green_balls * green_balls + red_balls * red_balls + blue_balls * blue_balls) / 
  (total_balls * total_balls)

-- Theorem statement
theorem prob_same_color_is_49_128 : prob_same_color = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_49_128_l2093_209361


namespace NUMINAMATH_CALUDE_house_wall_nails_l2093_209388

/-- The number of nails needed for large planks -/
def large_planks_nails : ℕ := 15

/-- The number of nails needed for small planks -/
def small_planks_nails : ℕ := 5

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := large_planks_nails + small_planks_nails

theorem house_wall_nails : total_nails = 20 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_nails_l2093_209388


namespace NUMINAMATH_CALUDE_sum_of_squares_l2093_209382

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 0) (h_power : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2093_209382


namespace NUMINAMATH_CALUDE_share_calculation_l2093_209376

theorem share_calculation (total : ℚ) (a b c : ℚ) 
  (h1 : total = 510)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : b = 90 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l2093_209376


namespace NUMINAMATH_CALUDE_exists_unsolvable_chessboard_l2093_209392

/-- Represents a 12x12 chessboard where each square can be black or white -/
def Chessboard := Fin 12 → Fin 12 → Bool

/-- Represents a row or column flip operation -/
inductive FlipOperation
| row (i : Fin 12)
| col (j : Fin 12)

/-- Applies a flip operation to a chessboard -/
def applyFlip (board : Chessboard) (op : FlipOperation) : Chessboard :=
  match op with
  | FlipOperation.row i => fun x y => if x = i then !board x y else board x y
  | FlipOperation.col j => fun x y => if y = j then !board x y else board x y

/-- Checks if all squares on the board are black -/
def allBlack (board : Chessboard) : Prop :=
  ∀ i j, board i j = true

/-- Theorem: There exists an initial chessboard configuration that cannot be made all black -/
theorem exists_unsolvable_chessboard : 
  ∃ (initial : Chessboard), ¬∃ (ops : List FlipOperation), allBlack (ops.foldl applyFlip initial) :=
sorry

end NUMINAMATH_CALUDE_exists_unsolvable_chessboard_l2093_209392


namespace NUMINAMATH_CALUDE_train_overtake_l2093_209384

-- Define the speeds of the trains
def speed_A : ℝ := 30
def speed_B : ℝ := 45

-- Define the overtake distance
def overtake_distance : ℝ := 180

-- Define the time difference between train departures
def time_difference : ℝ := 2

-- Theorem statement
theorem train_overtake :
  speed_A * (time_difference + (overtake_distance / speed_B)) = overtake_distance ∧
  speed_B * (overtake_distance / speed_B) = overtake_distance := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_l2093_209384


namespace NUMINAMATH_CALUDE_intersection_point_d_l2093_209349

def g (c : ℤ) (x : ℝ) : ℝ := 5 * x + c

theorem intersection_point_d (c : ℤ) (d : ℤ) :
  (g c (-5) = d) ∧ (g c d = -5) → d = -5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_d_l2093_209349


namespace NUMINAMATH_CALUDE_angle_C_measure_l2093_209306

-- Define the angles
variable (A B C : ℝ)

-- Define the parallel lines property
variable (p_parallel_q : Bool)

-- State the theorem
theorem angle_C_measure :
  p_parallel_q ∧ A = (1/6) * B → C = 25.71 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2093_209306


namespace NUMINAMATH_CALUDE_triangular_pyramid_base_layer_l2093_209331

/-- The number of spheres in a regular triangular pyramid with n layers -/
def pyramidSum (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The number of spheres in the base layer of a regular triangular pyramid with n layers -/
def baseLayer (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_pyramid_base_layer :
  ∃ n : ℕ, pyramidSum n = 120 ∧ baseLayer n = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_base_layer_l2093_209331


namespace NUMINAMATH_CALUDE_tv_cost_l2093_209372

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 600 →
  furniture_fraction = 3/4 →
  tv_cost = savings - (furniture_fraction * savings) →
  tv_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l2093_209372


namespace NUMINAMATH_CALUDE_square_root_equal_self_l2093_209353

theorem square_root_equal_self : ∀ x : ℝ, x = Real.sqrt x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_square_root_equal_self_l2093_209353


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_of_2023_pow_6_plus_1_l2093_209317

theorem least_odd_prime_factor_of_2023_pow_6_plus_1 :
  (Nat.minFac (2023^6 + 1)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_of_2023_pow_6_plus_1_l2093_209317


namespace NUMINAMATH_CALUDE_log_sum_sixteen_sixtyfour_l2093_209316

theorem log_sum_sixteen_sixtyfour : Real.log 64 / Real.log 16 + Real.log 16 / Real.log 64 = 13 / 6 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_sixteen_sixtyfour_l2093_209316


namespace NUMINAMATH_CALUDE_dad_steps_l2093_209389

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad, Masha, and Yasha -/
def step_relation (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha ∧ 5 * s.masha = 3 * s.yasha

/-- Theorem stating that given the conditions, Dad took 90 steps -/
theorem dad_steps (s : Steps) :
  step_relation s → s.masha + s.yasha = 400 → s.dad = 90 := by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l2093_209389


namespace NUMINAMATH_CALUDE_ratio_problem_l2093_209351

theorem ratio_problem (x y : ℝ) (h : (2*x - y) / (x + y) = 2/3) : x / y = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2093_209351


namespace NUMINAMATH_CALUDE_number_problem_l2093_209364

theorem number_problem : ∃ x : ℝ, 4 * x = 28 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2093_209364


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l2093_209319

theorem angle_terminal_side_value (k : ℝ) (θ : ℝ) (h : k < 0) :
  (∃ (r : ℝ), r > 0 ∧ -4 * k = r * Real.cos θ ∧ 3 * k = r * Real.sin θ) →
  2 * Real.sin θ + Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l2093_209319


namespace NUMINAMATH_CALUDE_parabola_properties_l2093_209311

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Points on the parabola -/
structure ParabolaPoints (p : Parabola) where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h₁ : y₁ = p.a * x₁^2 + p.b * x₁
  h₂ : y₂ = p.a * x₂^2 + p.b * x₂
  h₃ : x₁ < x₂
  h₄ : x₁ + x₂ = 2

/-- Theorem about parabola properties -/
theorem parabola_properties (p : Parabola) (pts : ParabolaPoints p)
  (h₁ : p.a * 3^2 + p.b * 3 = 3) :
  (p.b = 1 - 3 * p.a) ∧
  (pts.y₁ = pts.y₂ → p.a = 1) ∧
  (pts.y₁ < pts.y₂ → 0 < p.a ∧ p.a < 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2093_209311


namespace NUMINAMATH_CALUDE_six_cows_satisfy_condition_unique_cow_count_l2093_209365

/-- Represents the farm with cows and chickens -/
structure Farm where
  cows : ℕ
  chickens : ℕ

/-- The total number of legs on the farm -/
def totalLegs (f : Farm) : ℕ := 5 * f.cows + 2 * f.chickens

/-- The total number of heads on the farm -/
def totalHeads (f : Farm) : ℕ := f.cows + f.chickens

/-- The farm satisfies the given condition -/
def satisfiesCondition (f : Farm) : Prop :=
  totalLegs f = 20 + 2 * totalHeads f

/-- Theorem stating that the farm with 6 cows satisfies the condition -/
theorem six_cows_satisfy_condition :
  ∃ (f : Farm), f.cows = 6 ∧ satisfiesCondition f :=
sorry

/-- Theorem stating that 6 is the only number of cows that satisfies the condition -/
theorem unique_cow_count :
  ∀ (f : Farm), satisfiesCondition f → f.cows = 6 :=
sorry

end NUMINAMATH_CALUDE_six_cows_satisfy_condition_unique_cow_count_l2093_209365


namespace NUMINAMATH_CALUDE_bill_split_correct_l2093_209354

-- Define the given values
def total_bill : ℚ := 139
def num_people : ℕ := 3
def tip_percentage : ℚ := 1 / 10

-- Define the function to calculate the amount each person should pay
def amount_per_person (bill : ℚ) (people : ℕ) (tip : ℚ) : ℚ :=
  (bill * (1 + tip)) / people

-- Theorem statement
theorem bill_split_correct :
  amount_per_person total_bill num_people tip_percentage = 5097 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bill_split_correct_l2093_209354


namespace NUMINAMATH_CALUDE_percentage_comparisons_l2093_209343

theorem percentage_comparisons (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  (x / y) * 100 = 80 ∧
  ((y - x) / x) * 100 = 25 ∧
  ((y - x) / y) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparisons_l2093_209343


namespace NUMINAMATH_CALUDE_right_triangle_angles_l2093_209310

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- length of one leg
  b : ℝ  -- length of the other leg
  h : ℝ  -- length of the hypotenuse
  right_angle : a^2 + b^2 = h^2  -- Pythagorean theorem

-- Define the quadrilateral formed by the perpendicular bisector
structure Quadrilateral where
  d1 : ℝ  -- length of one diagonal
  d2 : ℝ  -- length of the other diagonal

-- Main theorem
theorem right_triangle_angles (triangle : RightTriangle) (quad : Quadrilateral) :
  quad.d1 / quad.d2 = (1 + Real.sqrt 3) / (2 * Real.sqrt 2) →
  ∃ (angle1 angle2 : ℝ),
    angle1 = 15 * π / 180 ∧
    angle2 = 75 * π / 180 ∧
    angle1 + angle2 = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l2093_209310


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2093_209393

theorem quadratic_roots_sum_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (8 * x₁^2 + 2 * k * x₁ + k - 1 = 0) ∧ 
    (8 * x₂^2 + 2 * k * x₂ + k - 1 = 0) ∧ 
    (x₁^2 + x₂^2 = 1) ∧
    (4 * k^2 - 32 * (k - 1) ≥ 0)) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2093_209393


namespace NUMINAMATH_CALUDE_equation_solution_expression_factorization_inequalities_solution_l2093_209314

-- Part 1: Equation solution
theorem equation_solution (x : ℝ) : 
  1 / (x - 2) - x / (x^2 - 4) = 2 / (x + 2) → x = 6 :=
by sorry

-- Part 2: Expression factorization
theorem expression_factorization (x : ℝ) :
  x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (x + 3) * (x - 3) :=
by sorry

-- Part 3: System of inequalities solution
theorem inequalities_solution (x : ℝ) :
  ((x - 3) / 3 < 1 ∧ 3 * x - 4 ≤ 2 * (3 * x - 2)) ↔ (0 ≤ x ∧ x < 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_expression_factorization_inequalities_solution_l2093_209314


namespace NUMINAMATH_CALUDE_alphabet_letters_with_dot_only_l2093_209396

theorem alphabet_letters_with_dot_only (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h_total : total = 40)
  (h_both : both = 10)
  (h_line_only : line_only = 24)
  (h_all_types : total = both + line_only + (total - both - line_only)) :
  total - both - line_only = 6 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_letters_with_dot_only_l2093_209396


namespace NUMINAMATH_CALUDE_complementary_event_correct_l2093_209356

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- An event in the sample space of drawing balls -/
inductive Event
  | AtLeastOneWhite
  | AllRed

/-- The complementary event function -/
def complementary_event : Event → Event
  | Event.AtLeastOneWhite => Event.AllRed
  | Event.AllRed => Event.AtLeastOneWhite

theorem complementary_event_correct (bag : Bag) (h1 : bag.red = 3) (h2 : bag.white = 2) :
  complementary_event Event.AtLeastOneWhite = Event.AllRed :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_correct_l2093_209356


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l2093_209303

/-- A sequence of three real numbers is geometric if the ratio between consecutive terms is constant. -/
def IsGeometricSequence (x y z : ℝ) : Prop :=
  y / x = z / y

/-- The statement that a = ±6 is equivalent to the sequence 4, a, 9 being geometric. -/
theorem geometric_sequence_condition :
  ∀ a : ℝ, IsGeometricSequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l2093_209303


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l2093_209366

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

theorem tenth_term_of_specific_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l2093_209366
