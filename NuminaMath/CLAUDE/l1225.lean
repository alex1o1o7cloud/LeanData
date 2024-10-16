import Mathlib

namespace NUMINAMATH_CALUDE_mary_final_card_count_l1225_122519

/-- The number of baseball cards Mary has after repairing some torn cards, receiving gifts, and buying new ones. -/
def final_card_count (initial_cards torn_cards repaired_percentage gift_cards bought_cards : ℕ) : ℕ :=
  let repaired_cards := (torn_cards * repaired_percentage) / 100
  let cards_after_repair := initial_cards - torn_cards + repaired_cards
  cards_after_repair + gift_cards + bought_cards

/-- Theorem stating that Mary ends up with 82 baseball cards given the initial conditions. -/
theorem mary_final_card_count :
  final_card_count 18 8 75 26 40 = 82 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_card_count_l1225_122519


namespace NUMINAMATH_CALUDE_train_bus_cost_difference_l1225_122579

/-- The cost difference between a train ride and a bus ride --/
def cost_difference (train_cost bus_cost : ℝ) : ℝ := train_cost - bus_cost

/-- The total cost of a train ride and a bus ride --/
def total_cost (train_cost bus_cost : ℝ) : ℝ := train_cost + bus_cost

theorem train_bus_cost_difference :
  ∃ (train_cost bus_cost : ℝ),
    bus_cost = 1.75 ∧
    total_cost train_cost bus_cost = 9.85 ∧
    cost_difference train_cost bus_cost = 6.35 := by
  sorry

end NUMINAMATH_CALUDE_train_bus_cost_difference_l1225_122579


namespace NUMINAMATH_CALUDE_x_plus_y_squared_l1225_122570

theorem x_plus_y_squared (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : 
  (x + y)^2 = 135 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_squared_l1225_122570


namespace NUMINAMATH_CALUDE_stock_value_decrease_l1225_122517

theorem stock_value_decrease (F : ℝ) (h1 : F > 0) : 
  let J := 0.9 * F
  let M := J / 1.2
  (F - M) / F * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_stock_value_decrease_l1225_122517


namespace NUMINAMATH_CALUDE_intersection_point_l1225_122551

-- Define the system of equations
def system_solution (a b : ℝ) : ℝ × ℝ := (-1, 3)

-- Define the condition that the system solution satisfies the equations
def system_satisfies (a b : ℝ) : Prop :=
  let (x, y) := system_solution a b
  2 * x + y = b ∧ x - y = a

-- Define the lines
def line1 (x : ℝ) (b : ℝ) : ℝ := -2 * x + b
def line2 (x : ℝ) (a : ℝ) : ℝ := x - a

-- State the theorem
theorem intersection_point (a b : ℝ) (h : system_satisfies a b) :
  let (x, y) := system_solution a b
  line1 x b = y ∧ line2 x a = y := by sorry

end NUMINAMATH_CALUDE_intersection_point_l1225_122551


namespace NUMINAMATH_CALUDE_product_of_specific_numbers_l1225_122506

theorem product_of_specific_numbers : 469157 * 9999 = 4690872843 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_numbers_l1225_122506


namespace NUMINAMATH_CALUDE_terry_more_stickers_than_steven_l1225_122540

/-- Given the number of stickers each person has, prove that Terry has 20 more stickers than Steven -/
theorem terry_more_stickers_than_steven 
  (ryan_stickers : ℕ) 
  (steven_stickers : ℕ) 
  (terry_stickers : ℕ) 
  (total_stickers : ℕ) 
  (h1 : ryan_stickers = 30)
  (h2 : steven_stickers = 3 * ryan_stickers)
  (h3 : terry_stickers > steven_stickers)
  (h4 : ryan_stickers + steven_stickers + terry_stickers = total_stickers)
  (h5 : total_stickers = 230) :
  terry_stickers - steven_stickers = 20 := by
sorry

end NUMINAMATH_CALUDE_terry_more_stickers_than_steven_l1225_122540


namespace NUMINAMATH_CALUDE_union_of_sets_l1225_122587

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 5}
  A ∪ B = {1, 2, 3, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1225_122587


namespace NUMINAMATH_CALUDE_problem_solution_l1225_122586

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1225_122586


namespace NUMINAMATH_CALUDE_remaining_doughnuts_theorem_l1225_122550

/-- Represents the types of doughnuts -/
inductive DoughnutType
  | Glazed
  | Chocolate
  | RaspberryFilled

/-- Represents a person who ate doughnuts -/
structure Person where
  glazed : Nat
  chocolate : Nat
  raspberryFilled : Nat

/-- Calculates the remaining doughnuts after consumption -/
def remainingDoughnuts (initial : DoughnutType → Nat) (people : List Person) : DoughnutType → Nat :=
  fun type =>
    initial type - (people.map fun p =>
      match type with
      | DoughnutType.Glazed => p.glazed
      | DoughnutType.Chocolate => p.chocolate
      | DoughnutType.RaspberryFilled => p.raspberryFilled
    ).sum

/-- The main theorem stating the remaining quantities of doughnuts -/
theorem remaining_doughnuts_theorem (initial : DoughnutType → Nat) (people : List Person)
  (h_initial_glazed : initial DoughnutType.Glazed = 10)
  (h_initial_chocolate : initial DoughnutType.Chocolate = 8)
  (h_initial_raspberry : initial DoughnutType.RaspberryFilled = 6)
  (h_people : people = [
    ⟨2, 1, 0⟩, -- Person A
    ⟨1, 0, 0⟩, -- Person B
    ⟨0, 3, 0⟩, -- Person C
    ⟨1, 0, 1⟩, -- Person D
    ⟨0, 0, 1⟩, -- Person E
    ⟨0, 0, 2⟩  -- Person F
  ]) :
  (remainingDoughnuts initial people DoughnutType.Glazed = 6) ∧
  (remainingDoughnuts initial people DoughnutType.Chocolate = 4) ∧
  (remainingDoughnuts initial people DoughnutType.RaspberryFilled = 2) :=
by sorry


end NUMINAMATH_CALUDE_remaining_doughnuts_theorem_l1225_122550


namespace NUMINAMATH_CALUDE_sixth_root_of_six_sqrt_2_over_sqrt_3_of_6_l1225_122547

theorem sixth_root_of_six (x : ℝ) (h : x > 0) : 
  (x^(1/2)) / (x^(1/3)) = x^(1/6) := by
  sorry

-- The specific case for x = 6
theorem sqrt_2_over_sqrt_3_of_6 : 
  (6^(1/2)) / (6^(1/3)) = 6^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_six_sqrt_2_over_sqrt_3_of_6_l1225_122547


namespace NUMINAMATH_CALUDE_painted_face_probability_for_specific_prism_l1225_122560

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in the prism -/
def total_cubes (p : RectangularPrism) : ℕ :=
  p.length * p.width * p.height

/-- Calculates the number of corner cubes -/
def corner_cubes : ℕ := 8

/-- Calculates the number of edge cubes -/
def edge_cubes (p : RectangularPrism) : ℕ :=
  4 * (p.length - 2) + 8 * (p.height - 2)

/-- Calculates the number of face cubes -/
def face_cubes (p : RectangularPrism) : ℕ :=
  2 * (p.length * p.height) - edge_cubes p - corner_cubes

/-- Calculates the probability of a randomly chosen cube showing a painted face when rolled -/
def painted_face_probability (p : RectangularPrism) : ℚ :=
  (3 * corner_cubes + 2 * edge_cubes p + face_cubes p) / (6 * total_cubes p)

theorem painted_face_probability_for_specific_prism :
  let p : RectangularPrism := ⟨20, 1, 7⟩
  painted_face_probability p = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_painted_face_probability_for_specific_prism_l1225_122560


namespace NUMINAMATH_CALUDE_M_subset_N_l1225_122521

-- Define set M
def M : Set ℝ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}

-- Define set N
def N : Set ℝ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1225_122521


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1225_122535

/-- A complex number z satisfying z ⋅ i = 3 + 4i corresponds to a point in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant (z : ℂ) : z * Complex.I = 3 + 4 * Complex.I → z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1225_122535


namespace NUMINAMATH_CALUDE_square_difference_division_l1225_122595

theorem square_difference_division : (245^2 - 225^2) / 20 = 470 := by sorry

end NUMINAMATH_CALUDE_square_difference_division_l1225_122595


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1225_122505

theorem contrapositive_equivalence (m : ℕ+) : 
  (¬(∃ x : ℝ, x^2 + x - m.val = 0) → m.val ≤ 0) ↔ 
  (m.val > 0 → ∃ x : ℝ, x^2 + x - m.val = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1225_122505


namespace NUMINAMATH_CALUDE_olympiad_participants_impossibility_l1225_122516

theorem olympiad_participants_impossibility : ¬ ∃ (x : ℕ), x + (x + 43) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_participants_impossibility_l1225_122516


namespace NUMINAMATH_CALUDE_expression_evaluation_l1225_122575

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1225_122575


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1225_122576

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def numerator_sum : ℚ := arithmetic_sum 3 3 33
def denominator_sum : ℚ := arithmetic_sum 4 4 24

theorem arithmetic_sequences_ratio :
  numerator_sum / denominator_sum = 1683 / 1200 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1225_122576


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1225_122536

/-- 
For a quadratic equation (k-2)x^2 - 2kx + k = 6 to have real roots,
k must satisfy k ≥ 1.5 and k ≠ 2.
-/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1225_122536


namespace NUMINAMATH_CALUDE_integral_tangent_sine_l1225_122548

open Real MeasureTheory

theorem integral_tangent_sine (f : ℝ → ℝ) :
  (∫ x in Set.Icc (π/4) (arctan 3), 1 / ((3 * tan x + 5) * sin (2 * x))) = (1/10) * log (12/7) := by
  sorry

end NUMINAMATH_CALUDE_integral_tangent_sine_l1225_122548


namespace NUMINAMATH_CALUDE_circle_symmetry_min_value_l1225_122572

/-- The minimum value of 1/a + 3/b for a circle symmetric to a line --/
theorem circle_symmetry_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_symmetry : ∃ (x y : ℝ), x^2 + y^2 + 2*x - 6*y + 1 = 0 ∧ a*x - b*y + 3 = 0) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∃ (x y : ℝ), x^2 + y^2 + 2*x - 6*y + 1 = 0 ∧ a'*x - b'*y + 3 = 0) → 
    1/a + 3/b ≤ 1/a' + 3/b') ∧
  (1/a + 3/b = 16/3) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_min_value_l1225_122572


namespace NUMINAMATH_CALUDE_minimum_pass_rate_four_subjects_l1225_122538

theorem minimum_pass_rate_four_subjects 
  (math_pass : Real) (chinese_pass : Real) (english_pass : Real) (chemistry_pass : Real)
  (h_math : math_pass = 0.99)
  (h_chinese : chinese_pass = 0.98)
  (h_english : english_pass = 0.96)
  (h_chemistry : chemistry_pass = 0.92) :
  1 - (1 - math_pass + 1 - chinese_pass + 1 - english_pass + 1 - chemistry_pass) = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_minimum_pass_rate_four_subjects_l1225_122538


namespace NUMINAMATH_CALUDE_pizzas_bought_l1225_122557

theorem pizzas_bought (cost_per_pizza total_paid : ℕ) (h1 : cost_per_pizza = 8) (h2 : total_paid = 24) :
  total_paid / cost_per_pizza = 3 := by
sorry

end NUMINAMATH_CALUDE_pizzas_bought_l1225_122557


namespace NUMINAMATH_CALUDE_f_inverse_g_l1225_122526

noncomputable def f (x : ℝ) : ℝ := 3 - 7*x + x^2

noncomputable def g (x : ℝ) : ℝ := (7 + Real.sqrt (37 + 4*x)) / 2

theorem f_inverse_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_f_inverse_g_l1225_122526


namespace NUMINAMATH_CALUDE_inequality_proof_l1225_122515

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : 
  Real.sqrt (x * y * z) ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1225_122515


namespace NUMINAMATH_CALUDE_sum_of_roots_l1225_122525

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x, x^2 - 12*a*x - 13*b = 0 ↔ x = c ∨ x = d) →
  (∀ x, x^2 - 12*c*x - 13*d = 0 ↔ x = a ∨ x = b) →
  a + b + c + d = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1225_122525


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_T_l1225_122564

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ 1 < x} := by sorry

-- Theorem for the range of T
theorem range_of_T (T : ℝ) :
  (∀ x : ℝ, f x ≥ -T^2 - 5/2*T - 1) →
  (T ≤ -3 ∨ T ≥ 1/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_T_l1225_122564


namespace NUMINAMATH_CALUDE_greatest_integer_for_fraction_l1225_122554

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem greatest_integer_for_fraction : 
  (∀ x : ℤ, x > 14 → ¬is_integer ((x^2 - 5*x + 14) / (x - 4))) ∧
  is_integer ((14^2 - 5*14 + 14) / (14 - 4)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_fraction_l1225_122554


namespace NUMINAMATH_CALUDE_time_after_3250_minutes_l1225_122549

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting date and time -/
def startDateTime : DateTime :=
  { year := 2020, month := 1, day := 1, hour := 3, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3250

/-- The resulting date and time -/
def resultDateTime : DateTime :=
  { year := 2020, month := 1, day := 3, hour := 9, minute := 10 }

theorem time_after_3250_minutes :
  addMinutes startDateTime minutesToAdd = resultDateTime :=
sorry

end NUMINAMATH_CALUDE_time_after_3250_minutes_l1225_122549


namespace NUMINAMATH_CALUDE_folded_quadrilateral_l1225_122543

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if two points coincide after folding -/
def coincide (p1 p2 : Point) : Prop :=
  ∃ (m : Point), (m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2) ∧
  (p2.y - p1.y) * (m.x - p1.x) = (p1.x - p2.x) * (m.y - p1.y)

/-- Calculates the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Main theorem -/
theorem folded_quadrilateral :
  ∀ (m n : ℝ),
  let q := Quadrilateral.mk
    (Point.mk 0 2)  -- A
    (Point.mk 4 0)  -- B
    (Point.mk 7 3)  -- C
    (Point.mk m n)  -- D
  coincide q.A q.B ∧ coincide q.C q.D →
  m = 3/5 ∧ n = 31/5 ∧ area q = 117/5 := by
  sorry

end NUMINAMATH_CALUDE_folded_quadrilateral_l1225_122543


namespace NUMINAMATH_CALUDE_car_rental_daily_rate_l1225_122577

theorem car_rental_daily_rate (weekly_rate : ℕ) (total_days : ℕ) (total_cost : ℕ) : 
  weekly_rate = 190 → total_days = 11 → total_cost = 310 →
  ∃ (daily_rate : ℕ), daily_rate = 30 ∧ total_cost = weekly_rate + daily_rate * (total_days - 7) :=
by sorry

end NUMINAMATH_CALUDE_car_rental_daily_rate_l1225_122577


namespace NUMINAMATH_CALUDE_total_original_cost_of_cars_l1225_122561

/-- Calculates the original price of a car before depreciation -/
def originalPrice (soldPrice : ℚ) (depreciationRate : ℚ) : ℚ :=
  soldPrice / (1 - depreciationRate)

/-- Proves that the total original cost of two cars is $3058.82 -/
theorem total_original_cost_of_cars 
  (oldCarSoldPrice : ℚ) 
  (secondOldestCarSoldPrice : ℚ) 
  (oldCarDepreciationRate : ℚ) 
  (secondOldestCarDepreciationRate : ℚ)
  (h1 : oldCarSoldPrice = 1800)
  (h2 : secondOldestCarSoldPrice = 900)
  (h3 : oldCarDepreciationRate = 1/10)
  (h4 : secondOldestCarDepreciationRate = 3/20) :
  originalPrice oldCarSoldPrice oldCarDepreciationRate + 
  originalPrice secondOldestCarSoldPrice secondOldestCarDepreciationRate = 3058.82 := by
  sorry

#eval originalPrice 1800 (1/10) + originalPrice 900 (3/20)

end NUMINAMATH_CALUDE_total_original_cost_of_cars_l1225_122561


namespace NUMINAMATH_CALUDE_pencils_per_group_l1225_122559

theorem pencils_per_group (total_pencils : ℕ) (num_groups : ℕ) 
  (h1 : total_pencils = 154) (h2 : num_groups = 14) :
  total_pencils / num_groups = 11 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_group_l1225_122559


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1225_122539

/-- Given an ellipse and a hyperbola with related equations, prove that the hyperbola's eccentricity is √6/2 -/
theorem hyperbola_eccentricity
  (m n : ℝ)
  (h_pos : 0 < m ∧ m < n)
  (h_ellipse : ∀ x y : ℝ, m * x^2 + n * y^2 = 1)
  (h_ellipse_ecc : Real.sqrt 2 / 2 = Real.sqrt (1 - (1/n) / (1/m)))
  (h_hyperbola : ∀ x y : ℝ, m * x^2 - n * y^2 = 1) :
  Real.sqrt 6 / 2 = Real.sqrt (1 + (1/n) / (1/m)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1225_122539


namespace NUMINAMATH_CALUDE_last_four_average_l1225_122558

theorem last_four_average (total_count : ℕ) (total_avg : ℝ) (first_five_avg : ℝ) (middle_num : ℝ) :
  total_count = 10 →
  total_avg = 210 →
  first_five_avg = 40 →
  middle_num = 1100 →
  (5 * first_five_avg + middle_num + 4 * (total_count * total_avg - 5 * first_five_avg - middle_num) / 4) / total_count = total_avg →
  (total_count * total_avg - 5 * first_five_avg - middle_num) / 4 = 200 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l1225_122558


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1225_122537

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (18 * x₁) / 27 = 7 / x₁ ∧ 
  (18 * x₂) / 27 = 7 / x₂ ∧ 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1225_122537


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1225_122594

/-- An isosceles triangle with two sides of length a and one side of length b -/
structure IsoscelesTriangle (a b : ℝ) : Type :=
  (side_length : ℝ)
  (base_length : ℝ)
  (is_isosceles : side_length = a ∧ base_length = b)

/-- A triangle similar to another triangle with a given scale factor -/
def SimilarTriangle (T : IsoscelesTriangle a b) (scale : ℝ) : Type :=
  IsoscelesTriangle (T.side_length * scale) (T.base_length * scale)

/-- The perimeter of a triangle -/
def perimeter (T : IsoscelesTriangle a b) : ℝ :=
  2 * T.side_length + T.base_length

theorem similar_triangle_perimeter 
  (T : IsoscelesTriangle 30 15) 
  (S : SimilarTriangle T 5) : 
  perimeter S = 375 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1225_122594


namespace NUMINAMATH_CALUDE_rational_cosine_summands_l1225_122593

theorem rational_cosine_summands (x : ℝ) 
  (h_S : ∃ q : ℚ, q = Real.sin (64 * x) + Real.sin (65 * x))
  (h_C : ∃ q : ℚ, q = Real.cos (64 * x) + Real.cos (65 * x)) :
  ∃ (q1 q2 : ℚ), q1 = Real.cos (64 * x) ∧ q2 = Real.cos (65 * x) :=
sorry

end NUMINAMATH_CALUDE_rational_cosine_summands_l1225_122593


namespace NUMINAMATH_CALUDE_gym_time_calculation_l1225_122510

/-- Calculates the total time spent at the gym per week -/
def gym_time_per_week (visits_per_week : ℕ) (weightlifting_time : ℝ) (warmup_cardio_ratio : ℝ) : ℝ :=
  visits_per_week * (weightlifting_time + warmup_cardio_ratio * weightlifting_time)

/-- Theorem: Given the specified gym routine, the total time spent at the gym per week is 4 hours -/
theorem gym_time_calculation :
  let visits_per_week : ℕ := 3
  let weightlifting_time : ℝ := 1
  let warmup_cardio_ratio : ℝ := 1/3
  gym_time_per_week visits_per_week weightlifting_time warmup_cardio_ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_gym_time_calculation_l1225_122510


namespace NUMINAMATH_CALUDE_calculate_expression_l1225_122501

theorem calculate_expression : 2 * 9 - Real.sqrt 36 + 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1225_122501


namespace NUMINAMATH_CALUDE_power_comparison_l1225_122588

theorem power_comparison : 4^15 = 8^10 ∧ 8^10 < 2^31 := by sorry

end NUMINAMATH_CALUDE_power_comparison_l1225_122588


namespace NUMINAMATH_CALUDE_collinear_vectors_m_values_l1225_122592

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def are_collinear (u v : V) : Prop := ∃ (k : ℝ), u = k • v

theorem collinear_vectors_m_values
  (a b : V)
  (h1 : ¬ are_collinear a b)
  (h2 : ∃ (k : ℝ), (m : ℝ) • a - 3 • b = k • (a + (2 - m) • b)) :
  m = -1 ∨ m = 3 :=
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_values_l1225_122592


namespace NUMINAMATH_CALUDE_cubic_factorization_l1225_122541

theorem cubic_factorization (a : ℝ) : a^3 - 3*a = a*(a + Real.sqrt 3)*(a - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1225_122541


namespace NUMINAMATH_CALUDE_square_of_negative_triple_l1225_122569

theorem square_of_negative_triple (a : ℝ) : (-3 * a)^2 = 9 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_triple_l1225_122569


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1225_122513

/-- An arithmetic sequence with first term 3 and sum of second and third terms 12 has second term equal to 5 -/
theorem arithmetic_sequence_second_term (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 3 →                                -- first term is 3
  a 2 + a 3 = 12 →                         -- sum of second and third terms is 12
  a 2 = 5 :=                               -- second term is 5
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1225_122513


namespace NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l1225_122582

/-- The percentage of copper in the first alloy -/
def first_alloy_copper_percentage : ℝ := 25

/-- The percentage of copper in the second alloy -/
def second_alloy_copper_percentage : ℝ := 50

/-- The weight of the first alloy used -/
def first_alloy_weight : ℝ := 200

/-- The weight of the second alloy used -/
def second_alloy_weight : ℝ := 800

/-- The total weight of the final alloy -/
def total_weight : ℝ := 1000

/-- The percentage of copper in the final alloy -/
def final_alloy_copper_percentage : ℝ := 45

theorem copper_percentage_in_first_alloy :
  (first_alloy_weight * first_alloy_copper_percentage / 100 +
   second_alloy_weight * second_alloy_copper_percentage / 100) / total_weight * 100 =
  final_alloy_copper_percentage :=
by sorry

end NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l1225_122582


namespace NUMINAMATH_CALUDE_probability_red_before_green_l1225_122532

def red_chips : ℕ := 4
def green_chips : ℕ := 3
def total_chips : ℕ := red_chips + green_chips

def favorable_arrangements : ℕ := Nat.choose (total_chips - 1) green_chips
def total_arrangements : ℕ := Nat.choose total_chips green_chips

theorem probability_red_before_green :
  (favorable_arrangements : ℚ) / total_arrangements = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_before_green_l1225_122532


namespace NUMINAMATH_CALUDE_inequality_solution_l1225_122571

theorem inequality_solution (x : ℝ) : 
  1 / (x + 2) + 5 / (x + 4) ≥ 1 ↔ x ∈ Set.Icc (-4 : ℝ) (-3) ∪ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1225_122571


namespace NUMINAMATH_CALUDE_certain_number_problem_l1225_122509

theorem certain_number_problem : ∃ x : ℝ, x * (5^4) = 70000 ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1225_122509


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1225_122578

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (hypotenuse : ℝ) : 
  leg = 15 → 
  angle = 60 * π / 180 → 
  hypotenuse = 10 * Real.sqrt 3 → 
  leg * Real.sin angle = hypotenuse * Real.sin (π / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1225_122578


namespace NUMINAMATH_CALUDE_sum_mod_nine_l1225_122553

theorem sum_mod_nine : (8150 + 8151 + 8152 + 8153 + 8154 + 8155) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l1225_122553


namespace NUMINAMATH_CALUDE_expression_simplification_l1225_122562

theorem expression_simplification :
  let x : ℝ := Real.sqrt 2 + 1
  let expr := ((2 * x - 1) / (x + 1) - x + 1) / ((x - 2) / (x^2 + 2*x + 1))
  expr = -12 * Real.sqrt 2 - 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1225_122562


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l1225_122542

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^3 * (x^2)^(1/2))^(1/4) = x := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l1225_122542


namespace NUMINAMATH_CALUDE_partnership_investment_l1225_122546

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (m : ℝ) : x > 0 → 
  (12 * x + 2 * x * 6 + 3 * x * (12 - m) = 3 * (12 * x)) →
  (12 * x = 18900 / 3) →
  m = 4 := by sorry

end NUMINAMATH_CALUDE_partnership_investment_l1225_122546


namespace NUMINAMATH_CALUDE_max_profit_is_21600_l1225_122512

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Calculates the total profit for a given production quantity -/
def totalProfit (p : Production) : ℝ :=
  2100 * p.a + 900 * p.b

/-- Checks if a production quantity satisfies all constraints -/
def isValid (p : Production) : Prop :=
  p.a ≥ 0 ∧ p.b ≥ 0 ∧
  1.5 * p.a + 0.5 * p.b ≤ 150 ∧
  1 * p.a + 0.3 * p.b ≤ 90 ∧
  5 * p.a + 3 * p.b ≤ 600

/-- Theorem stating that the maximum total profit is 21600 yuan -/
theorem max_profit_is_21600 :
  ∃ (p : Production), isValid p ∧
    totalProfit p = 21600 ∧
    ∀ (q : Production), isValid q → totalProfit q ≤ 21600 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_is_21600_l1225_122512


namespace NUMINAMATH_CALUDE_artichoke_dip_theorem_l1225_122528

/-- The amount of money Hakeem has to spend on artichokes -/
def budget : ℚ := 15

/-- The cost of one artichoke -/
def artichoke_cost : ℚ := 5/4

/-- The number of artichokes needed to make a batch of dip -/
def artichokes_per_batch : ℕ := 3

/-- The amount of dip (in ounces) produced from one batch -/
def dip_per_batch : ℚ := 5

/-- The maximum amount of dip (in ounces) that can be made with the given budget -/
def max_dip : ℚ := 20

theorem artichoke_dip_theorem :
  (budget / artichoke_cost).floor * (dip_per_batch / artichokes_per_batch) = max_dip :=
sorry

end NUMINAMATH_CALUDE_artichoke_dip_theorem_l1225_122528


namespace NUMINAMATH_CALUDE_fraction_equality_l1225_122518

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1225_122518


namespace NUMINAMATH_CALUDE_juanita_sunscreen_usage_l1225_122574

/-- Proves that Juanita uses 1 bottle of sunscreen per month -/
theorem juanita_sunscreen_usage
  (months_per_year : ℕ)
  (discount_rate : ℚ)
  (bottle_cost : ℚ)
  (total_discounted_cost : ℚ)
  (h1 : months_per_year = 12)
  (h2 : discount_rate = 30 / 100)
  (h3 : bottle_cost = 30)
  (h4 : total_discounted_cost = 252) :
  (total_discounted_cost / ((1 - discount_rate) * bottle_cost)) / months_per_year = 1 :=
sorry

end NUMINAMATH_CALUDE_juanita_sunscreen_usage_l1225_122574


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1225_122556

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 4 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2/5 * x ∨ y = -2/5 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(2/5)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1225_122556


namespace NUMINAMATH_CALUDE_bob_walking_distance_l1225_122534

/-- Proves that Bob walked 16 miles when he met Yolanda given the problem conditions --/
theorem bob_walking_distance (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) 
  (head_start : ℝ) : 
  total_distance = 31 ∧ 
  yolanda_speed = 3 ∧ 
  bob_speed = 4 ∧ 
  head_start = 1 →
  ∃ t : ℝ, t > 0 ∧ yolanda_speed * (t + head_start) + bob_speed * t = total_distance ∧ 
  bob_speed * t = 16 :=
by sorry

end NUMINAMATH_CALUDE_bob_walking_distance_l1225_122534


namespace NUMINAMATH_CALUDE_movie_children_count_l1225_122507

/-- Calculates the maximum number of children that can be taken to the movies given the ticket costs and total budget. -/
def max_children (adult_ticket_cost child_ticket_cost total_budget : ℕ) : ℕ :=
  ((total_budget - adult_ticket_cost) / child_ticket_cost)

/-- Theorem stating that given the specific costs and budget, the maximum number of children is 9. -/
theorem movie_children_count :
  let adult_ticket_cost := 8
  let child_ticket_cost := 3
  let total_budget := 35
  max_children adult_ticket_cost child_ticket_cost total_budget = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_children_count_l1225_122507


namespace NUMINAMATH_CALUDE_sum_of_possible_values_l1225_122598

theorem sum_of_possible_values (p q r ℓ : ℂ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → p ≠ q → q ≠ r → r ≠ p →
  p / (1 - q^2) = ℓ → q / (1 - r^2) = ℓ → r / (1 - p^2) = ℓ →
  ∃ (ℓ₁ ℓ₂ : ℂ), (∀ x : ℂ, x = ℓ → x = ℓ₁ ∨ x = ℓ₂) ∧ ℓ₁ + ℓ₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_values_l1225_122598


namespace NUMINAMATH_CALUDE_smallest_palindrome_base2_base4_l1225_122530

/-- Convert a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBinaryAux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBinaryAux (m / 2) ((m % 2) :: acc)
  toBinaryAux n []

/-- Convert a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBase4Aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBase4Aux (m / 4) ((m % 4) :: acc)
  toBase4Aux n []

/-- Check if a list is a palindrome -/
def isPalindrome (l : List ℕ) : Prop :=
  l = l.reverse

/-- The main theorem statement -/
theorem smallest_palindrome_base2_base4 :
  ∀ n : ℕ, n > 10 →
  (isPalindrome (toBinary n) ∧ isPalindrome (toBase4 n)) →
  n ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_palindrome_base2_base4_l1225_122530


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l1225_122514

/-- Given that M(4, 6) is the midpoint of CD and C has coordinates (10, 2),
    prove that the sum of the coordinates of point D is 8. -/
theorem sum_of_coordinates_of_D (C D M : ℝ × ℝ) : 
  C = (10, 2) →
  M = (4, 6) →
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l1225_122514


namespace NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l1225_122508

theorem temperature_difference_product (P : ℝ) : 
  (∃ (A B : ℝ), A = B + P ∧ 
   ∃ (A_t B_t : ℝ), A_t = B + P - 8 ∧ B_t = B + 2 ∧ 
   |A_t - B_t| = 4) →
  (P = 12 ∨ P = 4) :=
by sorry

theorem product_of_possible_P_values : 
  (∀ P : ℝ, (∃ (A B : ℝ), A = B + P ∧ 
   ∃ (A_t B_t : ℝ), A_t = B + P - 8 ∧ B_t = B + 2 ∧ 
   |A_t - B_t| = 4) →
  (P = 12 ∨ P = 4)) →
  12 * 4 = 48 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l1225_122508


namespace NUMINAMATH_CALUDE_childrens_cookbook_cost_l1225_122581

theorem childrens_cookbook_cost (dictionary_cost dinosaur_book_cost savings needed_more total_cost : ℕ) :
  dictionary_cost = 11 →
  dinosaur_book_cost = 19 →
  savings = 8 →
  needed_more = 29 →
  total_cost = savings + needed_more →
  total_cost - (dictionary_cost + dinosaur_book_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_childrens_cookbook_cost_l1225_122581


namespace NUMINAMATH_CALUDE_remaining_laps_after_break_l1225_122596

/-- The number of laps Jeff needs to swim over the weekend -/
def total_laps : ℕ := 98

/-- The number of laps Jeff swam on Saturday -/
def saturday_laps : ℕ := 27

/-- The number of laps Jeff swam on Sunday morning -/
def sunday_morning_laps : ℕ := 15

/-- Theorem stating the number of laps remaining after Jeff's break on Sunday -/
theorem remaining_laps_after_break : 
  total_laps - saturday_laps - sunday_morning_laps = 56 := by
  sorry

end NUMINAMATH_CALUDE_remaining_laps_after_break_l1225_122596


namespace NUMINAMATH_CALUDE_correct_operation_l1225_122545

theorem correct_operation (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = -x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1225_122545


namespace NUMINAMATH_CALUDE_bicycle_separation_l1225_122584

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 12

/-- Simon's speed in miles per hour -/
def simon_speed : ℝ := 16

/-- Time in hours after which Adam and Simon are 100 miles apart -/
def separation_time : ℝ := 5

/-- Distance between Adam and Simon after separation_time hours -/
def separation_distance : ℝ := 100

theorem bicycle_separation :
  let adam_distance := adam_speed * separation_time
  let simon_distance := simon_speed * separation_time
  (adam_distance ^ 2 + simon_distance ^ 2 : ℝ) = separation_distance ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_separation_l1225_122584


namespace NUMINAMATH_CALUDE_garden_area_difference_l1225_122503

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Represents a shed in the garden -/
structure Shed where
  length : ℝ
  width : ℝ

/-- Calculates the area of a shed -/
def shed_area (s : Shed) : ℝ := s.length * s.width

theorem garden_area_difference : 
  let karl_garden : Garden := { length := 30, width := 50 }
  let makenna_garden : Garden := { length := 35, width := 55 }
  let makenna_shed : Shed := { length := 5, width := 10 }
  (garden_area makenna_garden - shed_area makenna_shed) - garden_area karl_garden = 375 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l1225_122503


namespace NUMINAMATH_CALUDE_sum_a_b_is_negative_two_l1225_122563

theorem sum_a_b_is_negative_two (a b : ℝ) (h : |a - 1| + (b + 3)^2 = 0) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_is_negative_two_l1225_122563


namespace NUMINAMATH_CALUDE_angle_relation_l1225_122597

theorem angle_relation (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan (α - β) = 1/2) (h4 : Real.tan β = -1/7) :
  2*α - β = -3*π/4 := by sorry

end NUMINAMATH_CALUDE_angle_relation_l1225_122597


namespace NUMINAMATH_CALUDE_bryan_total_books_l1225_122585

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 9

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 56

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that Bryan has 504 books in total -/
theorem bryan_total_books : total_books = 504 := by sorry

end NUMINAMATH_CALUDE_bryan_total_books_l1225_122585


namespace NUMINAMATH_CALUDE_point_slope_problem_l1225_122523

/-- Given two points A(2, b) and B(3, -2) on a line with slope -1, prove that b = -1 -/
theorem point_slope_problem (b : ℝ) : 
  (let A : ℝ × ℝ := (2, b)
   let B : ℝ × ℝ := (3, -2)
   (B.2 - A.2) / (B.1 - A.1) = -1) → b = -1 := by
sorry

end NUMINAMATH_CALUDE_point_slope_problem_l1225_122523


namespace NUMINAMATH_CALUDE_correct_guess_and_multiply_l1225_122502

def coin_head_prob : ℚ := 2/3
def aaron_head_guess_prob : ℚ := 2/3

def correct_guess_prob : ℚ := 
  coin_head_prob * aaron_head_guess_prob + (1 - coin_head_prob) * (1 - aaron_head_guess_prob)

theorem correct_guess_and_multiply :
  correct_guess_prob = 5/9 ∧ 9000 * correct_guess_prob = 5000 := by sorry

end NUMINAMATH_CALUDE_correct_guess_and_multiply_l1225_122502


namespace NUMINAMATH_CALUDE_fence_coloring_theorem_l1225_122599

/-- A coloring of a fence is valid if any two boards separated by exactly 2, 3, or 5 boards
    are painted in different colors. -/
def is_valid_coloring (coloring : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, (coloring i ≠ coloring (i + 3)) ∧
           (coloring i ≠ coloring (i + 4)) ∧
           (coloring i ≠ coloring (i + 6))

/-- The minimum number of colors required to paint the fence -/
def min_colors : ℕ := 3

theorem fence_coloring_theorem :
  (∃ coloring : ℕ → ℕ, is_valid_coloring coloring ∧ (∀ i : ℕ, coloring i < min_colors)) ∧
  (∀ n : ℕ, n < min_colors → ¬∃ coloring : ℕ → ℕ, is_valid_coloring coloring ∧ (∀ i : ℕ, coloring i < n)) :=
sorry

end NUMINAMATH_CALUDE_fence_coloring_theorem_l1225_122599


namespace NUMINAMATH_CALUDE_fudge_difference_l1225_122531

/-- Converts pounds to ounces -/
def pounds_to_ounces (pounds : ℚ) : ℚ := 16 * pounds

theorem fudge_difference (marina_fudge : ℚ) (lazlo_fudge : ℚ) : 
  marina_fudge = 4.5 →
  lazlo_fudge = pounds_to_ounces 4 - 6 →
  pounds_to_ounces marina_fudge - lazlo_fudge = 14 := by
  sorry

end NUMINAMATH_CALUDE_fudge_difference_l1225_122531


namespace NUMINAMATH_CALUDE_cosine_square_expansion_l1225_122524

theorem cosine_square_expansion (z : ℝ) (h : z ≥ 3) :
  (3 - Real.cos (Real.sqrt (z^2 - 9)))^2 = 9 - 6 * Real.cos (Real.sqrt (z^2 - 9)) + (Real.cos (Real.sqrt (z^2 - 9)))^2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_square_expansion_l1225_122524


namespace NUMINAMATH_CALUDE_sum_and_product_500_l1225_122580

theorem sum_and_product_500 (x y : ℤ) : 
  x + y + x * y = 500 ↔ (x = 0 ∧ y = 500) ∨ (x = -2 ∧ y = -502) ∨ (x = 2 ∧ y = 166) ∨ (x = -4 ∧ y = -168) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_500_l1225_122580


namespace NUMINAMATH_CALUDE_atLeastOneMale_and_allFemales_mutuallyExclusive_l1225_122555

/-- Represents the outcome of selecting 2 students from the group -/
inductive Selection
| TwoMales
| OneMaleOneFemale
| TwoFemales

/-- The sample space of all possible selections -/
def sampleSpace : Set Selection :=
  {Selection.TwoMales, Selection.OneMaleOneFemale, Selection.TwoFemales}

/-- The event "At least 1 male student" -/
def atLeastOneMale : Set Selection :=
  {Selection.TwoMales, Selection.OneMaleOneFemale}

/-- The event "All female students" -/
def allFemales : Set Selection :=
  {Selection.TwoFemales}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set Selection) : Prop :=
  A ∩ B = ∅

theorem atLeastOneMale_and_allFemales_mutuallyExclusive :
  mutuallyExclusive atLeastOneMale allFemales :=
sorry

end NUMINAMATH_CALUDE_atLeastOneMale_and_allFemales_mutuallyExclusive_l1225_122555


namespace NUMINAMATH_CALUDE_number_solution_l1225_122504

theorem number_solution : ∃ x : ℝ, (50 + 5 * 12 / (180 / x) = 51) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l1225_122504


namespace NUMINAMATH_CALUDE_debby_soda_bottles_l1225_122533

/-- The number of soda bottles Debby drinks per day -/
def soda_per_day : ℕ := 9

/-- The number of days the soda bottles lasted -/
def days_lasted : ℕ := 40

/-- The total number of soda bottles Debby bought -/
def total_soda_bottles : ℕ := soda_per_day * days_lasted

/-- Theorem stating that the total number of soda bottles Debby bought is 360 -/
theorem debby_soda_bottles : total_soda_bottles = 360 := by
  sorry

end NUMINAMATH_CALUDE_debby_soda_bottles_l1225_122533


namespace NUMINAMATH_CALUDE_equation_solutions_l1225_122522

theorem equation_solutions :
  (∃ x : ℝ, 3 * x^3 - 15 = 9 ∧ x = 2) ∧
  (∃ x₁ x₂ : ℝ, 2 * (x₁ - 1)^2 = 72 ∧ 2 * (x₂ - 1)^2 = 72 ∧ x₁ = 7 ∧ x₂ = -5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1225_122522


namespace NUMINAMATH_CALUDE_dolphin_ratio_l1225_122511

theorem dolphin_ratio (initial_dolphins final_dolphins : ℕ) 
  (h1 : initial_dolphins = 65)
  (h2 : final_dolphins = 260) :
  (final_dolphins - initial_dolphins) / initial_dolphins = 3 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_ratio_l1225_122511


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1225_122527

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1225_122527


namespace NUMINAMATH_CALUDE_xy_unique_values_l1225_122520

def X : Finset ℤ := {2, 3, 7}
def Y : Finset ℤ := {-31, -24, 4}

theorem xy_unique_values : 
  Finset.card (Finset.image (λ (p : ℤ × ℤ) => p.1 * p.2) (X.product Y)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_unique_values_l1225_122520


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l1225_122544

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 3) : 
  (a * b) / (a + b + 1) + (a * c) / (a + c + 1) + (b * c) / (b + c + 1) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l1225_122544


namespace NUMINAMATH_CALUDE_push_up_difference_l1225_122566

theorem push_up_difference (zachary_pushups david_pushups : ℕ) 
  (h1 : zachary_pushups = 19)
  (h2 : david_pushups = 58) :
  david_pushups - zachary_pushups = 39 := by
  sorry

end NUMINAMATH_CALUDE_push_up_difference_l1225_122566


namespace NUMINAMATH_CALUDE_min_orders_is_three_l1225_122573

/-- Represents the shopping problem with given conditions -/
structure ShoppingProblem where
  item_price : ℕ  -- Original price of each item in yuan
  item_count : ℕ  -- Number of items
  discount_rate : ℚ  -- Discount rate (e.g., 0.6 for 60% off)
  additional_discount_threshold : ℕ  -- Threshold for additional discount in yuan
  additional_discount_amount : ℕ  -- Amount of additional discount in yuan

/-- Calculates the total cost after discounts for a given number of orders -/
def total_cost (problem : ShoppingProblem) (num_orders : ℕ) : ℚ :=
  sorry

/-- Theorem stating that 3 is the minimum number of orders that minimizes the total cost -/
theorem min_orders_is_three (problem : ShoppingProblem) 
  (h1 : problem.item_price = 48)
  (h2 : problem.item_count = 42)
  (h3 : problem.discount_rate = 0.6)
  (h4 : problem.additional_discount_threshold = 300)
  (h5 : problem.additional_discount_amount = 100) :
  ∀ n : ℕ, n ≠ 3 → total_cost problem 3 ≤ total_cost problem n :=
sorry

end NUMINAMATH_CALUDE_min_orders_is_three_l1225_122573


namespace NUMINAMATH_CALUDE_exam_questions_unique_solution_l1225_122583

theorem exam_questions_unique_solution (n : ℕ) : 
  (15 + (n - 20) / 3 : ℚ) / n = 1 / 2 → n = 50 :=
by sorry

end NUMINAMATH_CALUDE_exam_questions_unique_solution_l1225_122583


namespace NUMINAMATH_CALUDE_f_3_range_l1225_122589

/-- Given a quadratic function f(x) = ax^2 - c satisfying certain conditions,
    prove that f(3) is within a specific range. -/
theorem f_3_range (a c : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 - c)
    (h_1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
    (h_2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_f_3_range_l1225_122589


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1225_122591

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1225_122591


namespace NUMINAMATH_CALUDE_equation_solutions_l1225_122567

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 = x ↔ x = 0 ∨ x = 1/4) ∧
  (∀ x : ℝ, x^2 - 18*x + 1 = 0 ↔ x = 9 + 4*Real.sqrt 5 ∨ x = 9 - 4*Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1225_122567


namespace NUMINAMATH_CALUDE_set_representability_l1225_122568

-- Define the items
def item1 : Type := Unit  -- Placeholder for vague concept
def item2 : Set ℝ := {x : ℝ | x^2 + 3 = 0}
def item3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = p.2}

-- Define a predicate for set representability
def is_set_representable (α : Type) : Prop := Nonempty (Set α)

-- State the theorem
theorem set_representability :
  ¬ is_set_representable item1 ∧ 
  is_set_representable item2 ∧ 
  is_set_representable item3 :=
sorry

end NUMINAMATH_CALUDE_set_representability_l1225_122568


namespace NUMINAMATH_CALUDE_valid_configuration_exists_l1225_122565

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a line containing 4 points --/
structure Line where
  points : Fin 4 → Point

/-- The configuration of ships --/
structure ShipConfiguration where
  ships : Fin 10 → Point
  lines : Fin 5 → Line

/-- Checks if a line contains 4 distinct points from the given set of points --/
def Line.isValidLine (l : Line) (points : Fin 10 → Point) : Prop :=
  ∃ (indices : Fin 4 → Fin 10), (∀ i j, i ≠ j → indices i ≠ indices j) ∧
    (∀ i, l.points i = points (indices i))

/-- Checks if a configuration is valid --/
def ShipConfiguration.isValid (config : ShipConfiguration) : Prop :=
  ∀ l, config.lines l |>.isValidLine config.ships

/-- The theorem stating that a valid configuration exists --/
theorem valid_configuration_exists : ∃ (config : ShipConfiguration), config.isValid := by
  sorry


end NUMINAMATH_CALUDE_valid_configuration_exists_l1225_122565


namespace NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l1225_122529

theorem complex_sum_and_reciprocal : 
  let z : ℂ := 1 - I
  (z⁻¹ + z) = (3/2 : ℂ) - (1/2 : ℂ) * I := by sorry

end NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l1225_122529


namespace NUMINAMATH_CALUDE_garden_length_l1225_122500

/-- Proves that the length of the larger garden is 90 yards given the conditions -/
theorem garden_length (w : ℝ) (l : ℝ) : 
  l = 3 * w →  -- larger garden length is three times its width
  360 = 2 * l + 2 * w + 2 * (w / 2) + 2 * (l / 2) →  -- total fencing equals 360 yards
  l = 90 := by
  sorry

#check garden_length

end NUMINAMATH_CALUDE_garden_length_l1225_122500


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1225_122590

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 4 ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ min)) ∧
  (1/a + 1/b = 4 ↔ a = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1225_122590


namespace NUMINAMATH_CALUDE_power_function_property_l1225_122552

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 9 / f 3 = 2) : 
  f (1/9) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l1225_122552
