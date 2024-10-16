import Mathlib

namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1785_178597

theorem fractional_equation_solution_range (m x : ℝ) : 
  (m / (2 * x - 1) + 3 = 0) → 
  (x > 0) → 
  (m < 3 ∧ m ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1785_178597


namespace NUMINAMATH_CALUDE_combined_cost_increase_percentage_l1785_178522

/-- The percent increase in the combined cost of a bicycle and helmet --/
theorem combined_cost_increase_percentage
  (bicycle_cost : ℝ)
  (helmet_cost : ℝ)
  (bicycle_increase_percent : ℝ)
  (helmet_increase_percent : ℝ)
  (h1 : bicycle_cost = 160)
  (h2 : helmet_cost = 40)
  (h3 : bicycle_increase_percent = 5)
  (h4 : helmet_increase_percent = 10) :
  let new_bicycle_cost := bicycle_cost * (1 + bicycle_increase_percent / 100)
  let new_helmet_cost := helmet_cost * (1 + helmet_increase_percent / 100)
  let original_total := bicycle_cost + helmet_cost
  let new_total := new_bicycle_cost + new_helmet_cost
  (new_total - original_total) / original_total * 100 = 6 := by
  sorry

#check combined_cost_increase_percentage

end NUMINAMATH_CALUDE_combined_cost_increase_percentage_l1785_178522


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1785_178543

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  -- The length of the altitude to the base
  altitude : ℝ
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The triangle is isosceles
  isIsosceles : True

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry -- The actual calculation of the area would go here

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_area_l1785_178543


namespace NUMINAMATH_CALUDE_policemen_cover_all_streets_l1785_178550

-- Define the type for intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the type for streets
inductive Street : Type
| ABCD | EFG | HIJK  -- Horizontal
| AEH | BFI | DGJ    -- Vertical
| HFC | CGK          -- Diagonal

-- Define a function to check if an intersection is on a street
def isOnStreet (i : Intersection) (s : Street) : Prop :=
  match s with
  | Street.ABCD => i = Intersection.A ∨ i = Intersection.B ∨ i = Intersection.C ∨ i = Intersection.D
  | Street.EFG => i = Intersection.E ∨ i = Intersection.F ∨ i = Intersection.G
  | Street.HIJK => i = Intersection.H ∨ i = Intersection.I ∨ i = Intersection.J ∨ i = Intersection.K
  | Street.AEH => i = Intersection.A ∨ i = Intersection.E ∨ i = Intersection.H
  | Street.BFI => i = Intersection.B ∨ i = Intersection.F ∨ i = Intersection.I
  | Street.DGJ => i = Intersection.D ∨ i = Intersection.G ∨ i = Intersection.J
  | Street.HFC => i = Intersection.H ∨ i = Intersection.F ∨ i = Intersection.C
  | Street.CGK => i = Intersection.C ∨ i = Intersection.G ∨ i = Intersection.K

-- Define a function to check if a street is covered by a set of intersections
def isCovered (s : Street) (placements : List Intersection) : Prop :=
  ∃ i ∈ placements, isOnStreet i s

-- Theorem: Placing policemen at B, G, and H covers all streets
theorem policemen_cover_all_streets :
  let placements := [Intersection.B, Intersection.G, Intersection.H]
  ∀ s : Street, isCovered s placements :=
by
  sorry


end NUMINAMATH_CALUDE_policemen_cover_all_streets_l1785_178550


namespace NUMINAMATH_CALUDE_kiran_work_completion_l1785_178517

/-- Given that Kiran completes 1/3 of the work in 6 days, prove that he will finish the remaining work in 12 days. -/
theorem kiran_work_completion (work_rate : ℝ) (h1 : work_rate * 6 = 1/3) : 
  work_rate * 12 = 2/3 := by sorry

end NUMINAMATH_CALUDE_kiran_work_completion_l1785_178517


namespace NUMINAMATH_CALUDE_sin_sum_identity_l1785_178586

theorem sin_sum_identity : 
  Real.sin (13 * π / 180) * Real.sin (58 * π / 180) + 
  Real.sin (77 * π / 180) * Real.sin (32 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l1785_178586


namespace NUMINAMATH_CALUDE_parcel_weight_proof_l1785_178584

theorem parcel_weight_proof (x y z : ℝ) 
  (h1 : x + y = 132)
  (h2 : y + z = 145)
  (h3 : z + x = 150) :
  x + y + z = 213.5 := by
sorry

end NUMINAMATH_CALUDE_parcel_weight_proof_l1785_178584


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1785_178506

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  a 3 + a 4 + a 5 + a 6 + a 7 = 45 →
  a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1785_178506


namespace NUMINAMATH_CALUDE_expression_simplification_l1785_178596

theorem expression_simplification (x y : ℝ) : 
  3 * x + 4 * y^2 + 2 - (5 - 3 * x - 2 * y^2) = 6 * x + 6 * y^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1785_178596


namespace NUMINAMATH_CALUDE_factor_expression_l1785_178535

theorem factor_expression (x y a b : ℝ) :
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1785_178535


namespace NUMINAMATH_CALUDE_sqrt_x_plus_2y_is_plus_minus_one_l1785_178574

theorem sqrt_x_plus_2y_is_plus_minus_one (x y : ℝ) 
  (h : Real.sqrt (x - 2) + abs (2 * y + 1) = 0) : 
  Real.sqrt (x + 2 * y) = 1 ∨ Real.sqrt (x + 2 * y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_2y_is_plus_minus_one_l1785_178574


namespace NUMINAMATH_CALUDE_amy_doll_cost_l1785_178521

def doll_cost (initial_amount : ℕ) (dolls_bought : ℕ) (remaining_amount : ℕ) : ℚ :=
  (initial_amount - remaining_amount : ℚ) / dolls_bought

theorem amy_doll_cost :
  doll_cost 100 3 97 = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_doll_cost_l1785_178521


namespace NUMINAMATH_CALUDE_kamals_biology_marks_l1785_178563

-- Define the known marks and average
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def average_marks : ℕ := 74
def num_subjects : ℕ := 5

-- Define the theorem
theorem kamals_biology_marks :
  let total_marks := average_marks * num_subjects
  let known_marks_sum := english_marks + math_marks + physics_marks + chemistry_marks
  let biology_marks := total_marks - known_marks_sum
  biology_marks = 85 := by sorry

end NUMINAMATH_CALUDE_kamals_biology_marks_l1785_178563


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l1785_178571

theorem average_of_three_numbers (y : ℝ) : 
  (15 + 28 + y) / 3 = 25 → y = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l1785_178571


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l1785_178590

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a, b > 0,
    if its eccentricity e and the slope of its asymptotes k satisfy e = √2 |k|,
    then the equation of its asymptotes is y = ±x -/
theorem hyperbola_asymptotes_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let k := b / a
  e = Real.sqrt 2 * abs k →
  ∃ (f : ℝ → ℝ), (∀ x, f x = x ∨ f x = -x) ∧
    (∀ x y, y = f x ↔ (x^2 / a^2 - y^2 / b^2 = 1 → False)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l1785_178590


namespace NUMINAMATH_CALUDE_f_sequence_pairwise_coprime_l1785_178509

/-- The function f(x) = x^2002 - x^2001 + 1 -/
def f (x : ℕ) : ℕ := x^2002 - x^2001 + 1

/-- The sequence generated by repeatedly applying f to m -/
def f_sequence (m : ℕ) : ℕ → ℕ
  | 0 => m
  | n + 1 => f (f_sequence m n)

/-- Two natural numbers are coprime -/
def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The main theorem stating that all elements in the sequence are pairwise coprime -/
theorem f_sequence_pairwise_coprime (m : ℕ+) :
  ∀ i j, i ≠ j → are_coprime (f_sequence m i) (f_sequence m j) :=
sorry

end NUMINAMATH_CALUDE_f_sequence_pairwise_coprime_l1785_178509


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l1785_178593

theorem max_value_on_ellipse :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 → 2*x + y ≤ Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l1785_178593


namespace NUMINAMATH_CALUDE_system_solution_l1785_178588

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 7) (eq2 : x + 2 * y = 8) : x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1785_178588


namespace NUMINAMATH_CALUDE_nuts_to_raisins_cost_ratio_l1785_178513

/-- The ratio of the cost of nuts to raisins given the mixture proportions and cost ratio -/
theorem nuts_to_raisins_cost_ratio 
  (raisin_pounds : ℝ) 
  (nuts_pounds : ℝ)
  (raisin_cost : ℝ)
  (nuts_cost : ℝ)
  (h1 : raisin_pounds = 3)
  (h2 : nuts_pounds = 4)
  (h3 : raisin_cost > 0)
  (h4 : nuts_cost > 0)
  (h5 : raisin_pounds * raisin_cost = 0.15789473684210525 * (raisin_pounds * raisin_cost + nuts_pounds * nuts_cost)) :
  nuts_cost / raisin_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_nuts_to_raisins_cost_ratio_l1785_178513


namespace NUMINAMATH_CALUDE_diamond_jewel_percentage_is_35_percent_l1785_178529

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percent : ℝ
  ruby_jewel_percent : ℝ
  diamond_jewel_percent : ℝ

/-- Calculates the percentage of diamond jewels in the urn -/
def diamond_jewel_percentage (u : UrnComposition) : ℝ :=
  u.diamond_jewel_percent

/-- The theorem stating the percentage of diamond jewels in the urn -/
theorem diamond_jewel_percentage_is_35_percent (u : UrnComposition) 
  (h1 : u.bead_percent = 30)
  (h2 : u.ruby_jewel_percent = 35)
  (h3 : u.bead_percent + u.ruby_jewel_percent + u.diamond_jewel_percent = 100) :
  diamond_jewel_percentage u = 35 := by
  sorry

#check diamond_jewel_percentage_is_35_percent

end NUMINAMATH_CALUDE_diamond_jewel_percentage_is_35_percent_l1785_178529


namespace NUMINAMATH_CALUDE_right_triangle_area_l1785_178549

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 15) (h_side : a = 12) : (1/2) * a * b = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1785_178549


namespace NUMINAMATH_CALUDE_ski_trips_theorem_l1785_178557

/-- Represents the ski lift problem -/
structure SkiLiftProblem where
  lift_time : ℕ  -- Time to ride the lift up (in minutes)
  ski_time : ℕ   -- Time to ski down (in minutes)
  known_trips : ℕ  -- Known number of trips in 2 hours
  known_hours : ℕ  -- Known number of hours for known_trips

/-- Calculates the number of ski trips possible in a given number of hours -/
def ski_trips (problem : SkiLiftProblem) (hours : ℕ) : ℕ :=
  3 * hours

/-- Theorem stating the relationship between hours and number of ski trips -/
theorem ski_trips_theorem (problem : SkiLiftProblem) (hours : ℕ) :
  problem.lift_time = 15 →
  problem.ski_time = 5 →
  problem.known_trips = 6 →
  problem.known_hours = 2 →
  ski_trips problem hours = 3 * hours :=
by
  sorry

#check ski_trips_theorem

end NUMINAMATH_CALUDE_ski_trips_theorem_l1785_178557


namespace NUMINAMATH_CALUDE_calculation_proof_l1785_178594

theorem calculation_proof :
  (1) * (-3)^2 - (-1)^3 - (-2) - |(-12)| = 0 ∧
  -2^2 * 3 * (-3/2) / (2/3) - 4 * (-3/2)^2 = 18 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1785_178594


namespace NUMINAMATH_CALUDE_green_marbles_count_l1785_178564

theorem green_marbles_count (G : ℕ) : 
  (2 / (2 + G : ℝ)) * (1 / (1 + G : ℝ)) = 0.1 → G = 3 := by
sorry

end NUMINAMATH_CALUDE_green_marbles_count_l1785_178564


namespace NUMINAMATH_CALUDE_binomial_square_condition_l1785_178575

theorem binomial_square_condition (a : ℝ) : 
  (∃ (p q : ℝ), ∀ x, 4*x^2 + 16*x + a = (p*x + q)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l1785_178575


namespace NUMINAMATH_CALUDE_inequality_proof_l1785_178556

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1785_178556


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_length_l1785_178591

/-- Given a rectangle made from a rope of length 100cm with longer sides of 28cm,
    the length of each shorter side is 22cm. -/
theorem rectangle_shorter_side_length
  (total_length : ℝ)
  (longer_side : ℝ)
  (h1 : total_length = 100)
  (h2 : longer_side = 28)
  : (total_length - 2 * longer_side) / 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_length_l1785_178591


namespace NUMINAMATH_CALUDE_straw_purchase_solution_l1785_178533

/-- Represents the cost and quantity of straws --/
structure StrawPurchase where
  costA : ℚ  -- Cost per pack of type A straws
  costB : ℚ  -- Cost per pack of type B straws
  maxA : ℕ   -- Maximum number of type A straws that can be purchased

/-- Verifies if the given costs satisfy the purchase scenarios --/
def satisfiesPurchaseScenarios (sp : StrawPurchase) : Prop :=
  12 * sp.costA + 15 * sp.costB = 171 ∧
  24 * sp.costA + 28 * sp.costB = 332

/-- Checks if the maximum number of type A straws satisfies the constraints --/
def satisfiesConstraints (sp : StrawPurchase) : Prop :=
  sp.maxA ≤ 100 ∧
  sp.costA * sp.maxA + sp.costB * (100 - sp.maxA) ≤ 600 ∧
  ∀ m : ℕ, m > sp.maxA → sp.costA * m + sp.costB * (100 - m) > 600

/-- Theorem stating the solution to the straw purchase problem --/
theorem straw_purchase_solution :
  ∃ sp : StrawPurchase,
    sp.costA = 8 ∧ sp.costB = 5 ∧ sp.maxA = 33 ∧
    satisfiesPurchaseScenarios sp ∧
    satisfiesConstraints sp := by
  sorry

end NUMINAMATH_CALUDE_straw_purchase_solution_l1785_178533


namespace NUMINAMATH_CALUDE_at_least_four_boxes_same_items_l1785_178561

theorem at_least_four_boxes_same_items (boxes : Finset Nat) (items : Nat → Nat) : 
  boxes.card = 376 → 
  (∀ b ∈ boxes, items b ≤ 125) → 
  ∃ n : Nat, ∃ same_boxes : Finset Nat, same_boxes ⊆ boxes ∧ same_boxes.card ≥ 4 ∧ 
    ∀ b ∈ same_boxes, items b = n :=
by sorry

end NUMINAMATH_CALUDE_at_least_four_boxes_same_items_l1785_178561


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_99_l1785_178579

theorem greatest_prime_factor_of_99 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 99 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 99 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_99_l1785_178579


namespace NUMINAMATH_CALUDE_mans_walking_rate_l1785_178552

theorem mans_walking_rate 
  (woman_speed : ℝ) 
  (woman_travel_time : ℝ) 
  (woman_wait_time : ℝ) 
  (h1 : woman_speed = 15) 
  (h2 : woman_travel_time = 2 / 60) 
  (h3 : woman_wait_time = 4 / 60) : 
  ∃ (man_speed : ℝ), 
    woman_speed * woman_travel_time = man_speed * (woman_travel_time + woman_wait_time) ∧ 
    man_speed = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_mans_walking_rate_l1785_178552


namespace NUMINAMATH_CALUDE_impossibility_of_1x3_rectangle_l1785_178559

/-- Represents a cell on the grid -/
structure Cell :=
  (x : Fin 8)
  (y : Fin 8)

/-- Represents a 1x2 rectangle on the grid -/
structure Rectangle :=
  (topLeft : Cell)
  (isVertical : Bool)

/-- Checks if a cell is covered by a rectangle -/
def isCovered (c : Cell) (r : Rectangle) : Prop :=
  (c.x = r.topLeft.x ∧ c.y = r.topLeft.y) ∨
  (r.isVertical ∧ c.x = r.topLeft.x ∧ c.y = r.topLeft.y + 1) ∨
  (¬r.isVertical ∧ c.x = r.topLeft.x + 1 ∧ c.y = r.topLeft.y)

/-- Checks if three consecutive cells form a 1x3 rectangle -/
def is1x3Rectangle (c1 c2 c3 : Cell) : Prop :=
  (c1.x = c2.x ∧ c2.x = c3.x ∧ c2.y = c1.y + 1 ∧ c3.y = c2.y + 1) ∨
  (c1.y = c2.y ∧ c2.y = c3.y ∧ c2.x = c1.x + 1 ∧ c3.x = c2.x + 1)

/-- The main theorem -/
theorem impossibility_of_1x3_rectangle :
  ∃ (configuration : Finset Rectangle),
    configuration.card = 12 ∧
    (∀ c1 c2 c3 : Cell,
      is1x3Rectangle c1 c2 c3 →
      ∃ r ∈ configuration, isCovered c1 r ∨ isCovered c2 r ∨ isCovered c3 r) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_1x3_rectangle_l1785_178559


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1785_178528

/-- Given a line l: y = k(x + 1/2) and a circle C: x^2 + y^2 = 1,
    prove that the line always intersects the circle for any real k. -/
theorem line_intersects_circle (k : ℝ) : 
  ∃ x y : ℝ, y = k * (x + 1/2) ∧ x^2 + y^2 = 1 := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_line_intersects_circle_l1785_178528


namespace NUMINAMATH_CALUDE_object_length_doubles_on_day_two_l1785_178565

/-- Calculates the length multiplier after n days -/
def lengthMultiplier (n : ℕ) : ℚ :=
  (n + 2 : ℚ) / 2

theorem object_length_doubles_on_day_two :
  ∃ n : ℕ, lengthMultiplier n = 2 ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_object_length_doubles_on_day_two_l1785_178565


namespace NUMINAMATH_CALUDE_biotechnology_graduates_l1785_178554

theorem biotechnology_graduates (total : ℕ) (job : ℕ) (second_degree : ℕ) (neither : ℕ) :
  total = 73 →
  job = 32 →
  second_degree = 45 →
  neither = 9 →
  ∃ (both : ℕ), both = 13 ∧ job + second_degree - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_biotechnology_graduates_l1785_178554


namespace NUMINAMATH_CALUDE_number_of_dogs_l1785_178540

theorem number_of_dogs (total_legs : ℕ) (num_humans : ℕ) (human_legs : ℕ) (dog_legs : ℕ)
  (h1 : total_legs = 24)
  (h2 : num_humans = 2)
  (h3 : human_legs = 2)
  (h4 : dog_legs = 4) :
  (total_legs - num_humans * human_legs) / dog_legs = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_dogs_l1785_178540


namespace NUMINAMATH_CALUDE_barChartMostEffective_l1785_178507

-- Define an enumeration for chart types
inductive ChartType
  | BarChart
  | LineChart
  | PieChart

-- Define a function to evaluate the effectiveness of a chart type for comparing quantities
def effectivenessForQuantityComparison (chart : ChartType) : Nat :=
  match chart with
  | ChartType.BarChart => 3
  | ChartType.LineChart => 2
  | ChartType.PieChart => 1

-- Theorem stating that BarChart is the most effective for quantity comparison
theorem barChartMostEffective :
  ∀ (chart : ChartType),
    chart ≠ ChartType.BarChart →
    effectivenessForQuantityComparison ChartType.BarChart > effectivenessForQuantityComparison chart :=
by
  sorry


end NUMINAMATH_CALUDE_barChartMostEffective_l1785_178507


namespace NUMINAMATH_CALUDE_productivity_increase_l1785_178577

theorem productivity_increase (original_hours new_hours : ℝ) 
  (wage_increase : ℝ) (productivity_increase : ℝ) : 
  original_hours = 8 → 
  new_hours = 7 → 
  wage_increase = 0.05 →
  (new_hours / original_hours) * (1 + productivity_increase) = 1 + wage_increase →
  productivity_increase = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_productivity_increase_l1785_178577


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l1785_178500

theorem sqrt_fraction_equality : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l1785_178500


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1785_178567

/-- Represents a number in a given base with repeated digits -/
def repeatedDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid for a given base -/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (A C : Nat),
    isValidDigit A 8 ∧
    isValidDigit C 6 ∧
    repeatedDigitNumber A 8 = repeatedDigitNumber C 6 ∧
    repeatedDigitNumber A 8 = 19 ∧
    (∀ (A' C' : Nat),
      isValidDigit A' 8 →
      isValidDigit C' 6 →
      repeatedDigitNumber A' 8 = repeatedDigitNumber C' 6 →
      repeatedDigitNumber A' 8 ≥ 19) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1785_178567


namespace NUMINAMATH_CALUDE_polynomial_ratio_theorem_l1785_178502

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^2009 - 19*x^2008 + 1

-- Define the set of distinct zeros of f
def zeros (f : ℝ → ℝ) : Set ℝ := {x | f x = 0}

-- Define the polynomial P
def P (z : ℝ) : ℝ := sorry

-- Theorem statement
theorem polynomial_ratio_theorem 
  (h1 : ∀ r ∈ zeros f, P (r - 1/r) = 0) 
  (h2 : Fintype (zeros f)) 
  (h3 : Fintype.card (zeros f) = 2009) :
  P 2 / P (-2) = 36 / 49 := by sorry

end NUMINAMATH_CALUDE_polynomial_ratio_theorem_l1785_178502


namespace NUMINAMATH_CALUDE_line_slope_equals_k_l1785_178587

/-- Given a line passing through points (-1, -4) and (5, k), 
    if the slope of the line is equal to k, then k = 4/5 -/
theorem line_slope_equals_k (k : ℚ) : 
  (k - (-4)) / (5 - (-1)) = k → k = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_equals_k_l1785_178587


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l1785_178536

theorem candy_sampling_percentage (caught_sampling : Real) (total_sampling : Real) 
  (h1 : caught_sampling = 22)
  (h2 : total_sampling = 25.88235294117647) :
  total_sampling - caught_sampling = 3.88235294117647 := by
sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l1785_178536


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l1785_178541

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k h : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l1785_178541


namespace NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l1785_178555

theorem triangle_angle_b_is_pi_third 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : b^2 = a*c) 
  (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) 
  (h3 : A + B + C = π) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0) : 
  B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l1785_178555


namespace NUMINAMATH_CALUDE_amount_from_cars_and_buses_is_309_l1785_178585

/-- Calculates the amount raised from cars and buses given the total amount raised and the amounts from other vehicle types. -/
def amount_from_cars_and_buses (total_raised : ℕ) (suv_charge truck_charge motorcycle_charge : ℕ) (num_suvs num_trucks num_motorcycles : ℕ) : ℕ :=
  total_raised - (suv_charge * num_suvs + truck_charge * num_trucks + motorcycle_charge * num_motorcycles)

/-- Theorem stating that the amount raised from cars and buses is $309. -/
theorem amount_from_cars_and_buses_is_309 :
  amount_from_cars_and_buses 500 12 10 15 3 8 5 = 309 := by
  sorry

end NUMINAMATH_CALUDE_amount_from_cars_and_buses_is_309_l1785_178585


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1785_178580

theorem quadratic_inequality_solution_range (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (c < 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1785_178580


namespace NUMINAMATH_CALUDE_area_transformation_l1785_178516

-- Define a function representing the area under a curve
noncomputable def area_under_curve (f : ℝ → ℝ) : ℝ := sorry

-- Define the original function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem area_transformation (h : area_under_curve g = 15) :
  area_under_curve (fun x ↦ 4 * g (2 * x - 4)) = 30 := by sorry

end NUMINAMATH_CALUDE_area_transformation_l1785_178516


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1785_178527

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides. -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (n : ℕ) : 
  P = 180 → s = 15 → P = n * s → n = 12 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1785_178527


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_theorem_l1785_178572

theorem ceiling_negative_sqrt_theorem :
  ⌈-Real.sqrt ((64 : ℝ) / 9 - 1)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_theorem_l1785_178572


namespace NUMINAMATH_CALUDE_magnitude_z_squared_l1785_178518

-- Define the complex number z
def z : ℂ := 1 + Complex.I^5

-- Theorem statement
theorem magnitude_z_squared : Complex.abs (z^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_squared_l1785_178518


namespace NUMINAMATH_CALUDE_sibling_count_l1785_178532

theorem sibling_count (boys girls : ℕ) : 
  boys = 1 ∧ 
  boys - 1 = 0 ∧ 
  girls - 1 = boys → 
  boys + girls = 3 := by
sorry

end NUMINAMATH_CALUDE_sibling_count_l1785_178532


namespace NUMINAMATH_CALUDE_geometric_sequence_k_value_l1785_178534

/-- A geometric sequence with sum S_n = 3 * 2^n + k -/
structure GeometricSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  k : ℝ
  sum_formula : ∀ n : ℕ+, S n = 3 * 2^(n : ℝ) + k

/-- The value of k in the geometric sequence sum formula -/
theorem geometric_sequence_k_value (seq : GeometricSequence) : seq.k = -3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_k_value_l1785_178534


namespace NUMINAMATH_CALUDE_escalator_time_to_cover_l1785_178531

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover its length -/
theorem escalator_time_to_cover (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 15 →
  person_speed = 3 →
  escalator_length = 180 →
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_to_cover_l1785_178531


namespace NUMINAMATH_CALUDE_largest_difference_l1785_178558

def A : ℕ := 3 * 2010^2011
def B : ℕ := 2010^2011
def C : ℕ := 2009 * 2010^2010
def D : ℕ := 3 * 2010^2010
def E : ℕ := 2010^2010
def F : ℕ := 2010^2009

theorem largest_difference : 
  (A - B) > (B - C) ∧ 
  (A - B) > (C - D) ∧ 
  (A - B) > (D - E) ∧ 
  (A - B) > (E - F) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l1785_178558


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1785_178546

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : IsEven f) :
  IsEven (fun x ↦ f (f x)) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1785_178546


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1785_178510

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) : 
  original_price = 390 →
  final_price = 248.625 →
  ∃ (first_discount : ℝ),
    first_discount = 15 ∧
    final_price = original_price * (100 - first_discount) / 100 * 75 / 100 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1785_178510


namespace NUMINAMATH_CALUDE_certain_number_multiplied_l1785_178578

theorem certain_number_multiplied (x : ℝ) : x - 7 = 9 → 3 * x = 48 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplied_l1785_178578


namespace NUMINAMATH_CALUDE_measure_8_liters_possible_min_operations_is_30_l1785_178520

/-- Represents the state of the two vessels --/
structure VesselState :=
  (vessel15 : ℕ)
  (vessel16 : ℕ)

/-- Represents an operation on the vessels --/
inductive Operation
  | Fill15
  | Fill16
  | Empty15
  | Empty16
  | Pour15To16
  | Pour16To15

/-- Applies an operation to a vessel state --/
def applyOperation (state : VesselState) (op : Operation) : VesselState :=
  match op with
  | Operation.Fill15 => ⟨15, state.vessel16⟩
  | Operation.Fill16 => ⟨state.vessel15, 16⟩
  | Operation.Empty15 => ⟨0, state.vessel16⟩
  | Operation.Empty16 => ⟨state.vessel15, 0⟩
  | Operation.Pour15To16 => 
      let amount := min state.vessel15 (16 - state.vessel16)
      ⟨state.vessel15 - amount, state.vessel16 + amount⟩
  | Operation.Pour16To15 => 
      let amount := min state.vessel16 (15 - state.vessel15)
      ⟨state.vessel15 + amount, state.vessel16 - amount⟩

/-- Checks if a sequence of operations results in 8 liters in either vessel --/
def achieves8Liters (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.vessel15 = 8 ∨ finalState.vessel16 = 8

/-- The main theorem stating that it's possible to measure 8 liters --/
theorem measure_8_liters_possible : ∃ (ops : List Operation), achieves8Liters ops :=
  sorry

/-- The theorem stating that the minimum number of operations is 30 --/
theorem min_operations_is_30 : 
  (∃ (ops : List Operation), achieves8Liters ops ∧ ops.length = 30) ∧
  (∀ (ops : List Operation), achieves8Liters ops → ops.length ≥ 30) :=
  sorry

end NUMINAMATH_CALUDE_measure_8_liters_possible_min_operations_is_30_l1785_178520


namespace NUMINAMATH_CALUDE_steves_matching_socks_l1785_178524

theorem steves_matching_socks (total_socks : ℕ) (mismatching_socks : ℕ) 
  (h1 : total_socks = 25) 
  (h2 : mismatching_socks = 17) : 
  (total_socks - mismatching_socks) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_steves_matching_socks_l1785_178524


namespace NUMINAMATH_CALUDE_equation_solutions_l1785_178508

theorem equation_solutions : 
  ∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = 4 + Real.sqrt 57 ∨ x = 4 - Real.sqrt 57) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1785_178508


namespace NUMINAMATH_CALUDE_jill_draws_spade_prob_l1785_178512

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Represents the probability of drawing a spade from a standard deck -/
def ProbSpade : ℚ := NumSpades / StandardDeck

/-- Represents the probability of not drawing a spade from a standard deck -/
def ProbNotSpade : ℚ := 1 - ProbSpade

/-- Represents the probability that Jack draws a spade -/
def ProbJackSpade : ℚ := ProbSpade

/-- Represents the probability that Jill draws a spade -/
def ProbJillSpade : ℚ := ProbNotSpade * ProbSpade

/-- Represents the probability that John draws a spade -/
def ProbJohnSpade : ℚ := ProbNotSpade * ProbNotSpade * ProbSpade

/-- Represents the probability that a spade is drawn in one cycle -/
def ProbSpadeInCycle : ℚ := ProbJackSpade + ProbJillSpade + ProbJohnSpade

theorem jill_draws_spade_prob : 
  ProbJillSpade / ProbSpadeInCycle = 12 / 37 := by sorry

end NUMINAMATH_CALUDE_jill_draws_spade_prob_l1785_178512


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1785_178539

theorem tan_alpha_plus_pi_fourth (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1785_178539


namespace NUMINAMATH_CALUDE_rectangle_ratio_squared_l1785_178538

theorem rectangle_ratio_squared (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≤ b) : 
  (a / b + 1 / 2 = b / Real.sqrt (a^2 + b^2)) → (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_squared_l1785_178538


namespace NUMINAMATH_CALUDE_probability_of_selection_l1785_178568

def multiple_choice_count : ℕ := 12
def fill_in_blank_count : ℕ := 4
def open_ended_count : ℕ := 6
def total_questions : ℕ := multiple_choice_count + fill_in_blank_count + open_ended_count
def selection_count : ℕ := 3

theorem probability_of_selection (multiple_choice_count fill_in_blank_count open_ended_count total_questions selection_count : ℕ) 
  (h1 : multiple_choice_count = 12)
  (h2 : fill_in_blank_count = 4)
  (h3 : open_ended_count = 6)
  (h4 : total_questions = multiple_choice_count + fill_in_blank_count + open_ended_count)
  (h5 : selection_count = 3) :
  (Nat.choose multiple_choice_count 1 * Nat.choose open_ended_count 2 +
   Nat.choose multiple_choice_count 2 * Nat.choose open_ended_count 1 +
   Nat.choose multiple_choice_count 1 * Nat.choose open_ended_count 1 * Nat.choose fill_in_blank_count 1) /
  (Nat.choose total_questions selection_count - Nat.choose (fill_in_blank_count + open_ended_count) selection_count) = 43 / 71 := by
  sorry

#check probability_of_selection

end NUMINAMATH_CALUDE_probability_of_selection_l1785_178568


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l1785_178548

theorem opposite_of_negative_three : -(- 3) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l1785_178548


namespace NUMINAMATH_CALUDE_sequence_inequality_l1785_178505

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) → k > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1785_178505


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l1785_178523

theorem expression_equals_negative_one (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x ∧ z ≠ -x) :
  (x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1 :=
sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l1785_178523


namespace NUMINAMATH_CALUDE_salary_change_percentage_l1785_178511

theorem salary_change_percentage (original : ℝ) (h : original > 0) :
  let decreased := original * (1 - 0.5)
  let increased := decreased * (1 + 0.5)
  increased = original * 0.75 ∧ (original - increased) / original = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l1785_178511


namespace NUMINAMATH_CALUDE_complex_equality_l1785_178553

theorem complex_equality : (1 - Complex.I)^2 * Complex.I = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equality_l1785_178553


namespace NUMINAMATH_CALUDE_calculate_income_before_tax_l1785_178504

/-- Given tax rates and differential savings, calculate the annual income before tax -/
theorem calculate_income_before_tax 
  (original_rate : ℝ) 
  (new_rate : ℝ) 
  (differential_savings : ℝ) 
  (h1 : original_rate = 0.42)
  (h2 : new_rate = 0.32)
  (h3 : differential_savings = 4240) :
  ∃ (income : ℝ), income * (original_rate - new_rate) = differential_savings ∧ income = 42400 := by
  sorry

end NUMINAMATH_CALUDE_calculate_income_before_tax_l1785_178504


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l1785_178542

theorem missing_digit_divisible_by_six : ∃ (n : ℕ), 
  n ≥ 31610 ∧ n ≤ 31619 ∧ n % 10 = 4 ∧ n % 100 = 14 ∧ n % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l1785_178542


namespace NUMINAMATH_CALUDE_z_values_l1785_178583

theorem z_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  let z := (x - 3)^2 * (x + 4) / (3 * x - 4)
  z = 0 ∨ z = 192 := by sorry

end NUMINAMATH_CALUDE_z_values_l1785_178583


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1785_178525

theorem fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25/6 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1785_178525


namespace NUMINAMATH_CALUDE_jake_has_fewer_balloons_l1785_178515

/-- The number of balloons each person has in the park scenario -/
structure BalloonCounts where
  allan : ℕ
  jake_initial : ℕ
  jake_bought : ℕ
  emily : ℕ

/-- The difference in balloon count between Jake and the combined total of Allan and Emily -/
def balloon_difference (counts : BalloonCounts) : ℤ :=
  (counts.jake_initial + counts.jake_bought : ℤ) - (counts.allan + counts.emily)

/-- Theorem stating that Jake has 4 fewer balloons than Allan and Emily combined -/
theorem jake_has_fewer_balloons (counts : BalloonCounts)
  (h1 : counts.allan = 6)
  (h2 : counts.jake_initial = 3)
  (h3 : counts.jake_bought = 4)
  (h4 : counts.emily = 5) :
  balloon_difference counts = -4 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_fewer_balloons_l1785_178515


namespace NUMINAMATH_CALUDE_riding_to_total_ratio_l1785_178501

/-- Represents the number of horses and owners -/
def total_count : ℕ := 18

/-- Represents the number of legs walking on the ground -/
def legs_on_ground : ℕ := 90

/-- Represents the number of owners riding their horses -/
def riding_owners : ℕ := total_count - (legs_on_ground - 4 * total_count) / 2

/-- Theorem stating the ratio of riding owners to total owners -/
theorem riding_to_total_ratio :
  (riding_owners : ℚ) / total_count = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_riding_to_total_ratio_l1785_178501


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_125_l1785_178581

theorem greatest_prime_factor_of_125 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 125 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 125 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_125_l1785_178581


namespace NUMINAMATH_CALUDE_rectangle_diagonals_plus_three_l1785_178537

theorem rectangle_diagonals_plus_three (rectangle_diagonals : ℕ) : 
  rectangle_diagonals = 2 → rectangle_diagonals + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonals_plus_three_l1785_178537


namespace NUMINAMATH_CALUDE_train_crossing_time_l1785_178560

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole --/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 675)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1785_178560


namespace NUMINAMATH_CALUDE_sum_of_specific_series_l1785_178589

def geometric_series (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_specific_series :
  let a : ℚ := 1/2
  let r : ℚ := -1/4
  let n : ℕ := 6
  geometric_series a r n = 4095/10240 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_series_l1785_178589


namespace NUMINAMATH_CALUDE_asafa_arrives_5_min_after_florence_l1785_178547

/-- Represents a point in the route -/
inductive Point | P | Q | R | S

/-- Represents a runner -/
inductive Runner | Asafa | Florence

/-- The speed of a runner in km/h -/
def speed (r : Runner) : ℝ :=
  match r with
  | Runner.Asafa => 21
  | Runner.Florence => 16.8  -- This is derived, not given directly

/-- The distance between two points in km -/
def distance (p1 p2 : Point) : ℝ :=
  match p1, p2 with
  | Point.P, Point.Q => 8
  | Point.Q, Point.R => 15
  | Point.R, Point.S => 7
  | Point.P, Point.R => 17  -- This is derived, not given directly
  | _, _ => 0  -- For all other combinations

/-- The time difference in minutes between Florence and Asafa arriving at point R -/
def time_difference_at_R : ℝ := 5

/-- The theorem to be proved -/
theorem asafa_arrives_5_min_after_florence :
  let total_distance_asafa := distance Point.P Point.Q + distance Point.Q Point.R + distance Point.R Point.S
  let total_distance_florence := distance Point.P Point.R + distance Point.R Point.S
  let total_time := total_distance_asafa / speed Runner.Asafa
  let time_asafa_RS := distance Point.R Point.S / speed Runner.Asafa
  let time_florence_RS := distance Point.R Point.S / speed Runner.Florence
  time_florence_RS - time_asafa_RS = time_difference_at_R / 60 := by
  sorry

end NUMINAMATH_CALUDE_asafa_arrives_5_min_after_florence_l1785_178547


namespace NUMINAMATH_CALUDE_digit_product_sum_28_l1785_178599

/-- Represents a base-10 digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Converts two digits to a two-digit number -/
def toTwoDigitNumber (a b : Digit) : TwoDigitNumber :=
  ⟨a.val * 10 + b.val, by sorry⟩

/-- Converts a digit to a three-digit number where all digits are the same -/
def toThreeDigitSameNumber (e : Digit) : Nat :=
  e.val * 100 + e.val * 10 + e.val

theorem digit_product_sum_28 
  (A B C D E : Digit) 
  (h_unique : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_product : (toTwoDigitNumber A B).val * (toTwoDigitNumber C D).val = toThreeDigitSameNumber E) :
  A.val + B.val + C.val + D.val + E.val = 28 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_sum_28_l1785_178599


namespace NUMINAMATH_CALUDE_cross_section_area_of_cut_prism_l1785_178503

/-- The area of the cross-section when a plane cuts a right prism with equilateral triangular base -/
theorem cross_section_area_of_cut_prism (V : ℝ) (α : ℝ) :
  V > 0 → 0 < α → α < π / 2 →
  ∃ (S : ℝ), S = (3 * Real.sqrt 3 * V^2 / (Real.sin α)^2 / Real.cos α)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cross_section_area_of_cut_prism_l1785_178503


namespace NUMINAMATH_CALUDE_not_red_card_probability_l1785_178530

/-- Given a deck of cards where the odds of drawing a red card are 1:3,
    the probability of drawing a card that is not red is 3/4. -/
theorem not_red_card_probability (odds : ℚ) (h : odds = 1/3) :
  1 - odds / (1 + odds) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_not_red_card_probability_l1785_178530


namespace NUMINAMATH_CALUDE_solve_equation_l1785_178526

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.01) : x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1785_178526


namespace NUMINAMATH_CALUDE_inequality_solution_l1785_178576

theorem inequality_solution (x : ℝ) :
  (1 - (2*x - 2)/5 < (3 - 4*x)/2) ↔ (x < 1/16) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1785_178576


namespace NUMINAMATH_CALUDE_vacation_pictures_l1785_178544

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The number of pictures Megan still has from her vacation -/
def remaining_pictures : ℕ := zoo_pictures + museum_pictures - deleted_pictures

theorem vacation_pictures : remaining_pictures = 2 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l1785_178544


namespace NUMINAMATH_CALUDE_cosine_difference_l1785_178592

theorem cosine_difference (α β : ℝ) 
  (h1 : α + β = π / 3)
  (h2 : Real.tan α + Real.tan β = 2) :
  Real.cos (α - β) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_l1785_178592


namespace NUMINAMATH_CALUDE_unique_consecutive_set_sum_18_l1785_178545

/-- A set of consecutive positive integers -/
def ConsecutiveSet (a n : ℕ) : Set ℕ := {x | ∃ k, 0 ≤ k ∧ k < n ∧ x = a + k}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSetSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Main theorem: There is exactly one set of consecutive positive integers with sum 18 -/
theorem unique_consecutive_set_sum_18 :
  ∃! p : ℕ × ℕ, p.2 ≥ 2 ∧ ConsecutiveSetSum p.1 p.2 = 18 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_set_sum_18_l1785_178545


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l1785_178562

def satisfies_conditions (n : ℕ) : Prop :=
  ∀ d : ℕ, 2 ≤ d → d ≤ 10 → n % d = d - 1

theorem smallest_satisfying_number : 
  satisfies_conditions 2519 ∧ 
  ∀ m : ℕ, m < 2519 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l1785_178562


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1785_178598

theorem final_sum_after_operations (x y D : ℝ) (h : x - y = D) :
  4 * ((x - 5) + (y - 5)) = 4 * (x + y) - 40 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1785_178598


namespace NUMINAMATH_CALUDE_cubic_root_property_l1785_178582

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if 4 and -3 are roots of the equation, then (b+c)/a = -13 -/
theorem cubic_root_property (a b c d : ℝ) (ha : a ≠ 0) :
  (a * (4 : ℝ)^3 + b * (4 : ℝ)^2 + c * (4 : ℝ) + d = 0) →
  (a * (-3 : ℝ)^3 + b * (-3 : ℝ)^2 + c * (-3 : ℝ) + d = 0) →
  (b + c) / a = -13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_property_l1785_178582


namespace NUMINAMATH_CALUDE_number_percentage_equality_l1785_178595

theorem number_percentage_equality (x : ℝ) : 
  (25 / 100) * x = (20 / 100) * 30 → x = 24 := by
sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l1785_178595


namespace NUMINAMATH_CALUDE_max_leftover_candies_l1785_178514

theorem max_leftover_candies (n : ℕ) : ∃ (k : ℕ), n = 8 * k + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_leftover_candies_l1785_178514


namespace NUMINAMATH_CALUDE_train_length_calculation_l1785_178551

/-- Given a train and a platform with equal length, if the train crosses the platform
    in one minute at a speed of 36 km/hr, then the length of the train is 300 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) 
  (speed : ℝ) (time : ℝ) : 
  train_length = platform_length →
  speed = 36 →
  time = 1 / 60 →
  train_length = 300 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1785_178551


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l1785_178569

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length given specific conditions -/
theorem bridge_length_proof :
  bridge_length 150 45 30 = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l1785_178569


namespace NUMINAMATH_CALUDE_smallest_n_complex_equation_l1785_178566

theorem smallest_n_complex_equation (n : ℕ) (a b : ℝ) : 
  n > 3 ∧ 
  0 < a ∧ 
  0 < b ∧ 
  (∀ k : ℕ, 3 < k ∧ k < n → ¬∃ x y : ℝ, 0 < x ∧ 0 < y ∧ (x + y * I) ^ k + x = (x - y * I) ^ k + y) ∧
  (a + b * I) ^ n + a = (a - b * I) ^ n + b →
  b / a = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_n_complex_equation_l1785_178566


namespace NUMINAMATH_CALUDE_intersection_distance_and_difference_l1785_178519

def f (x : ℝ) := 5 * x^2 + 3 * x - 2

theorem intersection_distance_and_difference :
  ∃ (C D : ℝ × ℝ),
    (f C.1 = 4 ∧ C.2 = 4) ∧
    (f D.1 = 4 ∧ D.2 = 4) ∧
    C ≠ D ∧
    ∃ (p q : ℕ),
      p = 129 ∧
      q = 5 ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = (Real.sqrt p / q)^2 ∧
      p - q = 124 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_and_difference_l1785_178519


namespace NUMINAMATH_CALUDE_composite_square_area_l1785_178570

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square composed of rectangles -/
structure CompositeSquare where
  rectangle : Rectangle
  
/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The side length of the composite square -/
def CompositeSquare.sideLength (s : CompositeSquare) : ℝ := s.rectangle.length + s.rectangle.width

/-- The area of the composite square -/
def CompositeSquare.area (s : CompositeSquare) : ℝ := (s.sideLength) ^ 2

theorem composite_square_area (s : CompositeSquare) 
  (h : s.rectangle.perimeter = 40) : s.area = 400 := by
  sorry

end NUMINAMATH_CALUDE_composite_square_area_l1785_178570


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1785_178573

/-- Given a complex number z satisfying (z-i)i = 2+i, prove that |z| = √5 -/
theorem magnitude_of_z (z : ℂ) (h : (z - Complex.I) * Complex.I = 2 + Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1785_178573
