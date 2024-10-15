import Mathlib

namespace NUMINAMATH_CALUDE_pear_weighs_130_l2359_235910

/-- The weight of an apple in grams -/
def apple_weight : ℝ := sorry

/-- The weight of a pear in grams -/
def pear_weight : ℝ := sorry

/-- The weight of a banana in grams -/
def banana_weight : ℝ := sorry

/-- The first condition: one apple, three pears, and two bananas weigh 920 grams -/
axiom condition1 : apple_weight + 3 * pear_weight + 2 * banana_weight = 920

/-- The second condition: two apples, four bananas, and five pears weigh 1,710 grams -/
axiom condition2 : 2 * apple_weight + 4 * banana_weight + 5 * pear_weight = 1710

/-- Theorem stating that a pear weighs 130 grams -/
theorem pear_weighs_130 : pear_weight = 130 := by sorry

end NUMINAMATH_CALUDE_pear_weighs_130_l2359_235910


namespace NUMINAMATH_CALUDE_volunteer_distribution_l2359_235959

theorem volunteer_distribution (n : ℕ) (k : ℕ) (m : ℕ) : n = 5 ∧ k = 3 ∧ m = 3 →
  (Nat.choose n 2 * Nat.choose (n - 2) 2 * Nat.choose (n - 4) 1) / 2 * Nat.factorial k = 90 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l2359_235959


namespace NUMINAMATH_CALUDE_class_size_problem_l2359_235936

theorem class_size_problem (average_age : ℝ) (teacher_age : ℝ) (new_average : ℝ) :
  average_age = 10 →
  teacher_age = 26 →
  new_average = average_age + 1 →
  ∃ n : ℕ, (n : ℝ) * average_age + teacher_age = (n + 1 : ℝ) * new_average ∧ n = 15 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l2359_235936


namespace NUMINAMATH_CALUDE_cube_root_of_64_l2359_235937

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l2359_235937


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2359_235977

theorem min_value_expression (x y : ℝ) : 4 * x^2 + 4 * x * Real.sin y - Real.cos y^2 ≥ -1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, 4 * x^2 + 4 * x * Real.sin y - Real.cos y^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2359_235977


namespace NUMINAMATH_CALUDE_final_price_is_135_l2359_235924

/-- The original price of the dress -/
def original_price : ℝ := 250

/-- The first discount rate -/
def first_discount_rate : ℝ := 0.4

/-- The additional holiday discount rate -/
def holiday_discount_rate : ℝ := 0.1

/-- The price after the first discount -/
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)

/-- The final price after both discounts -/
def final_price : ℝ := price_after_first_discount * (1 - holiday_discount_rate)

/-- Theorem stating that the final price is $135 -/
theorem final_price_is_135 : final_price = 135 := by sorry

end NUMINAMATH_CALUDE_final_price_is_135_l2359_235924


namespace NUMINAMATH_CALUDE_seat_39_is_51_l2359_235987

/-- Calculates the seat number for the nth person in a circular seating arrangement --/
def seatNumber (n : ℕ) (totalSeats : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let binaryRep := (n - 1).digits 2
    let seatCalc := binaryRep.foldl (fun acc (b : ℕ) => (2 * acc + b) % totalSeats) 1
    if seatCalc = 0 then totalSeats else seatCalc

/-- The theorem stating that the 39th person sits on seat 51 in a 128-seat arrangement --/
theorem seat_39_is_51 : seatNumber 39 128 = 51 := by
  sorry

/-- Verifies the seating arrangement for the first few people --/
example : List.map (fun n => seatNumber n 128) [1, 2, 3, 4, 5] = [1, 65, 33, 97, 17] := by
  sorry

end NUMINAMATH_CALUDE_seat_39_is_51_l2359_235987


namespace NUMINAMATH_CALUDE_pascal_row8_sum_and_difference_l2359_235907

/-- Pascal's Triangle sum for a given row -/
def pascal_sum (n : ℕ) : ℕ := 2^n

theorem pascal_row8_sum_and_difference :
  (pascal_sum 8 = 256) ∧
  (pascal_sum 8 - pascal_sum 7 = 128) := by
  sorry

end NUMINAMATH_CALUDE_pascal_row8_sum_and_difference_l2359_235907


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l2359_235935

def digit_sum (n : ℕ) : ℕ := sorry

def is_smallest_with_properties (n : ℕ) : Prop :=
  (n % 5 = 0) ∧
  (digit_sum n = 100) ∧
  (∀ m : ℕ, m < n → (m % 5 ≠ 0 ∨ digit_sum m ≠ 100))

theorem smallest_number_with_properties :
  is_smallest_with_properties 599999999995 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l2359_235935


namespace NUMINAMATH_CALUDE_inequality_proof_l2359_235990

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1/16 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2359_235990


namespace NUMINAMATH_CALUDE_problem_statement_l2359_235903

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x * y > a * b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → x^2 + y^2 ≥ 1/2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → 4/x + 1/y ≥ 9) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2359_235903


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2359_235931

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem negation_of_universal_proposition :
  (¬ ∀ n ∈ M, n > 1) ↔ (∃ n ∈ M, n ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2359_235931


namespace NUMINAMATH_CALUDE_tire_comparison_l2359_235954

def type_A : List ℕ := [94, 96, 99, 99, 105, 107]
def type_B : List ℕ := [95, 95, 98, 99, 104, 109]

def mode (l : List ℕ) : ℕ := sorry
def range (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℚ := sorry
def variance (l : List ℕ) : ℚ := sorry

theorem tire_comparison :
  (mode type_A > mode type_B) ∧
  (range type_A < range type_B) ∧
  (mean type_A = mean type_B) ∧
  (variance type_A < variance type_B) := by sorry

end NUMINAMATH_CALUDE_tire_comparison_l2359_235954


namespace NUMINAMATH_CALUDE_max_a_when_f_has_minimum_l2359_235983

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

/-- Proposition: If f(x) has a minimum value, then the maximum value of a is 1 -/
theorem max_a_when_f_has_minimum (a : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m) → a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_a_when_f_has_minimum_l2359_235983


namespace NUMINAMATH_CALUDE_rachels_homework_l2359_235922

/-- 
Given that Rachel has 5 pages of math homework and 3 more pages of math homework 
than reading homework, prove that she has 2 pages of reading homework.
-/
theorem rachels_homework (math_pages reading_pages : ℕ) : 
  math_pages = 5 → 
  math_pages = reading_pages + 3 → 
  reading_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l2359_235922


namespace NUMINAMATH_CALUDE_max_value_of_f_l2359_235985

def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2359_235985


namespace NUMINAMATH_CALUDE_square_corners_l2359_235970

theorem square_corners (S : ℤ) : ∃ (A B C D : ℤ),
  A + B + 9 = S ∧
  B + C + 6 = S ∧
  D + C + 12 = S ∧
  D + A + 15 = S ∧
  A + C + 17 = S ∧
  A + B + C + D = 123 ∧
  A = 26 ∧ B = 37 ∧ C = 29 ∧ D = 31 := by
  sorry

end NUMINAMATH_CALUDE_square_corners_l2359_235970


namespace NUMINAMATH_CALUDE_exist_similar_triangles_same_color_l2359_235919

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define similarity between triangles
def areSimilar (t1 t2 : Triangle) (ratio : ℝ) : Prop := sorry

-- Define the main theorem
theorem exist_similar_triangles_same_color :
  ∃ (t1 t2 : Triangle) (color : Color),
    areSimilar t1 t2 1995 ∧
    colorFunction t1.a = color ∧
    colorFunction t1.b = color ∧
    colorFunction t1.c = color ∧
    colorFunction t2.a = color ∧
    colorFunction t2.b = color ∧
    colorFunction t2.c = color := by
  sorry

end NUMINAMATH_CALUDE_exist_similar_triangles_same_color_l2359_235919


namespace NUMINAMATH_CALUDE_jill_marathon_time_l2359_235963

/-- The length of a marathon in kilometers -/
def marathon_length : ℝ := 40

/-- Jack's marathon time in hours -/
def jack_time : ℝ := 4.5

/-- The ratio of Jack's speed to Jill's speed -/
def speed_ratio : ℝ := 0.888888888888889

/-- Jill's marathon time in hours -/
def jill_time : ℝ := 4

theorem jill_marathon_time :
  marathon_length / (marathon_length / jack_time * (1 / speed_ratio)) = jill_time := by
  sorry

end NUMINAMATH_CALUDE_jill_marathon_time_l2359_235963


namespace NUMINAMATH_CALUDE_inequality_proof_l2359_235949

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2359_235949


namespace NUMINAMATH_CALUDE_intersection_M_N_l2359_235908

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2359_235908


namespace NUMINAMATH_CALUDE_binary_conversion_l2359_235997

-- Define the binary number
def binary_num : List Bool := [true, true, false, false, true, true]

-- Function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

-- Function to convert decimal to base 5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 51 ∧
  decimal_to_base5 (binary_to_decimal binary_num) = [2, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_conversion_l2359_235997


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l2359_235944

/-- Given two rectangles with equal areas, where one rectangle has dimensions 12 inches by W inches,
    and the other has dimensions 9 inches by 20 inches, prove that W equals 15 inches. -/
theorem equal_area_rectangles (W : ℝ) :
  (12 * W = 9 * 20) → W = 15 := by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l2359_235944


namespace NUMINAMATH_CALUDE_sally_has_88_cards_l2359_235971

/-- The number of Pokemon cards Sally has after receiving a gift and making a purchase -/
def sallys_cards (initial : ℕ) (gift : ℕ) (purchase : ℕ) : ℕ :=
  initial + gift + purchase

/-- Theorem: Sally has 88 Pokemon cards after starting with 27, receiving 41 as a gift, and buying 20 -/
theorem sally_has_88_cards : sallys_cards 27 41 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_88_cards_l2359_235971


namespace NUMINAMATH_CALUDE_stream_speed_l2359_235973

/-- Proves that given a man's downstream speed of 18 km/h, upstream speed of 6 km/h,
    and still water speed of 12 km/h, the speed of the stream is 6 km/h. -/
theorem stream_speed (v_downstream v_upstream v_stillwater : ℝ)
    (h_downstream : v_downstream = 18)
    (h_upstream : v_upstream = 6)
    (h_stillwater : v_stillwater = 12)
    (h_downstream_eq : v_downstream = v_stillwater + (v_downstream - v_upstream) / 2)
    (h_upstream_eq : v_upstream = v_stillwater - (v_downstream - v_upstream) / 2) :
    (v_downstream - v_upstream) / 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l2359_235973


namespace NUMINAMATH_CALUDE_xy_expression_value_l2359_235969

theorem xy_expression_value (x y m : ℝ) 
  (eq1 : x + y + m = 6) 
  (eq2 : 3 * x - y + m = 4) : 
  -2 * x * y + 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_xy_expression_value_l2359_235969


namespace NUMINAMATH_CALUDE_job_completion_time_l2359_235905

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  (1 / x + 1 / 30 = 1 / 10) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l2359_235905


namespace NUMINAMATH_CALUDE_sales_job_base_salary_l2359_235980

/-- The base salary of a sales job, given the following conditions:
  - The original salary was $75,000 per year
  - The new job pays a base salary plus 15% commission
  - Each sale is worth $750
  - 266.67 sales per year are needed to not lose money
-/
theorem sales_job_base_salary :
  ∀ (original_salary : ℝ) (commission_rate : ℝ) (sale_value : ℝ) (sales_needed : ℝ),
    original_salary = 75000 →
    commission_rate = 0.15 →
    sale_value = 750 →
    sales_needed = 266.67 →
    ∃ (base_salary : ℝ),
      base_salary + sales_needed * commission_rate * sale_value = original_salary ∧
      base_salary = 45000 :=
by sorry

end NUMINAMATH_CALUDE_sales_job_base_salary_l2359_235980


namespace NUMINAMATH_CALUDE_genevieve_cherry_shortage_l2359_235994

/-- The amount Genevieve was short when buying cherries -/
def amount_short (cost_per_kg : ℕ) (amount_had : ℕ) (kg_bought : ℕ) : ℕ :=
  cost_per_kg * kg_bought - amount_had

/-- Proof that Genevieve was short $400 -/
theorem genevieve_cherry_shortage : amount_short 8 1600 250 = 400 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_cherry_shortage_l2359_235994


namespace NUMINAMATH_CALUDE_polygon_area_bound_l2359_235909

/-- A polygon in 2D space --/
structure Polygon where
  -- We don't need to define the exact structure of the polygon,
  -- just its projections and area
  proj_ox : ℝ
  proj_bisector13 : ℝ
  proj_oy : ℝ
  proj_bisector24 : ℝ
  area : ℝ

/-- Theorem stating that the area of a polygon with given projections is bounded --/
theorem polygon_area_bound (p : Polygon)
  (h1 : p.proj_ox = 4)
  (h2 : p.proj_bisector13 = 3 * Real.sqrt 2)
  (h3 : p.proj_oy = 5)
  (h4 : p.proj_bisector24 = 4 * Real.sqrt 2)
  : p.area ≤ 17.5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_bound_l2359_235909


namespace NUMINAMATH_CALUDE_farm_animals_l2359_235961

theorem farm_animals (chickens buffalos : ℕ) : 
  chickens + buffalos = 9 →
  2 * chickens + 4 * buffalos = 26 →
  chickens = 5 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l2359_235961


namespace NUMINAMATH_CALUDE_total_selling_price_l2359_235964

/-- Calculate the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem total_selling_price
  (quantity : ℕ)
  (profit_per_meter : ℚ)
  (cost_price_per_meter : ℚ)
  (h1 : quantity = 92)
  (h2 : profit_per_meter = 24)
  (h3 : cost_price_per_meter = 83.5)
  : (quantity : ℚ) * (cost_price_per_meter + profit_per_meter) = 9890 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_l2359_235964


namespace NUMINAMATH_CALUDE_parabola_focus_l2359_235966

/-- The focus of a parabola with equation y^2 = -6x has coordinates (-3/2, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -6*x → (x + 3/2)^2 + y^2 = (3/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l2359_235966


namespace NUMINAMATH_CALUDE_equation_solutions_l2359_235933

theorem equation_solutions :
  ∀ (x n : ℕ+) (p : ℕ), 
    Prime p → 
    (x^3 + 3*x + 14 = 2*p^(n : ℕ)) → 
    ((x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2359_235933


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l2359_235975

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) (c : ℝ) : Set ℝ := {x | P x = c}

/-- Theorem: If 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (c : ℝ) (h : 8 ∈ A_P P c) :
  ∃ (x : ℝ), x ∈ A_P P c ∧ ∃ (n : ℤ), x = 2 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l2359_235975


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2359_235991

theorem intersection_of_lines (x y : ℚ) : 
  x = 155 / 67 ∧ y = 5 / 67 ↔ 
  11 * x - 5 * y = 40 ∧ 9 * x + 2 * y = 15 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2359_235991


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2359_235939

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), n > 0 ∧ n = 6 ∧
  (∃ (p : ℕ), p < n ∧ Prime p ∧ Odd p ∧ (n^2 - n + 4) % p = 0) ∧
  (∃ (q : ℕ), q < n ∧ Prime q ∧ (n^2 - n + 4) % q ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    ¬((∃ (p : ℕ), p < m ∧ Prime p ∧ Odd p ∧ (m^2 - m + 4) % p = 0) ∧
      (∃ (q : ℕ), q < m ∧ Prime q ∧ (m^2 - m + 4) % q ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2359_235939


namespace NUMINAMATH_CALUDE_coconut_trees_per_square_meter_l2359_235934

/-- Represents the coconut farm scenario -/
structure CoconutFarm where
  size : ℝ
  treesPerSquareMeter : ℝ
  coconutsPerTree : ℕ
  harvestFrequency : ℕ
  pricePerCoconut : ℝ
  earningsAfterSixMonths : ℝ

/-- Theorem stating the number of coconut trees per square meter -/
theorem coconut_trees_per_square_meter (farm : CoconutFarm)
  (h1 : farm.size = 20)
  (h2 : farm.coconutsPerTree = 6)
  (h3 : farm.harvestFrequency = 3)
  (h4 : farm.pricePerCoconut = 0.5)
  (h5 : farm.earningsAfterSixMonths = 240) :
  farm.treesPerSquareMeter = 2 := by
  sorry


end NUMINAMATH_CALUDE_coconut_trees_per_square_meter_l2359_235934


namespace NUMINAMATH_CALUDE_eggs_given_by_marie_l2359_235974

/-- Given that Joyce initially had 8 eggs and ended up with 14 eggs in total,
    prove that Marie gave Joyce 6 eggs. -/
theorem eggs_given_by_marie 
  (initial_eggs : ℕ) 
  (total_eggs : ℕ) 
  (h1 : initial_eggs = 8) 
  (h2 : total_eggs = 14) : 
  total_eggs - initial_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_given_by_marie_l2359_235974


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2359_235982

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3) ^ 2 - 6 * (a 3) - 1 = 0 →
  (a 15) ^ 2 - 6 * (a 15) - 1 = 0 →
  (a 7) + (a 8) + (a 9) + (a 10) + (a 11) = 15 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2359_235982


namespace NUMINAMATH_CALUDE_blue_cube_problem_l2359_235930

theorem blue_cube_problem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_cube_problem_l2359_235930


namespace NUMINAMATH_CALUDE_problem_statement_l2359_235915

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define symmetry about a point
def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem problem_statement :
  (∀ a : ℝ, a > 0 → a ≠ 1 → log_base a (a * (-1) + 2 * a) = 1) ∧
  (∃ f : ℝ → ℝ, symmetric_about_origin (fun x ↦ f (x - 3)) ∧
    ¬ symmetric_about f (3, 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2359_235915


namespace NUMINAMATH_CALUDE_calculation_problem_l2359_235955

theorem calculation_problem (n : ℝ) : n = -6.4 ↔ 10 * 1.8 - (n * 1.5 / 0.3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_calculation_problem_l2359_235955


namespace NUMINAMATH_CALUDE_coin_division_problem_l2359_235947

theorem coin_division_problem (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(m % 7 = 3 ∧ m % 4 = 2)) →
  n % 7 = 3 →
  n % 4 = 2 →
  n % 8 = 2 :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l2359_235947


namespace NUMINAMATH_CALUDE_middle_letter_value_is_eight_l2359_235929

/-- Represents a three-letter word in Scrabble -/
structure ScrabbleWord where
  first_letter_value : ℕ
  middle_letter_value : ℕ
  last_letter_value : ℕ

/-- Calculates the total value of a ScrabbleWord before tripling -/
def word_value (word : ScrabbleWord) : ℕ :=
  word.first_letter_value + word.middle_letter_value + word.last_letter_value

/-- Theorem: Given the conditions, the middle letter's value is 8 -/
theorem middle_letter_value_is_eight 
  (word : ScrabbleWord)
  (h1 : word.first_letter_value = 1)
  (h2 : word.last_letter_value = 1)
  (h3 : 3 * (word_value word) = 30) :
  word.middle_letter_value = 8 := by
  sorry


end NUMINAMATH_CALUDE_middle_letter_value_is_eight_l2359_235929


namespace NUMINAMATH_CALUDE_circle_area_l2359_235998

theorem circle_area (c : ℝ) (h : c = 18 * Real.pi) :
  ∃ r : ℝ, c = 2 * Real.pi * r ∧ Real.pi * r^2 = 81 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l2359_235998


namespace NUMINAMATH_CALUDE_eggs_per_set_l2359_235960

theorem eggs_per_set (total_eggs : ℕ) (num_sets : ℕ) (h1 : total_eggs = 108) (h2 : num_sets = 9) :
  total_eggs / num_sets = 12 := by
sorry

end NUMINAMATH_CALUDE_eggs_per_set_l2359_235960


namespace NUMINAMATH_CALUDE_hohyeon_taller_than_seulgi_l2359_235928

/-- Seulgi's height in centimeters -/
def seulgi_height : ℕ := 159

/-- Hohyeon's height in centimeters -/
def hohyeon_height : ℕ := 162

/-- Theorem stating that Hohyeon is taller than Seulgi -/
theorem hohyeon_taller_than_seulgi : hohyeon_height > seulgi_height := by
  sorry

end NUMINAMATH_CALUDE_hohyeon_taller_than_seulgi_l2359_235928


namespace NUMINAMATH_CALUDE_independence_test_conclusions_not_always_correct_l2359_235901

-- Define the concept of independence tests
def IndependenceTest : Type := Unit

-- Define the properties of independence tests
axiom small_probability_principle : IndependenceTest → Prop
axiom conclusions_vary_with_samples : IndependenceTest → Prop
axiom not_only_method : IndependenceTest → Prop

-- Define the statement we want to prove false
def conclusions_always_correct (test : IndependenceTest) : Prop :=
  ∀ (sample : Type), true

-- Theorem statement
theorem independence_test_conclusions_not_always_correct :
  ∃ (test : IndependenceTest),
    small_probability_principle test ∧
    conclusions_vary_with_samples test ∧
    not_only_method test ∧
    ¬(conclusions_always_correct test) :=
by
  sorry

end NUMINAMATH_CALUDE_independence_test_conclusions_not_always_correct_l2359_235901


namespace NUMINAMATH_CALUDE_tenth_black_ball_probability_l2359_235965

/-- Represents the probability of drawing a black ball on the tenth draw from a box of colored balls. -/
def probability_tenth_black_ball (total_balls : ℕ) (black_balls : ℕ) : ℚ :=
  black_balls / total_balls

/-- Theorem stating that the probability of drawing a black ball on the tenth draw
    from a box with specific numbers of colored balls is 4/30. -/
theorem tenth_black_ball_probability :
  let red_balls : ℕ := 7
  let black_balls : ℕ := 4
  let yellow_balls : ℕ := 5
  let green_balls : ℕ := 6
  let white_balls : ℕ := 8
  let total_balls : ℕ := red_balls + black_balls + yellow_balls + green_balls + white_balls
  probability_tenth_black_ball total_balls black_balls = 4 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_black_ball_probability_l2359_235965


namespace NUMINAMATH_CALUDE_jill_study_time_difference_l2359_235999

/-- Represents the study time in minutes for each day -/
def StudyTime := Fin 3 → ℕ

theorem jill_study_time_difference (study : StudyTime) : 
  (study 0 = 120) →  -- First day study time in minutes
  (study 1 = 2 * study 0) →  -- Second day is double the first day
  (study 0 + study 1 + study 2 = 540) →  -- Total study time over 3 days
  (study 1 - study 2 = 60) :=  -- Difference between second and third day
by
  sorry

end NUMINAMATH_CALUDE_jill_study_time_difference_l2359_235999


namespace NUMINAMATH_CALUDE_compound_propositions_l2359_235938

-- Define the propositions p and q
def p : Prop := ∃ x : ℝ, x > 2 ∧ x > 1
def q : Prop := ∀ a b : ℝ, a > b → (1 / a) < (1 / b)

-- Define that p is sufficient but not necessary for x > 1
axiom p_sufficient : p → ∃ x : ℝ, x > 1
axiom p_not_necessary : ∃ x : ℝ, x > 1 ∧ ¬(x > 2)

-- Theorem stating that p ∧ ¬q is true, while other compounds are false
theorem compound_propositions :
  (p ∧ ¬q) ∧ ¬(p ∧ q) ∧ ¬(¬p ∨ q) ∧ ¬(¬p ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_compound_propositions_l2359_235938


namespace NUMINAMATH_CALUDE_gate_ticket_price_l2359_235950

/-- The price of plane tickets bought at the gate -/
def gate_price : ℝ := 200

/-- The number of people who pre-bought tickets -/
def pre_bought_count : ℕ := 20

/-- The price of pre-bought tickets -/
def pre_bought_price : ℝ := 155

/-- The number of people who bought tickets at the gate -/
def gate_count : ℕ := 30

/-- The additional amount paid in total by those who bought at the gate -/
def additional_gate_cost : ℝ := 2900

theorem gate_ticket_price :
  gate_price * gate_count = pre_bought_price * pre_bought_count + additional_gate_cost :=
by sorry

end NUMINAMATH_CALUDE_gate_ticket_price_l2359_235950


namespace NUMINAMATH_CALUDE_xy_difference_squared_l2359_235941

theorem xy_difference_squared (x y b c : ℝ) 
  (h1 : x * y = c^2) 
  (h2 : 1 / x^2 + 1 / y^2 = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := by
sorry

end NUMINAMATH_CALUDE_xy_difference_squared_l2359_235941


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2359_235904

-- Define propositions p and q
def p (x y : ℝ) : Prop := x > 0 ∧ y > 0
def q (x y : ℝ) : Prop := x * y > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q : 
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2359_235904


namespace NUMINAMATH_CALUDE_triangle_exists_l2359_235940

/-- A triangle with vertices in ℝ² --/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- The area of a triangle --/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The altitudes of a triangle --/
def Triangle.altitudes (t : Triangle) : List ℝ := sorry

/-- Theorem: There exists a triangle with all altitudes less than 1 and area greater than or equal to 10 --/
theorem triangle_exists : ∃ t : Triangle, (∀ h ∈ t.altitudes, h < 1) ∧ t.area ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exists_l2359_235940


namespace NUMINAMATH_CALUDE_center_square_side_length_l2359_235968

theorem center_square_side_length 
  (large_square_side : ℝ) 
  (l_shape_area_fraction : ℝ) 
  (num_l_shapes : ℕ) : 
  large_square_side = 120 →
  l_shape_area_fraction = 1/5 →
  num_l_shapes = 4 →
  ∃ (center_square_side : ℝ),
    center_square_side = 60 ∧
    center_square_side^2 = large_square_side^2 - num_l_shapes * l_shape_area_fraction * large_square_side^2 :=
by sorry

end NUMINAMATH_CALUDE_center_square_side_length_l2359_235968


namespace NUMINAMATH_CALUDE_specific_triangle_area_l2359_235951

/-- Represents a triangle with given properties -/
structure Triangle where
  base : ℝ
  side : ℝ
  median : ℝ

/-- Calculates the area of a triangle given its properties -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that a triangle with base 30, side 14, and median 13 has an area of 168 -/
theorem specific_triangle_area :
  let t : Triangle := { base := 30, side := 14, median := 13 }
  triangleArea t = 168 := by
  sorry

end NUMINAMATH_CALUDE_specific_triangle_area_l2359_235951


namespace NUMINAMATH_CALUDE_line_angle_slope_relation_l2359_235906

/-- Given two lines L₁ and L₂ in the xy-plane, prove that mn = 1/3 under specific conditions. -/
theorem line_angle_slope_relation (m n : ℝ) : 
  -- L₁ has equation y = 3mx
  -- L₂ has equation y = nx
  -- L₁ makes three times as large of an angle with the horizontal as L₂
  -- L₁ has 3 times the slope of L₂
  (∃ (θ₁ θ₂ : ℝ), θ₁ = 3 * θ₂ ∧ Real.tan θ₁ = 3 * m ∧ Real.tan θ₂ = n) →
  -- L₁ has 3 times the slope of L₂
  3 * m = n →
  -- L₁ is not vertical
  m ≠ 0 →
  -- Conclusion: mn = 1/3
  m * n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_slope_relation_l2359_235906


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2359_235926

/-- Represents a four-digit number as a tuple of four natural numbers -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Checks if a given FourDigitNumber satisfies all the required conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  a + b = c + d ∧
  b + d = 2 * (a + c) ∧
  a + d = c ∧
  b + c - a = 3 * d

/-- Theorem stating that there exists a unique four-digit number satisfying all conditions -/
theorem unique_four_digit_number : ∃! n : FourDigitNumber, satisfiesConditions n := by
  sorry


end NUMINAMATH_CALUDE_unique_four_digit_number_l2359_235926


namespace NUMINAMATH_CALUDE_certain_number_problem_l2359_235979

theorem certain_number_problem (x : ℤ) : 
  ((7 * (x + 5)) / 5 : ℚ) - 5 = 33 ↔ x = 22 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2359_235979


namespace NUMINAMATH_CALUDE_popsicle_sticks_count_l2359_235988

theorem popsicle_sticks_count 
  (num_groups : ℕ) 
  (sticks_per_group : ℕ) 
  (sticks_left : ℕ) 
  (h1 : num_groups = 10)
  (h2 : sticks_per_group = 15)
  (h3 : sticks_left = 20) :
  num_groups * sticks_per_group + sticks_left = 170 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_count_l2359_235988


namespace NUMINAMATH_CALUDE_f_value_at_pi_sixth_f_monotone_increasing_intervals_l2359_235945

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

theorem f_value_at_pi_sixth : f (π / 6) = 0 := by sorry

theorem f_monotone_increasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(π / 6) + k * π) ((π / 3) + k * π)) := by sorry

end NUMINAMATH_CALUDE_f_value_at_pi_sixth_f_monotone_increasing_intervals_l2359_235945


namespace NUMINAMATH_CALUDE_eds_pets_l2359_235911

theorem eds_pets (dogs cats : ℕ) (h1 : dogs = 2) (h2 : cats = 3) : 
  let fish := 2 * (dogs + cats)
  dogs + cats + fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_eds_pets_l2359_235911


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2359_235993

/-- A geometric sequence {a_n} with a_1 = 1 and a_5 = 9 has a_3 = 3 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 5 / a 1)^(1/4)) →  -- Geometric sequence condition
  a 1 = 1 →
  a 5 = 9 →
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2359_235993


namespace NUMINAMATH_CALUDE_sequence_properties_l2359_235978

theorem sequence_properties (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2*n) :
  (a 2 = 5 ∧ a 3 = 11 ∧ a 4 = 19) ∧
  (∀ n : ℕ, a n = n^2 + n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2359_235978


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l2359_235913

theorem lattice_points_on_hyperbola : 
  ∃! (s : Finset (ℤ × ℤ)), s.card = 4 ∧ ∀ (x y : ℤ), (x, y) ∈ s ↔ x^2 - y^2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l2359_235913


namespace NUMINAMATH_CALUDE_linear_decreasing_slope_l2359_235992

/-- A function that represents a linear equation with slope (m-3) and y-intercept 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x + 4

/-- The property that the function decreases as x increases -/
def decreasing (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → f m x₁ > f m x₂

theorem linear_decreasing_slope (m : ℝ) : decreasing m → m < 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_slope_l2359_235992


namespace NUMINAMATH_CALUDE_exists_excursion_with_frequent_participants_l2359_235996

/-- Represents an excursion --/
structure Excursion where
  participants : Finset Nat
  deriving Inhabited

/-- The problem statement --/
theorem exists_excursion_with_frequent_participants
  (n : Nat) -- number of excursions
  (excursions : Finset Excursion)
  (h1 : excursions.card = n) -- there are n excursions
  (h2 : ∀ e ∈ excursions, e.participants.card ≥ 4) -- each excursion has at least 4 participants
  (h3 : ∀ e ∈ excursions, e.participants.card ≤ 20) -- each excursion has at most 20 participants
  : ∃ e ∈ excursions, ∀ s ∈ e.participants,
    (excursions.filter (λ ex : Excursion => s ∈ ex.participants)).card ≥ n / 17 :=
sorry

end NUMINAMATH_CALUDE_exists_excursion_with_frequent_participants_l2359_235996


namespace NUMINAMATH_CALUDE_qiqi_mistake_xiaoming_jiajia_relation_l2359_235972

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := -x^2 + 4*x
def B (x : ℝ) : ℝ := 2*x^2 + 5*x - 4

-- Define Jiajia's correct answer
def correct_answer : ℝ := -18

-- Define Qiqi's mistaken coefficient
def qiqi_coefficient : ℝ := 3

-- Define the value of x
def x_value : ℝ := -2

-- Theorem 1: Qiqi's mistaken coefficient
theorem qiqi_mistake :
  A x_value + (2*x_value^2 + qiqi_coefficient*x_value - 4) = correct_answer + 16 :=
sorry

-- Theorem 2: Relationship between Xiaoming's and Jiajia's results
theorem xiaoming_jiajia_relation :
  A (-x_value) + B (-x_value) = -(A x_value + B x_value) :=
sorry

end NUMINAMATH_CALUDE_qiqi_mistake_xiaoming_jiajia_relation_l2359_235972


namespace NUMINAMATH_CALUDE_power_sum_equality_l2359_235918

theorem power_sum_equality : 2^345 + 9^8 / 9^5 = 2^345 + 729 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2359_235918


namespace NUMINAMATH_CALUDE_range_of_a_l2359_235942

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2359_235942


namespace NUMINAMATH_CALUDE_prime_factors_and_recalculation_l2359_235958

def original_number : ℕ := 546

theorem prime_factors_and_recalculation (n : ℕ) (h : n = original_number) :
  (∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ n ∧ largest ∣ n ∧
    (∀ p : ℕ, p.Prime → p ∣ n → smallest ≤ p) ∧
    (∀ p : ℕ, p.Prime → p ∣ n → p ≤ largest) ∧
    smallest + largest = 15) ∧
  (∃ (factors : List ℕ),
    (∀ p ∈ factors, p.Prime ∧ p ∣ n) ∧
    (∀ p : ℕ, p.Prime → p ∣ n → p ∈ factors) ∧
    (List.prod (List.map (· * 2) factors) = 8736)) :=
by sorry

end NUMINAMATH_CALUDE_prime_factors_and_recalculation_l2359_235958


namespace NUMINAMATH_CALUDE_power_evaluation_l2359_235962

theorem power_evaluation (a b : ℕ) (h : 360 = 2^a * 3^2 * 5^b) 
  (h2 : ∀ k > a, ¬ 2^k ∣ 360) (h5 : ∀ k > b, ¬ 5^k ∣ 360) : 
  (2/3 : ℚ)^(b-a) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_power_evaluation_l2359_235962


namespace NUMINAMATH_CALUDE_problem_statement_l2359_235902

-- Define proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, (f x₁ - f x₂) * (x₁ - x₂) ≥ 0

-- Define proposition q
def q : Prop :=
  ∀ x y : ℝ, x + y > 2 → x > 1 ∨ y > 1

-- Define decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem problem_statement (f : ℝ → ℝ) :
  (¬(p f ∧ q)) ∧ q → is_decreasing f :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2359_235902


namespace NUMINAMATH_CALUDE_perpendicular_diameter_bisects_chord_equal_central_angles_equal_arcs_equal_chords_equal_arcs_equal_arcs_equal_central_angles_l2359_235957

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a chord
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

-- Define an arc
structure Arc (c : Circle) where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

-- Define a central angle
def CentralAngle (c : Circle) (a : Arc c) : ℝ := sorry

-- Define the length of a chord
def chordLength (c : Circle) (ch : Chord c) : ℝ := sorry

-- Define the length of an arc
def arcLength (c : Circle) (a : Arc c) : ℝ := sorry

-- Define a diameter
def Diameter (c : Circle) := Chord c

-- Define perpendicularity between a diameter and a chord
def isPerpendicular (d : Diameter c) (ch : Chord c) : Prop := sorry

-- Define bisection of a chord
def bisectsChord (d : Diameter c) (ch : Chord c) : Prop := sorry

-- Theorem 1: A diameter perpendicular to a chord bisects the chord
theorem perpendicular_diameter_bisects_chord (c : Circle) (d : Diameter c) (ch : Chord c) :
  isPerpendicular d ch → bisectsChord d ch := sorry

-- Theorem 2: Equal central angles correspond to equal arcs
theorem equal_central_angles_equal_arcs (c : Circle) (a1 a2 : Arc c) :
  CentralAngle c a1 = CentralAngle c a2 → arcLength c a1 = arcLength c a2 := sorry

-- Theorem 3: Equal chords correspond to equal arcs
theorem equal_chords_equal_arcs (c : Circle) (ch1 ch2 : Chord c) (a1 a2 : Arc c) :
  chordLength c ch1 = chordLength c ch2 → arcLength c a1 = arcLength c a2 := sorry

-- Theorem 4: Equal arcs correspond to equal central angles
theorem equal_arcs_equal_central_angles (c : Circle) (a1 a2 : Arc c) :
  arcLength c a1 = arcLength c a2 → CentralAngle c a1 = CentralAngle c a2 := sorry

end NUMINAMATH_CALUDE_perpendicular_diameter_bisects_chord_equal_central_angles_equal_arcs_equal_chords_equal_arcs_equal_arcs_equal_central_angles_l2359_235957


namespace NUMINAMATH_CALUDE_positive_y_floor_product_l2359_235900

theorem positive_y_floor_product (y : ℝ) : 
  y > 0 → y * ⌊y⌋ = 90 → y = 10 := by sorry

end NUMINAMATH_CALUDE_positive_y_floor_product_l2359_235900


namespace NUMINAMATH_CALUDE_second_number_proof_l2359_235984

theorem second_number_proof (x : ℕ) : 
  (∃ k m : ℕ, 1657 = 127 * k + 6 ∧ x = 127 * m + 5 ∧ 
   ∀ d : ℕ, d > 127 → (1657 % d ≠ 6 ∨ x % d ≠ 5)) → 
  x = 1529 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2359_235984


namespace NUMINAMATH_CALUDE_solution_of_system_l2359_235946

def system_of_equations (x y z : ℝ) : Prop :=
  1 / x = y + z ∧ 1 / y = z + x ∧ 1 / z = x + y

theorem solution_of_system :
  ∃ (x y z : ℝ), system_of_equations x y z ∧
    ((x = Real.sqrt 2 / 2 ∧ y = Real.sqrt 2 / 2 ∧ z = Real.sqrt 2 / 2) ∨
     (x = -Real.sqrt 2 / 2 ∧ y = -Real.sqrt 2 / 2 ∧ z = -Real.sqrt 2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l2359_235946


namespace NUMINAMATH_CALUDE_triangle_area_l2359_235917

/-- Given a triangle with perimeter 36 and inradius 2.5, its area is 45 -/
theorem triangle_area (p : ℝ) (r : ℝ) (area : ℝ) 
    (h1 : p = 36) (h2 : r = 2.5) (h3 : area = r * p / 2) : area = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2359_235917


namespace NUMINAMATH_CALUDE_vector_operation_proof_l2359_235916

def vector1 : ℝ × ℝ := (4, -5)
def vector2 : ℝ × ℝ := (-2, 8)

theorem vector_operation_proof :
  2 • (vector1 + vector2) = (4, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l2359_235916


namespace NUMINAMATH_CALUDE_line_arrangements_l2359_235914

/-- The number of different arrangements for 3 boys and 4 girls standing in a line under various conditions -/
theorem line_arrangements (n : ℕ) (boys : ℕ) (girls : ℕ) : 
  boys = 3 → girls = 4 → n = boys + girls →
  (∃ (arrangements_1 arrangements_2 arrangements_3 arrangements_4 : ℕ),
    /- Condition 1: Person A and B must stand at the two ends -/
    arrangements_1 = 240 ∧
    /- Condition 2: Person A cannot stand at the left end, and person B cannot stand at the right end -/
    arrangements_2 = 3720 ∧
    /- Condition 3: Person A and B must stand next to each other -/
    arrangements_3 = 1440 ∧
    /- Condition 4: The 3 boys are arranged from left to right in descending order of height -/
    arrangements_4 = 840) :=
by sorry

end NUMINAMATH_CALUDE_line_arrangements_l2359_235914


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2359_235948

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 4*x₁ + 3 = 0) ∧ 
  (x₂^2 - 4*x₂ + 3 = 0) ∧ 
  x₁ = 3 ∧ 
  x₂ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2359_235948


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l2359_235923

/-- Represents the number of classes available --/
def num_classes : ℕ := 3

/-- Represents the number of students requesting to change classes --/
def num_students : ℕ := 4

/-- Represents the maximum number of additional students a class can accept --/
def max_per_class : ℕ := 2

/-- Calculates the number of ways to distribute students among classes --/
def distribution_ways : ℕ := 54

/-- Theorem stating the number of ways to distribute students --/
theorem student_distribution_theorem :
  (num_classes = 3) →
  (num_students = 4) →
  (max_per_class = 2) →
  distribution_ways = 54 :=
by
  sorry

#check student_distribution_theorem

end NUMINAMATH_CALUDE_student_distribution_theorem_l2359_235923


namespace NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l2359_235920

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a symmetry operation
def symmetryOperation (Q : Quadrilateral) : Quadrilateral := sorry

-- Define a cyclic quadrilateral
def isCyclic (Q : Quadrilateral) : Prop := sorry

-- Define a permissible quadrilateral
def isPermissible (Q : Quadrilateral) : Prop := sorry

-- Define equality of quadrilaterals
def equalQuadrilaterals (Q1 Q2 : Quadrilateral) : Prop := sorry

-- Define the application of n symmetry operations
def applyNOperations (Q : Quadrilateral) (n : ℕ) : Quadrilateral := sorry

theorem quadrilateral_symmetry_theorem (Q : Quadrilateral) :
  (isCyclic Q → equalQuadrilaterals Q (applyNOperations Q 3)) ∧
  (isPermissible Q → equalQuadrilaterals Q (applyNOperations Q 6)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l2359_235920


namespace NUMINAMATH_CALUDE_roulette_probability_l2359_235953

theorem roulette_probability (p_X p_Y p_Z p_W : ℚ) : 
  p_X = 1/4 → p_Y = 1/3 → p_W = 1/6 → p_X + p_Y + p_Z + p_W = 1 → p_Z = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_roulette_probability_l2359_235953


namespace NUMINAMATH_CALUDE_rhombus_area_l2359_235995

/-- Given a rhombus with perimeter 48 and sum of diagonals 26, its area is 25 -/
theorem rhombus_area (perimeter : ℝ) (diagonal_sum : ℝ) (area : ℝ) : 
  perimeter = 48 → diagonal_sum = 26 → area = 25 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2359_235995


namespace NUMINAMATH_CALUDE_journey_time_proof_l2359_235976

/-- Proves that a journey of 224 km, divided into two equal halves with different speeds, takes 10 hours -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 224)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l2359_235976


namespace NUMINAMATH_CALUDE_m_minus_n_equals_negative_interval_l2359_235989

-- Define the sets M and N
def M : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem m_minus_n_equals_negative_interval :
  setDifference M N = {x | -3 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_negative_interval_l2359_235989


namespace NUMINAMATH_CALUDE_cubic_derivative_value_l2359_235927

theorem cubic_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) 
  (h1 : ∀ x, f x = x^3)
  (h2 : deriv f x₀ = 3) :
  x₀ = 1 ∨ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_derivative_value_l2359_235927


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2359_235956

theorem max_value_of_expression (x : ℝ) : (2*x^2 + 8*x + 16) / (2*x^2 + 8*x + 6) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2359_235956


namespace NUMINAMATH_CALUDE_linear_function_property_l2359_235981

theorem linear_function_property (a : ℝ) :
  (∃ y : ℝ, y = a * 3 + (1 - a) ∧ y = 7) →
  (∃ y : ℝ, y = a * 8 + (1 - a) ∧ y = 22) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_property_l2359_235981


namespace NUMINAMATH_CALUDE_circle_diameter_l2359_235921

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = π * r^2 → A = 196 * π → d = 2 * r → d = 28 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l2359_235921


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_68_l2359_235986

theorem consecutive_even_integers_sum_68 :
  ∃ (x y z w : ℕ+), 
    (x : ℤ) + y + z + w = 68 ∧
    y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧
    Even x ∧ Even y ∧ Even z ∧ Even w :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_68_l2359_235986


namespace NUMINAMATH_CALUDE_sum_of_digits_of_squared_repeated_ones_l2359_235943

/-- The number formed by repeating the digit '1' eight times -/
def repeated_ones : ℕ := 11111111

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_squared_repeated_ones : sum_of_digits (repeated_ones ^ 2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_squared_repeated_ones_l2359_235943


namespace NUMINAMATH_CALUDE_exactly_two_late_probability_l2359_235932

/-- The probability of a worker being late on any given day -/
def p_late : ℚ := 1 / 40

/-- The probability of a worker being on time on any given day -/
def p_on_time : ℚ := 1 - p_late

/-- The number of workers considered -/
def n_workers : ℕ := 3

/-- The number of workers that need to be late -/
def n_late : ℕ := 2

theorem exactly_two_late_probability :
  (n_workers.choose n_late : ℚ) * p_late ^ n_late * p_on_time ^ (n_workers - n_late) = 117 / 64000 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_late_probability_l2359_235932


namespace NUMINAMATH_CALUDE_toy_truck_cost_l2359_235952

/-- The amount spent on toy trucks, given the total spent on toys and the costs of toy cars and skateboard. -/
theorem toy_truck_cost (total_toys : ℚ) (toy_cars : ℚ) (skateboard : ℚ) 
  (h1 : total_toys = 25.62)
  (h2 : toy_cars = 14.88)
  (h3 : skateboard = 4.88) :
  total_toys - (toy_cars + skateboard) = 5.86 := by
  sorry

end NUMINAMATH_CALUDE_toy_truck_cost_l2359_235952


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l2359_235912

theorem pure_imaginary_modulus (b : ℝ) : 
  (∃ y : ℝ, (1 + b * Complex.I) * (2 - Complex.I) = y * Complex.I) → 
  Complex.abs (1 + b * Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l2359_235912


namespace NUMINAMATH_CALUDE_exactly_three_true_l2359_235925

-- Define the propositions
def prop1 : Prop := ∀ p q : Prop, (p ∧ q → False) → (p → False) ∧ (q → False)

def prop2 : Prop := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)

def prop3 : Prop := 
  let p : ℝ → Prop := λ x ↦ x ≤ 1
  let q : ℝ → Prop := λ x ↦ 1 / x < 1
  (∀ x : ℝ, ¬(p x) → q x) ∧ ¬(∀ x : ℝ, q x → ¬(p x))

noncomputable def prop4 : Prop :=
  let X : ℝ → ℝ := λ _ ↦ 0  -- Placeholder for normal distribution
  ∀ C : ℝ, (∀ x : ℝ, (X x > C + 1) ↔ (X x < C - 1)) → C = 3

-- The main theorem
theorem exactly_three_true : 
  (¬prop1 ∧ prop2 ∧ prop3 ∧ prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_true_l2359_235925


namespace NUMINAMATH_CALUDE_allowance_calculation_l2359_235967

def initial_amount : ℕ := 10
def total_amount : ℕ := 18

theorem allowance_calculation :
  total_amount - initial_amount = 8 := by sorry

end NUMINAMATH_CALUDE_allowance_calculation_l2359_235967
