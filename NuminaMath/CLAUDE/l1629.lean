import Mathlib

namespace NUMINAMATH_CALUDE_wood_per_chair_l1629_162952

def total_wood : ℕ := 672
def wood_per_table : ℕ := 12
def num_tables : ℕ := 24
def num_chairs : ℕ := 48

theorem wood_per_chair :
  (total_wood - num_tables * wood_per_table) / num_chairs = 8 := by
  sorry

end NUMINAMATH_CALUDE_wood_per_chair_l1629_162952


namespace NUMINAMATH_CALUDE_decaf_percentage_l1629_162977

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 30)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := initial_stock * (initial_decaf_percent / 100) +
                     additional_stock * (additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 36 := by
sorry


end NUMINAMATH_CALUDE_decaf_percentage_l1629_162977


namespace NUMINAMATH_CALUDE_jm_length_l1629_162910

/-- Triangle DEF with medians and centroid -/
structure TriangleWithCentroid where
  -- Define the triangle
  DE : ℝ
  DF : ℝ
  EF : ℝ
  -- Define the centroid
  J : ℝ × ℝ
  -- Define M as the foot of the altitude from J to EF
  M : ℝ × ℝ

/-- The theorem stating the length of JM in the given triangle -/
theorem jm_length (t : TriangleWithCentroid) 
  (h1 : t.DE = 14) 
  (h2 : t.DF = 15) 
  (h3 : t.EF = 21) : 
  Real.sqrt ((t.J.1 - t.M.1)^2 + (t.J.2 - t.M.2)^2) = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_jm_length_l1629_162910


namespace NUMINAMATH_CALUDE_farm_ratio_l1629_162951

/-- Given a farm with horses and cows, prove that the initial ratio of horses to cows is 4:1 --/
theorem farm_ratio (initial_horses initial_cows : ℕ) : 
  (initial_horses - 15 : ℚ) / (initial_cows + 15 : ℚ) = 7 / 3 →
  (initial_horses - 15) = (initial_cows + 15 + 60) →
  (initial_horses : ℚ) / initial_cows = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_farm_ratio_l1629_162951


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l1629_162907

theorem no_solution_implies_a_leq_8 (a : ℝ) :
  (∀ x : ℝ, ¬(abs (x - 5) + abs (x + 3) < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l1629_162907


namespace NUMINAMATH_CALUDE_expression_equals_24_times_30_to_1001_l1629_162982

theorem expression_equals_24_times_30_to_1001 :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_24_times_30_to_1001_l1629_162982


namespace NUMINAMATH_CALUDE_equation_solution_l1629_162988

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (4 * x + 10) / Real.sqrt (8 * x + 2) = 2 / Real.sqrt 5) → x = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1629_162988


namespace NUMINAMATH_CALUDE_vector_relations_l1629_162998

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-3, 1)

theorem vector_relations (k : ℝ) :
  (((k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) → k = 3/2) ∧
  ((∃ t : ℝ, (k * a.1 + b.1, k * a.2 + b.2) = t • (a.1 - 3 * b.1, a.2 - 3 * b.2)) → k = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_relations_l1629_162998


namespace NUMINAMATH_CALUDE_original_manufacturing_cost_l1629_162953

/-- 
Given a fixed selling price and information about profit changes,
prove that the original manufacturing cost was $70.
-/
theorem original_manufacturing_cost
  (P : ℝ) -- Selling price
  (h1 : P - P * 0.5 = 50) -- New manufacturing cost is $50
  : P * 0.7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_original_manufacturing_cost_l1629_162953


namespace NUMINAMATH_CALUDE_evaluate_f_l1629_162933

/-- The function f(x) = 2x^2 - 4x + 9 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

/-- Theorem stating that 2f(3) + 3f(-3) = 147 -/
theorem evaluate_f : 2 * f 3 + 3 * f (-3) = 147 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l1629_162933


namespace NUMINAMATH_CALUDE_angle_C_measure_l1629_162921

-- Define a scalene triangle ABC
structure ScaleneTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C
  sum_180 : A + B + C = 180

-- Theorem statement
theorem angle_C_measure (t : ScaleneTriangle) 
  (h1 : t.B = t.A + 20)  -- Angle B is 20 degrees larger than angle A
  (h2 : t.C = 2 * t.A)   -- Angle C is twice the size of angle A
  : t.C = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1629_162921


namespace NUMINAMATH_CALUDE_positive_sum_and_product_equivalence_l1629_162978

theorem positive_sum_and_product_equivalence (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_equivalence_l1629_162978


namespace NUMINAMATH_CALUDE_proposition_relationship_l1629_162989

theorem proposition_relationship :
  (∀ x : ℝ, (0 < x ∧ x < 5) → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l1629_162989


namespace NUMINAMATH_CALUDE_college_students_count_l1629_162902

theorem college_students_count :
  ∀ (total : ℕ) (enrolled_percent : ℚ) (not_enrolled : ℕ),
    enrolled_percent = 1/2 →
    not_enrolled = 440 →
    (1 - enrolled_percent) * total = not_enrolled →
    total = 880 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l1629_162902


namespace NUMINAMATH_CALUDE_field_trip_bus_occupancy_l1629_162981

theorem field_trip_bus_occupancy
  (num_vans : ℕ)
  (num_buses : ℕ)
  (people_per_van : ℕ)
  (total_people : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_buses = 8)
  (h3 : people_per_van = 6)
  (h4 : total_people = 180)
  : (total_people - num_vans * people_per_van) / num_buses = 18 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_bus_occupancy_l1629_162981


namespace NUMINAMATH_CALUDE_thermometer_price_is_two_l1629_162955

/-- Represents the sales data for thermometers and hot-water bottles --/
structure SalesData where
  thermometer_price : ℝ
  hotwater_bottle_price : ℝ
  total_sales : ℝ
  thermometer_to_bottle_ratio : ℕ
  bottles_sold : ℕ

/-- Theorem stating that the thermometer price is 2 dollars given the sales data --/
theorem thermometer_price_is_two (data : SalesData)
  (h1 : data.hotwater_bottle_price = 6)
  (h2 : data.total_sales = 1200)
  (h3 : data.thermometer_to_bottle_ratio = 7)
  (h4 : data.bottles_sold = 60)
  : data.thermometer_price = 2 := by
  sorry


end NUMINAMATH_CALUDE_thermometer_price_is_two_l1629_162955


namespace NUMINAMATH_CALUDE_solution_in_interval_l1629_162941

open Real

theorem solution_in_interval (x₀ : ℝ) (k : ℤ) : 
  (8 - x₀ = log x₀) → 
  (x₀ ∈ Set.Ioo (k : ℝ) (k + 1)) → 
  k = 7 := by
sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1629_162941


namespace NUMINAMATH_CALUDE_fourth_square_dots_l1629_162968

/-- The number of dots in the nth square of the series -/
def dots_in_square (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else dots_in_square (n - 1) + 4 * n

theorem fourth_square_dots :
  dots_in_square 4 = 37 := by
  sorry

end NUMINAMATH_CALUDE_fourth_square_dots_l1629_162968


namespace NUMINAMATH_CALUDE_remaining_tickets_l1629_162900

/-- Given an initial number of tickets, the number of tickets lost, and the number of tickets spent,
    the remaining number of tickets is equal to the initial number minus the lost tickets minus the spent tickets. -/
theorem remaining_tickets (initial lost spent : ℝ) : 
  initial - lost - spent = initial - (lost + spent) := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_l1629_162900


namespace NUMINAMATH_CALUDE_g_at_3_equals_20_l1629_162950

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def g (x : ℝ) : ℝ := 4 * (f⁻¹ x)

theorem g_at_3_equals_20 : g 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_g_at_3_equals_20_l1629_162950


namespace NUMINAMATH_CALUDE_survivor_same_tribe_quit_probability_l1629_162963

/-- The probability of all three quitters being from the same tribe in a Survivor-like scenario -/
theorem survivor_same_tribe_quit_probability :
  let total_people : ℕ := 20
  let tribe_size : ℕ := 10
  let num_quitters : ℕ := 3
  let total_combinations := Nat.choose total_people num_quitters
  let same_tribe_combinations := 2 * Nat.choose tribe_size num_quitters
  (same_tribe_combinations : ℚ) / total_combinations = 4 / 19 := by
  sorry

end NUMINAMATH_CALUDE_survivor_same_tribe_quit_probability_l1629_162963


namespace NUMINAMATH_CALUDE_library_items_count_l1629_162957

theorem library_items_count (notebooks : ℕ) (pens : ℕ) : 
  notebooks = 30 →
  pens = notebooks + 50 →
  notebooks + pens = 110 := by
  sorry

end NUMINAMATH_CALUDE_library_items_count_l1629_162957


namespace NUMINAMATH_CALUDE_equation_with_increasing_roots_l1629_162949

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x + 1) - (m + 1) / (x^2 + x) = (x + 1) / x

-- Define the concept of increasing roots
def has_increasing_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < y ∧ equation x m ∧ equation y m

-- Theorem statement
theorem equation_with_increasing_roots (m : ℝ) :
  has_increasing_roots m → m = -2 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_with_increasing_roots_l1629_162949


namespace NUMINAMATH_CALUDE_square_coverage_l1629_162966

/-- A square can be covered by smaller squares if the total area of the smaller squares
    is greater than or equal to the area of the larger square. -/
def can_cover (large_side small_side : ℝ) (num_small_squares : ℕ) : Prop :=
  large_side^2 ≤ (small_side^2 * num_small_squares)

/-- Theorem stating that a square with side length 7 can be covered by 8 squares
    with side length 3. -/
theorem square_coverage : can_cover 7 3 8 := by
  sorry

end NUMINAMATH_CALUDE_square_coverage_l1629_162966


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l1629_162985

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility (n : ℕ) : 
  (is_divisible_by (n + 3) 12 ∧ 
   is_divisible_by (n + 3) 15 ∧ 
   is_divisible_by (n + 3) 40) →
  (∀ k : ℕ, k < n → 
    ¬(is_divisible_by (k + 3) 12 ∧ 
      is_divisible_by (k + 3) 15 ∧ 
      is_divisible_by (k + 3) 40)) →
  (∃ m : ℕ, m ≠ 12 ∧ m ≠ 15 ∧ m ≠ 40 ∧ 
    is_divisible_by (n + 3) m) →
  ∃ m : ℕ, m ≠ 12 ∧ m ≠ 15 ∧ m ≠ 40 ∧ 
    is_divisible_by (n + 3) m ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l1629_162985


namespace NUMINAMATH_CALUDE_application_methods_for_five_graduates_three_universities_l1629_162901

/-- The number of different application methods for high school graduates to universities -/
def application_methods (num_graduates : ℕ) (num_universities : ℕ) : ℕ :=
  num_universities ^ num_graduates

/-- Theorem: Given 5 high school graduates and 3 universities, where each graduate can only apply to one university, the total number of different application methods is 3^5 -/
theorem application_methods_for_five_graduates_three_universities :
  application_methods 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_application_methods_for_five_graduates_three_universities_l1629_162901


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_5_squared_and_3_cubed_l1629_162972

theorem smallest_n_divisible_by_5_squared_and_3_cubed (n : ℕ) (k : ℕ) : 
  (∃ m : ℕ, n * (2^5) * (6^k) * (7^3) = m * (5^2)) ∧ 
  (∃ m : ℕ, n * (2^5) * (6^k) * (7^3) = m * (3^3)) ∧ 
  (∀ n' : ℕ, n' < n → ¬(∃ k' : ℕ, ∃ m : ℕ, n' * (2^5) * (6^k') * (7^3) = m * (5^2)) ∨ 
                     ¬(∃ k' : ℕ, ∃ m : ℕ, n' * (2^5) * (6^k') * (7^3) = m * (3^3))) →
  n = 25 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_5_squared_and_3_cubed_l1629_162972


namespace NUMINAMATH_CALUDE_cistern_water_depth_l1629_162944

/-- Proves that for a rectangular cistern with given dimensions and wet surface area, the water depth is as calculated. -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 4)
  (h_width : width = 8)
  (h_total_area : total_wet_surface_area = 62)
  (h_depth : h = (total_wet_surface_area - length * width) / (2 * (length + width))) :
  h = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cistern_water_depth_l1629_162944


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1629_162938

theorem price_increase_percentage (lower_price higher_price : ℝ) 
  (h1 : lower_price > 0)
  (h2 : higher_price > lower_price)
  (h3 : higher_price = lower_price * 1.4) :
  (higher_price - lower_price) / lower_price * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1629_162938


namespace NUMINAMATH_CALUDE_feeding_and_trapping_sets_l1629_162994

/-- A set is a feeding set for a sequence if every open subinterval of the set contains infinitely many terms of the sequence. -/
def IsFeeder (s : Set ℝ) (seq : ℕ → ℝ) : Prop :=
  ∀ a b, a < b → a ∈ s → b ∈ s → Set.Infinite {n : ℕ | seq n ∈ Set.Ioo a b}

/-- A set is a trapping set for a sequence if no infinite subset of the sequence remains outside the set. -/
def IsTrap (s : Set ℝ) (seq : ℕ → ℝ) : Prop :=
  Set.Finite {n : ℕ | seq n ∉ s}

theorem feeding_and_trapping_sets :
  (∃ seq : ℕ → ℝ, IsFeeder (Set.Icc 0 1) seq ∧ IsFeeder (Set.Icc 2 3) seq) ∧
  (¬ ∃ seq : ℕ → ℝ, IsTrap (Set.Icc 0 1) seq ∧ IsTrap (Set.Icc 2 3) seq) :=
sorry

end NUMINAMATH_CALUDE_feeding_and_trapping_sets_l1629_162994


namespace NUMINAMATH_CALUDE_extremum_at_one_lower_bound_ln_two_l1629_162929

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

-- Theorem for the first part of the problem
theorem extremum_at_one (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Icc (1 - ε) (1 + ε), f a x ≥ f a 1) ↔ a = 1 :=
sorry

-- Theorem for the second part of the problem
theorem lower_bound_ln_two (a : ℝ) (h : a > 0) :
  (∀ x ≥ 0, f a x ≥ Real.log 2) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_extremum_at_one_lower_bound_ln_two_l1629_162929


namespace NUMINAMATH_CALUDE_wood_weight_calculation_l1629_162913

/-- Given a square piece of wood with side length 4 inches weighing 20 ounces,
    calculate the weight of a second square piece with side length 7 inches. -/
theorem wood_weight_calculation (thickness : ℝ) (density : ℝ) :
  let side1 : ℝ := 4
  let weight1 : ℝ := 20
  let side2 : ℝ := 7
  let area1 : ℝ := side1 * side1
  let area2 : ℝ := side2 * side2
  let weight2 : ℝ := weight1 * (area2 / area1)
  weight2 = 61.25 := by sorry

end NUMINAMATH_CALUDE_wood_weight_calculation_l1629_162913


namespace NUMINAMATH_CALUDE_home_theater_savings_l1629_162945

def in_store_price : ℝ := 320
def in_store_discount : ℝ := 0.05
def website_monthly_payment : ℝ := 62
def website_num_payments : ℕ := 5
def website_shipping : ℝ := 10

theorem home_theater_savings :
  let website_total := website_monthly_payment * website_num_payments + website_shipping
  let in_store_discounted := in_store_price * (1 - in_store_discount)
  website_total - in_store_discounted = 16 := by sorry

end NUMINAMATH_CALUDE_home_theater_savings_l1629_162945


namespace NUMINAMATH_CALUDE_parabola_and_line_problem_l1629_162918

-- Define the parabola and directrix
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := y = x
def l₂ (x y : ℝ) : Prop := y = -x

-- Define the point E
def E : ℝ × ℝ := (4, 1)

-- Define the circle N
def circle_N (center : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = r^2

-- Theorem statement
theorem parabola_and_line_problem :
  -- Part 1: The coordinates of N are (2, 0)
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x y : ℝ), parabola p x y → 
    (∀ (x' : ℝ), directrix p x' → 
      ((x - x')^2 + y^2 = (x - 2)^2 + y^2))) ∧
  -- Part 2: No line l exists satisfying all conditions
  ¬∃ (m b : ℝ), 
    -- Define line l: y = mx + b
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      -- l intersects l₁ and l₂
      (y₁ = m*x₁ + b ∧ l₁ x₁ y₁) ∧
      (y₂ = m*x₂ + b ∧ l₂ x₂ y₂) ∧
      -- Midpoint of intersection points is E
      ((x₁ + x₂)/2 = E.1 ∧ (y₁ + y₂)/2 = E.2) ∧
      -- Chord length on circle N is 2
      (∃ (r : ℝ), 
        circle_N (2, 0) r 2 2 ∧ 
        circle_N (2, 0) r 2 (-2) ∧
        ∃ (x₃ y₃ x₄ y₄ : ℝ),
          y₃ = m*x₃ + b ∧ y₄ = m*x₄ + b ∧
          circle_N (2, 0) r x₃ y₃ ∧
          circle_N (2, 0) r x₄ y₄ ∧
          (x₃ - x₄)^2 + (y₃ - y₄)^2 = 4)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_problem_l1629_162918


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_l1629_162934

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p + q = 50 ∧ 
    p * q = k ∧ 
    k = 141 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_l1629_162934


namespace NUMINAMATH_CALUDE_steering_wheel_translational_on_straight_road_l1629_162967

/-- A road is considered straight if it has no curves or turns. -/
def is_straight_road (road : Type) : Prop := sorry

/-- A motion is translational if it involves no rotation. -/
def is_translational_motion (motion : Type) : Prop := sorry

/-- The steering wheel motion when driving on a given road. -/
def steering_wheel_motion (road : Type) : Type := sorry

/-- Theorem: The steering wheel motion is translational when driving on a straight road. -/
theorem steering_wheel_translational_on_straight_road (road : Type) :
  is_straight_road road → is_translational_motion (steering_wheel_motion road) := by sorry

end NUMINAMATH_CALUDE_steering_wheel_translational_on_straight_road_l1629_162967


namespace NUMINAMATH_CALUDE_tangent_line_implies_b_equals_3_l1629_162940

/-- A line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (k a b : ℝ) : Prop :=
  -- The point (1, 3) lies on both the line and the curve
  3 = k * 1 + 1 ∧ 3 = 1^3 + a * 1 + b ∧
  -- The slopes of the line and the curve are equal at (1, 3)
  k = 3 * 1^2 + a

theorem tangent_line_implies_b_equals_3 (k a b : ℝ) :
  is_tangent k a b → b = 3 := by
  sorry

#check tangent_line_implies_b_equals_3

end NUMINAMATH_CALUDE_tangent_line_implies_b_equals_3_l1629_162940


namespace NUMINAMATH_CALUDE_new_energy_vehicles_analysis_l1629_162975

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 149.24 * x - 33.64

-- Define the stock data for years 2017 to 2021
def stock_data : List (ℕ × ℝ) := [
  (1, 153.4),
  (2, 260.8),
  (3, 380.2),
  (4, 492.0),
  (5, 784.0)
]

-- Theorem statement
theorem new_energy_vehicles_analysis :
  -- 1. Predicted stock for 2023 exceeds 1000 million vehicles
  (regression_eq 7 > 1000) ∧
  -- 2. Stock shows increasing trend from 2017 to 2021
  (∀ i j, i < j → i ∈ List.map Prod.fst stock_data → j ∈ List.map Prod.fst stock_data →
    (stock_data.find? (λ p => p.fst = i)).map Prod.snd < (stock_data.find? (λ p => p.fst = j)).map Prod.snd) ∧
  -- 3. Residual for 2021 is 71.44
  (((stock_data.find? (λ p => p.fst = 5)).map Prod.snd).getD 0 - regression_eq 5 = 71.44) := by
  sorry

end NUMINAMATH_CALUDE_new_energy_vehicles_analysis_l1629_162975


namespace NUMINAMATH_CALUDE_system_of_equations_l1629_162995

theorem system_of_equations (x y : ℚ) 
  (eq1 : 4 * x + y = 8) 
  (eq2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l1629_162995


namespace NUMINAMATH_CALUDE_sixth_root_of_12984301300421_l1629_162996

theorem sixth_root_of_12984301300421 : 
  (12984301300421 : ℝ) ^ (1/6 : ℝ) = 51 := by sorry

end NUMINAMATH_CALUDE_sixth_root_of_12984301300421_l1629_162996


namespace NUMINAMATH_CALUDE_halloween_candy_count_l1629_162922

/-- The number of candy pieces remaining after Halloween --/
def remaining_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  debby_candy + sister_candy - eaten_candy

/-- Theorem stating the remaining candy count for the given scenario --/
theorem halloween_candy_count : remaining_candy 32 42 35 = 39 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l1629_162922


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1629_162920

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1629_162920


namespace NUMINAMATH_CALUDE_rabbit_problem_l1629_162927

/-- The cost price of an Auspicious Rabbit -/
def auspicious_cost : ℝ := 40

/-- The cost price of a Lucky Rabbit -/
def lucky_cost : ℝ := 44

/-- The selling price of an Auspicious Rabbit -/
def auspicious_price : ℝ := 60

/-- The selling price of a Lucky Rabbit -/
def lucky_price : ℝ := 70

/-- The total number of rabbits to be purchased -/
def total_rabbits : ℕ := 200

/-- The minimum required profit -/
def min_profit : ℝ := 4120

/-- The quantity ratio of Lucky Rabbits to Auspicious Rabbits based on the given costs -/
axiom quantity_ratio : (8800 / lucky_cost) = 2 * (4000 / auspicious_cost)

/-- The cost difference between Lucky and Auspicious Rabbits -/
axiom cost_difference : lucky_cost = auspicious_cost + 4

/-- Theorem stating the correct cost prices and minimum number of Lucky Rabbits -/
theorem rabbit_problem :
  (auspicious_cost = 40 ∧ lucky_cost = 44) ∧
  (∀ m : ℕ, m ≥ 20 →
    (lucky_price - lucky_cost) * m + (auspicious_price - auspicious_cost) * (total_rabbits - m) ≥ min_profit) ∧
  (∀ m : ℕ, m < 20 →
    (lucky_price - lucky_cost) * m + (auspicious_price - auspicious_cost) * (total_rabbits - m) < min_profit) :=
sorry

end NUMINAMATH_CALUDE_rabbit_problem_l1629_162927


namespace NUMINAMATH_CALUDE_area_of_graph_l1629_162974

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := abs (2 * x) + abs (3 * y) = 6

/-- The set of points satisfying the equation -/
def graph_set : Set (ℝ × ℝ) := {p | graph_equation p.1 p.2}

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

theorem area_of_graph : enclosed_area = 12 := by sorry

end NUMINAMATH_CALUDE_area_of_graph_l1629_162974


namespace NUMINAMATH_CALUDE_a_range_l1629_162991

/-- The inequality holds for all positive real x -/
def inequality_holds (a : ℝ) : Prop :=
  ∀ x > 0, a * Real.log (a * x) ≤ Real.exp x

/-- The theorem stating the range of a given the inequality -/
theorem a_range (a : ℝ) (h : inequality_holds a) : 0 < a ∧ a ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l1629_162991


namespace NUMINAMATH_CALUDE_max_erased_dots_l1629_162923

/-- Represents a domino tile with two halves -/
structure Domino :=
  (left : ℕ)
  (right : ℕ)

/-- The problem setup -/
def DominoArrangement :=
  { tiles : List Domino // tiles.length = 8 }

/-- The sum of dots on all visible tiles -/
def visibleDots (arr : DominoArrangement) : ℕ :=
  (arr.val.take 7).foldl (fun acc tile => acc + tile.left + tile.right) 0

/-- The total number of dots including the erased half -/
def totalDots (arr : DominoArrangement) (erased : ℕ) : ℕ :=
  visibleDots arr + erased

theorem max_erased_dots (arr : DominoArrangement) 
  (h1 : visibleDots arr = 37)
  (h2 : ∀ n : ℕ, totalDots arr n % 4 = 0 → n ≤ 3) :
  ∃ (n : ℕ), n ≤ 3 ∧ totalDots arr n % 4 = 0 ∧ 
    ∀ (m : ℕ), totalDots arr m % 4 = 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_max_erased_dots_l1629_162923


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1629_162916

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation 
  (l : Line)
  (A : Point)
  (given_line : Line)
  (h1 : A.liesOn l)
  (h2 : l.perpendicular given_line)
  (h3 : A.x = -1)
  (h4 : A.y = 3)
  (h5 : given_line.a = 1)
  (h6 : given_line.b = -2)
  (h7 : given_line.c = -3) :
  l.a = 2 ∧ l.b = 1 ∧ l.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1629_162916


namespace NUMINAMATH_CALUDE_expand_expression_l1629_162983

theorem expand_expression (x : ℝ) : (17 * x + 12) * (3 * x) = 51 * x^2 + 36 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1629_162983


namespace NUMINAMATH_CALUDE_y_coordinate_abs_value_l1629_162980

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- The distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

theorem y_coordinate_abs_value (p : Point) 
  (h1 : distToXAxis p = (1/2) * distToYAxis p) 
  (h2 : distToYAxis p = 12) : 
  |p.y| = 6 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_abs_value_l1629_162980


namespace NUMINAMATH_CALUDE_monogram_count_l1629_162911

/-- The number of letters in the alphabet before 'G' -/
def letters_before_g : Nat := 6

/-- The number of letters in the alphabet after 'G' -/
def letters_after_g : Nat := 18

/-- The total number of possible monograms with 'G' as the middle initial,
    and the other initials different and in alphabetical order -/
def total_monograms : Nat := letters_before_g * letters_after_g

theorem monogram_count :
  total_monograms = 108 := by
  sorry

end NUMINAMATH_CALUDE_monogram_count_l1629_162911


namespace NUMINAMATH_CALUDE_unique_number_property_l1629_162919

theorem unique_number_property : ∃! n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n.div 100 + n.mod 100 / 10 + n.mod 10 = 328 - n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l1629_162919


namespace NUMINAMATH_CALUDE_rose_bed_fraction_l1629_162925

/-- Proof that the rose bed occupies 1/20 of the park's area given the conditions -/
theorem rose_bed_fraction (park_length park_width : ℝ) 
  (flower_bed_fraction : ℝ) (rose_bed_fraction : ℝ) :
  park_length = 15 →
  park_width = 20 →
  flower_bed_fraction = 1/5 →
  rose_bed_fraction = 1/4 →
  (flower_bed_fraction * rose_bed_fraction * park_length * park_width) / 
  (park_length * park_width) = 1/20 := by
  sorry

#check rose_bed_fraction

end NUMINAMATH_CALUDE_rose_bed_fraction_l1629_162925


namespace NUMINAMATH_CALUDE_line_equation_l1629_162987

/-- Given a line with an angle of inclination of 45° and a y-intercept of 2,
    its equation is x - y + 2 = 0 -/
theorem line_equation (angle : ℝ) (y_intercept : ℝ) :
  angle = 45 ∧ y_intercept = 2 →
  ∀ x y : ℝ, (y = x + y_intercept) ↔ (x - y + y_intercept = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1629_162987


namespace NUMINAMATH_CALUDE_f_difference_l1629_162926

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(420) - f(360) = 143/20 -/
theorem f_difference : f 420 - f 360 = 143 / 20 := by sorry

end NUMINAMATH_CALUDE_f_difference_l1629_162926


namespace NUMINAMATH_CALUDE_max_true_statements_l1629_162958

theorem max_true_statements (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ a < b →
  (1 / a < 1 / b) ∧ 
  (a^2 > b^2) ∧ 
  (a < b) ∧ 
  (a > 0) ∧ 
  (b > 0) := by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l1629_162958


namespace NUMINAMATH_CALUDE_team_arrangement_solution_l1629_162943

/-- The total number of team members -/
def total_members : ℕ := 1000

/-- The minimum number of rows required -/
def min_rows : ℕ := 17

/-- Calculates the sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + n - 1) / 2

/-- Theorem stating the solution to the team arrangement problem -/
theorem team_arrangement_solution :
  ∃ (n : ℕ) (a : ℕ),
    n > min_rows ∧
    arithmetic_sum a n = total_members ∧
    n = 25 ∧
    a = 28 := by
  sorry

end NUMINAMATH_CALUDE_team_arrangement_solution_l1629_162943


namespace NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l1629_162914

/-- A set of four points in three-dimensional space. -/
structure FourPoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ
  p4 : ℝ × ℝ × ℝ

/-- Predicate to check if four points are non-coplanar. -/
def NonCoplanar (points : FourPoints) : Prop := sorry

/-- The number of planes determined by a set of four points. -/
def NumPlanesDetermined (points : FourPoints) : ℕ := sorry

/-- Theorem stating that four non-coplanar points determine exactly four planes. -/
theorem four_noncoplanar_points_determine_four_planes (points : FourPoints) :
  NonCoplanar points → NumPlanesDetermined points = 4 := by sorry

end NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l1629_162914


namespace NUMINAMATH_CALUDE_log_simplification_l1629_162908

theorem log_simplification (p q r s t z : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * t / (s * z)) = Real.log (z / t) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l1629_162908


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1629_162924

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 997)
  (h2 : divisor = 23)
  (h3 : quotient = 43)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 8 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1629_162924


namespace NUMINAMATH_CALUDE_scooter_price_l1629_162909

/-- Given an upfront payment of 20% of the total cost, which amounts to $240, prove that the total price of the scooter is $1200. -/
theorem scooter_price (upfront_percentage : ℝ) (upfront_amount : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_amount = 240 → 
  upfront_percentage * total_price = upfront_amount → 
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_l1629_162909


namespace NUMINAMATH_CALUDE_triangle_side_length_l1629_162976

theorem triangle_side_length (A B C : Real) (angleB : Real) (sideAB sideAC : Real) :
  angleB = π / 4 →
  sideAB = 100 →
  sideAC = 100 →
  (∃! bc : Real, bc = sideAB * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1629_162976


namespace NUMINAMATH_CALUDE_minimum_travel_cost_l1629_162937

-- Define the cities and distances
def X : City := sorry
def Y : City := sorry
def Z : City := sorry

-- Define the distances
def distance_XY : ℝ := 5000
def distance_XZ : ℝ := 4000

-- Define the cost functions
def bus_cost (distance : ℝ) : ℝ := 0.2 * distance
def plane_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the theorem
theorem minimum_travel_cost :
  ∃ (cost : ℝ),
    cost = plane_cost distance_XY + 
           plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
           plane_cost distance_XZ ∧
    cost = 2250 ∧
    ∀ (alternative_cost : ℝ),
      (alternative_cost = bus_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ) →
      cost ≤ alternative_cost :=
by sorry

end NUMINAMATH_CALUDE_minimum_travel_cost_l1629_162937


namespace NUMINAMATH_CALUDE_grandma_contribution_correct_l1629_162942

/-- The amount Zoe's grandma gave her for the trip -/
def grandma_contribution (total_cost : ℝ) (candy_bars : ℕ) (profit_per_bar : ℝ) : ℝ :=
  total_cost - (candy_bars : ℝ) * profit_per_bar

/-- Proof that the grandma's contribution is correct -/
theorem grandma_contribution_correct (total_cost : ℝ) (candy_bars : ℕ) (profit_per_bar : ℝ) :
  grandma_contribution total_cost candy_bars profit_per_bar =
  total_cost - (candy_bars : ℝ) * profit_per_bar :=
by sorry

end NUMINAMATH_CALUDE_grandma_contribution_correct_l1629_162942


namespace NUMINAMATH_CALUDE_johns_age_difference_l1629_162993

theorem johns_age_difference (brother_age : ℕ) (john_age : ℕ) : 
  brother_age = 8 → 
  john_age + brother_age = 10 → 
  6 * brother_age - john_age = 46 := by
sorry

end NUMINAMATH_CALUDE_johns_age_difference_l1629_162993


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1629_162905

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 8 where f(-2) = 10, prove that f(2) = 6 -/
theorem polynomial_evaluation (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 8
  f (-2) = 10 → f 2 = 6 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1629_162905


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1629_162969

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (-15/8, 13/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 4 * y = -8 * x - 2

theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1629_162969


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l1629_162970

-- Define the function f(x) = x + 2
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem sum_of_max_and_min_is_eight :
  let a : ℝ := 0
  let b : ℝ := 4
  (∀ x ∈ Set.Icc a b, f x ≤ f b) ∧ 
  (∀ x ∈ Set.Icc a b, f a ≤ f x) →
  f a + f b = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l1629_162970


namespace NUMINAMATH_CALUDE_increase_in_circumference_l1629_162948

/-- The increase in circumference of a circle when its diameter increases by π units -/
theorem increase_in_circumference (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + π)
  let increase := new_circumference - original_circumference
  increase = π^2 := by sorry

end NUMINAMATH_CALUDE_increase_in_circumference_l1629_162948


namespace NUMINAMATH_CALUDE_barber_loss_l1629_162964

/-- Represents the monetary transactions in the barbershop scenario -/
structure BarbershopScenario where
  haircut_price : ℕ
  counterfeit_bill : ℕ
  change_given : ℕ
  replacement_bill : ℕ

/-- Calculates the total loss for the barber in the given scenario -/
def calculate_loss (scenario : BarbershopScenario) : ℕ :=
  scenario.haircut_price + scenario.change_given + scenario.replacement_bill - scenario.counterfeit_bill

/-- Theorem stating that the barber's loss in the given scenario is $25 -/
theorem barber_loss (scenario : BarbershopScenario) 
  (h1 : scenario.haircut_price = 15)
  (h2 : scenario.counterfeit_bill = 20)
  (h3 : scenario.change_given = 5)
  (h4 : scenario.replacement_bill = 20) :
  calculate_loss scenario = 25 := by
  sorry

end NUMINAMATH_CALUDE_barber_loss_l1629_162964


namespace NUMINAMATH_CALUDE_julian_borrows_eight_l1629_162992

/-- The amount Julian borrows -/
def additional_borrowed (current_debt new_debt : ℕ) : ℕ :=
  new_debt - current_debt

/-- Proof that Julian borrows 8 dollars -/
theorem julian_borrows_eight :
  let current_debt := 20
  let new_debt := 28
  additional_borrowed current_debt new_debt = 8 := by
  sorry

end NUMINAMATH_CALUDE_julian_borrows_eight_l1629_162992


namespace NUMINAMATH_CALUDE_max_sections_school_l1629_162932

theorem max_sections_school (num_boys : ℕ) (num_girls : ℕ) (min_boys_per_section : ℕ) (min_girls_per_section : ℕ) 
  (h1 : num_boys = 2016) 
  (h2 : num_girls = 1284) 
  (h3 : min_boys_per_section = 80) 
  (h4 : min_girls_per_section = 60) : 
  (num_boys / min_boys_per_section + num_girls / min_girls_per_section : ℕ) = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sections_school_l1629_162932


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1629_162917

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1629_162917


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1629_162946

theorem fraction_subtraction : (18 : ℚ) / 45 - 3 / 8 = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1629_162946


namespace NUMINAMATH_CALUDE_rectangle_square_cut_l1629_162959

theorem rectangle_square_cut (m n : ℕ) (hm : m > 2) (hn : n > 2) :
  (m - 2) * (n - 2) = 8 ↔
  (2 * (m + n) - 4 = m * n) ∧ (m * n - 4 = 2 * (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_cut_l1629_162959


namespace NUMINAMATH_CALUDE_prob_not_pass_overall_prob_pass_technical_given_overall_l1629_162973

-- Define the probabilities of not passing each review aspect
def p_not_pass_norms : ℚ := 4/25
def p_not_pass_account : ℚ := 13/48
def p_not_pass_content : ℚ := 1/5

-- Define the probability of passing both overall review and technical skills test
def p_pass_both : ℚ := 35/100

-- Theorem for the probability of not passing overall review
theorem prob_not_pass_overall : 
  1 - (1 - p_not_pass_norms) * (1 - p_not_pass_account) * (1 - p_not_pass_content) = 51/100 := by sorry

-- Theorem for the probability of passing technical skills test given passing overall review
theorem prob_pass_technical_given_overall : 
  let p_pass_overall := 1 - (1 - (1 - p_not_pass_norms) * (1 - p_not_pass_account) * (1 - p_not_pass_content))
  p_pass_both / p_pass_overall = 5/7 := by sorry

end NUMINAMATH_CALUDE_prob_not_pass_overall_prob_pass_technical_given_overall_l1629_162973


namespace NUMINAMATH_CALUDE_range_of_a_l1629_162965

def A (a : ℝ) := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B := {x : ℝ | x < 0 ∨ x > 19}

theorem range_of_a (a : ℝ) : 
  (A a ⊆ (A a ∩ B)) → (a < 6 ∨ a > 9) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1629_162965


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1629_162915

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 + 100 * x + 9 * y^2 - 36 * y = 225

/-- The distance between the foci of the ellipse -/
def foci_distance : ℝ := 10.134

/-- Theorem: The distance between the foci of the ellipse defined by the given equation
    is approximately 10.134 -/
theorem ellipse_foci_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧
  (∀ x y : ℝ, ellipse_equation x y → abs (foci_distance - 10.134) < ε) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1629_162915


namespace NUMINAMATH_CALUDE_fraction_equality_l1629_162971

theorem fraction_equality (a b c x : ℝ) (hx : x = a / b) (hc : c ≠ 0) (hb : b ≠ 0) (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1629_162971


namespace NUMINAMATH_CALUDE_married_men_fraction_l1629_162984

theorem married_men_fraction (total_women : ℕ) (total_people : ℕ) 
  (h_women_positive : total_women > 0)
  (h_total_positive : total_people > 0)
  (h_single_prob : (3 : ℚ) / 7 = (total_women - (total_people - total_women)) / total_women) :
  (total_people - total_women) / total_people = (4 : ℚ) / 11 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l1629_162984


namespace NUMINAMATH_CALUDE_T_is_Y_shape_l1629_162960

/-- The set T of points (x, y) in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 6 < 5) ∨
               (5 = y - 6 ∧ x + 3 < 5) ∨
               (x + 3 = y - 6 ∧ 5 < x + 3)}

/-- The common start point of the "Y" shape -/
def commonPoint : ℝ × ℝ := (2, 11)

/-- The vertical line segment of the "Y" shape -/
def verticalSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 2 ∧ p.2 < 11}

/-- The horizontal line segment of the "Y" shape -/
def horizontalSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 11 ∧ p.1 < 2}

/-- The diagonal ray of the "Y" shape -/
def diagonalRay : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 > 2}

theorem T_is_Y_shape :
  T = verticalSegment ∪ horizontalSegment ∪ diagonalRay ∧
  commonPoint ∈ T ∧
  commonPoint ∈ verticalSegment ∧
  commonPoint ∈ horizontalSegment ∧
  commonPoint ∈ diagonalRay :=
sorry

end NUMINAMATH_CALUDE_T_is_Y_shape_l1629_162960


namespace NUMINAMATH_CALUDE_count_multiples_of_three_is_12960_l1629_162999

/-- A function that returns the count of six-digit multiples of 3 where each digit is not greater than 5 -/
def count_multiples_of_three : ℕ :=
  let first_digit_options := 5  -- digits 1 to 5
  let other_digit_options := 6  -- digits 0 to 5
  let last_digit_options := 2   -- two options to make the sum divisible by 3
  first_digit_options * (other_digit_options ^ 4) * last_digit_options

/-- Theorem stating that the count of six-digit multiples of 3 where each digit is not greater than 5 is 12960 -/
theorem count_multiples_of_three_is_12960 : count_multiples_of_three = 12960 := by
  sorry

#eval count_multiples_of_three

end NUMINAMATH_CALUDE_count_multiples_of_three_is_12960_l1629_162999


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1629_162961

theorem quadratic_equation_result (x : ℝ) : 
  2 * x^2 - 5 = 11 → 
  (4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2) ∨ 
  (4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1629_162961


namespace NUMINAMATH_CALUDE_initial_nurses_count_l1629_162979

/-- Proves that the initial number of nurses is 18 given the conditions of the problem -/
theorem initial_nurses_count (initial_doctors : ℕ) (quit_doctors quit_nurses remaining_staff : ℕ) 
  (h1 : initial_doctors = 11)
  (h2 : quit_doctors = 5)
  (h3 : quit_nurses = 2)
  (h4 : remaining_staff = 22)
  (h5 : initial_doctors - quit_doctors + (initial_nurses - quit_nurses) = remaining_staff) :
  initial_nurses = 18 :=
by
  sorry
where
  initial_nurses : ℕ := by sorry

end NUMINAMATH_CALUDE_initial_nurses_count_l1629_162979


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1629_162986

theorem functional_equation_solution (a : ℝ) (ha : a ≠ 0) :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f (a + x) = f x - x) →
  ∃ C : ℝ, ∀ x : ℝ, f x = C + x^2 / (2 * a) - x / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1629_162986


namespace NUMINAMATH_CALUDE_exists_n_with_factorial_property_and_digit_sum_l1629_162936

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem -/
theorem exists_n_with_factorial_property_and_digit_sum :
  ∃ n : ℕ, n > 0 ∧ 
    (Nat.factorial (n + 1) + Nat.factorial (n + 2) = Nat.factorial n * 1001) ∧
    (sum_of_digits n = 3) := by
  sorry

end NUMINAMATH_CALUDE_exists_n_with_factorial_property_and_digit_sum_l1629_162936


namespace NUMINAMATH_CALUDE_geometric_progression_existence_l1629_162990

/-- A geometric progression containing 27, 8, and 12 exists, and their positions satisfy m = 3p - 2n -/
theorem geometric_progression_existence :
  ∃ (a q : ℝ) (m n p : ℕ), 
    (a * q^(m-1) = 27) ∧ 
    (a * q^(n-1) = 8) ∧ 
    (a * q^(p-1) = 12) ∧ 
    (m = 3*p - 2*n) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_existence_l1629_162990


namespace NUMINAMATH_CALUDE_range_of_a_l1629_162997

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) →
  (∃ x : ℝ, x^2 + 4*x + a = 0) →
  a ∈ Set.Icc (Real.exp 1) 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1629_162997


namespace NUMINAMATH_CALUDE_simplify_expression_l1629_162904

theorem simplify_expression : 
  (Real.sqrt 392 / Real.sqrt 336) + (Real.sqrt 200 / Real.sqrt 128) + 1 = 41 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1629_162904


namespace NUMINAMATH_CALUDE_annual_increase_correct_l1629_162935

/-- The annual percentage increase in population -/
def annual_increase : ℝ := 10

/-- The present population -/
def present_population : ℕ := 15000

/-- The population after 2 years -/
def future_population : ℕ := 18150

/-- Theorem stating that the annual percentage increase is correct -/
theorem annual_increase_correct : 
  (present_population : ℝ) * (1 + annual_increase / 100)^2 = future_population := by
  sorry

#check annual_increase_correct

end NUMINAMATH_CALUDE_annual_increase_correct_l1629_162935


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1629_162954

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1629_162954


namespace NUMINAMATH_CALUDE_min_distance_point_coordinates_l1629_162912

/-- Given two fixed points C(0,4) and K(6,0) in a Cartesian coordinate system,
    with A being a moving point on the line segment OK,
    D being the midpoint of AC,
    and B obtained by rotating AD clockwise 90° around A,
    prove that when BK reaches its minimum value,
    the coordinates of point B are (26/5, 8/5). -/
theorem min_distance_point_coordinates :
  ∀ (A : ℝ × ℝ) (B : ℝ × ℝ),
  let C : ℝ × ℝ := (0, 4)
  let K : ℝ × ℝ := (6, 0)
  let O : ℝ × ℝ := (0, 0)
  -- A is on line segment OK
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (t * K.1, 0)) →
  -- D is midpoint of AC
  let D : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  -- B is obtained by rotating AD 90° clockwise around A
  (B.1 = A.1 + (D.2 - A.2) ∧ B.2 = A.2 - (D.1 - A.1)) →
  -- When BK reaches its minimum value
  (∀ (A' : ℝ × ℝ) (B' : ℝ × ℝ),
    (∃ t' : ℝ, 0 ≤ t' ∧ t' ≤ 1 ∧ A' = (t' * K.1, 0)) →
    let D' : ℝ × ℝ := ((A'.1 + C.1) / 2, (A'.2 + C.2) / 2)
    (B'.1 = A'.1 + (D'.2 - A'.2) ∧ B'.2 = A'.2 - (D'.1 - A'.1)) →
    (B.1 - K.1)^2 + (B.2 - K.2)^2 ≤ (B'.1 - K.1)^2 + (B'.2 - K.2)^2) →
  -- Then the coordinates of B are (26/5, 8/5)
  B = (26/5, 8/5) := by sorry

end NUMINAMATH_CALUDE_min_distance_point_coordinates_l1629_162912


namespace NUMINAMATH_CALUDE_conference_handshakes_theorem_l1629_162962

/-- Represents the number of handshakes in a conference with specific group interactions -/
def conference_handshakes (total : ℕ) (group_a : ℕ) (group_b : ℕ) (group_c : ℕ) 
  (known_per_b : ℕ) : ℕ :=
  let handshakes_ab := group_b * (group_a - known_per_b)
  let handshakes_bc := group_b * group_c
  let handshakes_c := group_c * (group_c - 1) / 2
  let handshakes_ac := group_a * group_c
  handshakes_ab + handshakes_bc + handshakes_c + handshakes_ac

/-- Theorem stating that the number of handshakes in the given conference scenario is 535 -/
theorem conference_handshakes_theorem :
  conference_handshakes 50 30 15 5 10 = 535 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_theorem_l1629_162962


namespace NUMINAMATH_CALUDE_sticker_distribution_l1629_162906

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def stars_and_bars (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute stickers onto sheets --/
def distribute_stickers (total_stickers sheets min_per_sheet : ℕ) : ℕ :=
  stars_and_bars (total_stickers - sheets * min_per_sheet) sheets

theorem sticker_distribution :
  distribute_stickers 10 5 2 = 1 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1629_162906


namespace NUMINAMATH_CALUDE_correct_expression_l1629_162903

/-- A type representing mathematical expressions --/
inductive MathExpression
  | DivideABC : MathExpression
  | MixedFraction : MathExpression
  | MultiplyAB : MathExpression
  | ThreeM : MathExpression

/-- A predicate that determines if an expression is correctly written --/
def is_correctly_written (e : MathExpression) : Prop :=
  match e with
  | MathExpression.ThreeM => True
  | _ => False

/-- The set of given expressions --/
def expression_set : Set MathExpression :=
  {MathExpression.DivideABC, MathExpression.MixedFraction, 
   MathExpression.MultiplyAB, MathExpression.ThreeM}

theorem correct_expression :
  ∃ (e : MathExpression), e ∈ expression_set ∧ is_correctly_written e :=
by sorry

end NUMINAMATH_CALUDE_correct_expression_l1629_162903


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1629_162956

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 6 / Real.sqrt 8) =
  (3 * Real.sqrt 385) / 154 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1629_162956


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l1629_162928

theorem triangle_angle_not_all_greater_than_60 :
  ∀ (a b c : ℝ), 
  (a + b + c = 180) →  -- Sum of angles in a triangle is 180°
  (a > 0) → (b > 0) → (c > 0) →  -- All angles are positive
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l1629_162928


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1629_162939

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1 / x) + (9 / y) = 1) : 
  x + y ≥ 16 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 / x) + (9 / y) = 1 ∧ x + y = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1629_162939


namespace NUMINAMATH_CALUDE_coffee_price_proof_l1629_162947

/-- The regular price of coffee in dollars per pound -/
def regular_price : ℝ := 40

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.6

/-- The price of a discounted quarter-pound package with a free chocolate bar -/
def discounted_quarter_pound_price : ℝ := 4

theorem coffee_price_proof :
  regular_price * (1 - discount_rate) / 4 = discounted_quarter_pound_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_price_proof_l1629_162947


namespace NUMINAMATH_CALUDE_parabola_focus_l1629_162931

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * x^2 + 4 * x + 1

/-- The focus of a parabola -/
def focus (f : ℝ × ℝ) (p : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x y, p x y ↔ y = a * x^2 + b * x + c) ∧
    f = (- b / (2 * a), c - b^2 / (4 * a) - 1 / (4 * a))

/-- Theorem: The focus of the parabola y = -2x^2 + 4x + 1 is (1, 23/8) -/
theorem parabola_focus : focus (1, 23/8) parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1629_162931


namespace NUMINAMATH_CALUDE_division_remainder_l1629_162930

theorem division_remainder : ∃ q : ℤ, 1346584 = 137 * q + 5 ∧ 0 ≤ 5 ∧ 5 < 137 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1629_162930
