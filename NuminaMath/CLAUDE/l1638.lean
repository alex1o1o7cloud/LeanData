import Mathlib

namespace NUMINAMATH_CALUDE_sample_is_sixteen_l1638_163801

/-- Represents a stratified sampling scenario in a factory -/
structure StratifiedSampling where
  totalSample : ℕ
  totalProducts : ℕ
  workshopProducts : ℕ
  h_positive : 0 < totalSample ∧ 0 < totalProducts ∧ 0 < workshopProducts
  h_valid : workshopProducts ≤ totalProducts

/-- Calculates the number of items sampled from a specific workshop -/
def sampleFromWorkshop (s : StratifiedSampling) : ℕ :=
  (s.totalSample * s.workshopProducts) / s.totalProducts

/-- Theorem stating that for the given scenario, the sample from the workshop is 16 -/
theorem sample_is_sixteen (s : StratifiedSampling) 
  (h_total_sample : s.totalSample = 128)
  (h_total_products : s.totalProducts = 2048)
  (h_workshop_products : s.workshopProducts = 256) : 
  sampleFromWorkshop s = 16 := by
  sorry

#eval sampleFromWorkshop { 
  totalSample := 128, 
  totalProducts := 2048, 
  workshopProducts := 256, 
  h_positive := by norm_num, 
  h_valid := by norm_num 
}

end NUMINAMATH_CALUDE_sample_is_sixteen_l1638_163801


namespace NUMINAMATH_CALUDE_convex_ngon_divided_into_equal_triangles_l1638_163894

/-- A convex n-gon that is circumscribed and divided into equal triangles by non-intersecting diagonals -/
structure ConvexNGon (n : ℕ) :=
  (convex : Bool)
  (circumscribed : Bool)
  (equal_triangles : Bool)
  (non_intersecting_diagonals : Bool)

/-- Theorem stating that the only possible value for n is 4 -/
theorem convex_ngon_divided_into_equal_triangles
  (n : ℕ) (ngon : ConvexNGon n) (h1 : n > 3)
  (h2 : ngon.convex = true)
  (h3 : ngon.circumscribed = true)
  (h4 : ngon.equal_triangles = true)
  (h5 : ngon.non_intersecting_diagonals = true) :
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_divided_into_equal_triangles_l1638_163894


namespace NUMINAMATH_CALUDE_factorization_valid_l1638_163888

-- Define the left-hand side of the equation
def lhs (x : ℝ) : ℝ := -8 * x^2 + 8 * x - 2

-- Define the right-hand side of the equation
def rhs (x : ℝ) : ℝ := -2 * (2 * x - 1)^2

-- Theorem stating that the left-hand side equals the right-hand side for all real x
theorem factorization_valid (x : ℝ) : lhs x = rhs x := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l1638_163888


namespace NUMINAMATH_CALUDE_student_council_committees_l1638_163895

theorem student_council_committees (n : ℕ) (k : ℕ) (m : ℕ) (p : ℕ) (w : ℕ) :
  n = 15 →  -- Total number of student council members
  k = 3 →   -- Size of welcoming committee
  m = 4 →   -- Size of planning committee
  p = 2 →   -- Size of finance committee
  w = 20 →  -- Number of ways to select welcoming committee
  (n.choose m) * (k.choose p) = 4095 :=
by sorry

end NUMINAMATH_CALUDE_student_council_committees_l1638_163895


namespace NUMINAMATH_CALUDE_complement_M_wrt_U_l1638_163869

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_M_wrt_U : 
  {x ∈ U | x ∉ M} = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_M_wrt_U_l1638_163869


namespace NUMINAMATH_CALUDE_max_charge_at_150_l1638_163822

-- Define the charge function
def charge (x : ℝ) : ℝ := 1000 * x - 5 * (x - 100)^2

-- State the theorem
theorem max_charge_at_150 :
  ∀ x ∈ Set.Icc 100 180,
    charge x ≤ charge 150 ∧
    charge 150 = 112500 := by
  sorry

-- Note: Set.Icc 100 180 represents the closed interval [100, 180]

end NUMINAMATH_CALUDE_max_charge_at_150_l1638_163822


namespace NUMINAMATH_CALUDE_regular_100gon_rectangle_two_colors_l1638_163806

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of the vertices of a polygon -/
def Coloring (n : ℕ) (k : ℕ) := Fin n → Fin k

/-- Four vertices form a rectangle in a regular polygon -/
def IsRectangle (p : RegularPolygon 100) (v1 v2 v3 v4 : Fin 100) : Prop :=
  sorry

/-- The number of distinct colors used for given vertices -/
def NumColors (c : Coloring 100 10) (vs : List (Fin 100)) : ℕ :=
  sorry

theorem regular_100gon_rectangle_two_colors :
  ∀ (p : RegularPolygon 100) (c : Coloring 100 10),
  ∃ (v1 v2 v3 v4 : Fin 100),
    IsRectangle p v1 v2 v3 v4 ∧ NumColors c [v1, v2, v3, v4] ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_regular_100gon_rectangle_two_colors_l1638_163806


namespace NUMINAMATH_CALUDE_purple_valley_skirts_l1638_163883

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

/-- The number of skirts in Seafoam Valley -/
def seafoam_skirts : ℕ := (2 * azure_skirts) / 3

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := seafoam_skirts / 4

/-- Theorem stating that Purple Valley has 10 skirts -/
theorem purple_valley_skirts : purple_skirts = 10 := by
  sorry

end NUMINAMATH_CALUDE_purple_valley_skirts_l1638_163883


namespace NUMINAMATH_CALUDE_kennedy_school_distance_l1638_163870

/-- Represents the fuel efficiency of Kennedy's car in miles per gallon -/
def fuel_efficiency : ℝ := 19

/-- Represents the initial amount of gas in Kennedy's car in gallons -/
def initial_gas : ℝ := 2

/-- Represents the distance to the softball park in miles -/
def distance_softball : ℝ := 6

/-- Represents the distance to the burger restaurant in miles -/
def distance_burger : ℝ := 2

/-- Represents the distance to her friend's house in miles -/
def distance_friend : ℝ := 4

/-- Represents the distance home in miles -/
def distance_home : ℝ := 11

/-- Theorem stating that Kennedy drove 15 miles to school -/
theorem kennedy_school_distance : 
  ∃ (distance_school : ℝ), 
    distance_school = fuel_efficiency * initial_gas - 
      (distance_softball + distance_burger + distance_friend + distance_home) ∧ 
    distance_school = 15 := by
  sorry

end NUMINAMATH_CALUDE_kennedy_school_distance_l1638_163870


namespace NUMINAMATH_CALUDE_unfactorable_expression_difference_of_squares_factorization_common_factor_factorization_perfect_square_trinomial_factorization_l1638_163892

theorem unfactorable_expression (x : ℝ) : ¬∃ (a b : ℝ), x^2 + 9 = a * b ∧ (a ≠ 1 ∨ b ≠ x^2 + 9) ∧ (a ≠ x^2 + 9 ∨ b ≠ 1) := by
  sorry

-- Helper theorems to show that other expressions can be factored
theorem difference_of_squares_factorization (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

theorem common_factor_factorization (x : ℝ) : 9*x - 9 = 9 * (x - 1) := by
  sorry

theorem perfect_square_trinomial_factorization (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_expression_difference_of_squares_factorization_common_factor_factorization_perfect_square_trinomial_factorization_l1638_163892


namespace NUMINAMATH_CALUDE_correct_payment_to_C_l1638_163872

/-- The amount to be paid to worker C -/
def payment_to_C (a_rate b_rate : ℚ) (total_payment : ℕ) (days_to_complete : ℕ) : ℚ :=
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * days_to_complete
  let c_work := 1 - ab_work
  c_work * total_payment

/-- Theorem stating the correct payment to worker C -/
theorem correct_payment_to_C :
  payment_to_C (1/6) (1/8) 2400 3 = 300 := by sorry

end NUMINAMATH_CALUDE_correct_payment_to_C_l1638_163872


namespace NUMINAMATH_CALUDE_expression_evaluation_l1638_163840

theorem expression_evaluation : 3^(0^(2^11)) + ((3^0)^2)^11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1638_163840


namespace NUMINAMATH_CALUDE_find_y_value_l1638_163864

theorem find_y_value (x y : ℚ) (h1 : x = 51) (h2 : x^3*y - 2*x^2*y + x*y = 63000) : y = 8/17 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1638_163864


namespace NUMINAMATH_CALUDE_bananas_arrangements_l1638_163859

def word_length : ℕ := 7
def a_count : ℕ := 3
def n_count : ℕ := 2

theorem bananas_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial) = 420 := by
  sorry

end NUMINAMATH_CALUDE_bananas_arrangements_l1638_163859


namespace NUMINAMATH_CALUDE_barbaras_candy_count_l1638_163861

/-- Given Barbara's initial candy count and the number of candies she bought,
    prove that her total candy count is the sum of these two quantities. -/
theorem barbaras_candy_count (initial_candies bought_candies : ℕ) :
  initial_candies = 9 →
  bought_candies = 18 →
  initial_candies + bought_candies = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_barbaras_candy_count_l1638_163861


namespace NUMINAMATH_CALUDE_expression_evaluation_l1638_163802

theorem expression_evaluation :
  let f (x : ℝ) := -7*x + 2*(x^2 - 1) - (2*x^2 - x + 3)
  f 1 = -11 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1638_163802


namespace NUMINAMATH_CALUDE_equal_milk_water_ratio_l1638_163803

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The first mixture with milk:water ratio 5:4 -/
def mixture_p : Mixture := ⟨5, 4⟩

/-- The second mixture with milk:water ratio 2:7 -/
def mixture_q : Mixture := ⟨2, 7⟩

/-- Combines two mixtures in a given ratio -/
def combine_mixtures (m1 m2 : Mixture) (r1 r2 : ℚ) : Mixture :=
  ⟨r1 * m1.milk + r2 * m2.milk, r1 * m1.water + r2 * m2.water⟩

/-- Theorem stating that combining mixture_p and mixture_q in ratio 5:1 results in equal milk and water -/
theorem equal_milk_water_ratio :
  let result := combine_mixtures mixture_p mixture_q 5 1
  result.milk = result.water := by sorry

end NUMINAMATH_CALUDE_equal_milk_water_ratio_l1638_163803


namespace NUMINAMATH_CALUDE_not_divisible_sum_of_not_divisible_product_plus_one_l1638_163865

theorem not_divisible_sum_of_not_divisible_product_plus_one (n : ℕ) 
  (h : ∀ (a b : ℕ), ¬(n ∣ 2^a * 3^b + 1)) :
  ∀ (c d : ℕ), ¬(n ∣ 2^c + 3^d) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_sum_of_not_divisible_product_plus_one_l1638_163865


namespace NUMINAMATH_CALUDE_max_value_theorem_l1638_163887

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  ∀ (a b : ℝ), 4 * a + 3 * b ≤ 10 → 3 * a + 5 * b ≤ 12 → 2 * a + b ≤ 46 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1638_163887


namespace NUMINAMATH_CALUDE_log_problem_l1638_163899

theorem log_problem (y : ℝ) (h : y = (Real.log 16 / Real.log 4) ^ (Real.log 4 / Real.log 16)) :
  Real.log y / Real.log 12 = 1 / (4 + 2 * Real.log 3 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1638_163899


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1638_163820

/-- The area of the union of a square with side length 12 and a circle with radius 6 
    centered at the center of the square is equal to 144. -/
theorem square_circle_union_area : 
  let square_side : ℝ := 12
  let circle_radius : ℝ := 6
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  square_area = circle_area + 144 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1638_163820


namespace NUMINAMATH_CALUDE_smallest_product_sum_l1638_163837

def digits : List Nat := [3, 4, 5, 6, 7]

def is_valid_configuration (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product_sum (a b c d e : Nat) : Nat :=
  (10 * a + b) * (10 * c + d) + e * (10 * a + b)

theorem smallest_product_sum :
  ∀ a b c d e : Nat,
    is_valid_configuration a b c d e →
    product_sum a b c d e ≥ 2448 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_sum_l1638_163837


namespace NUMINAMATH_CALUDE_square_roots_ratio_l1638_163852

theorem square_roots_ratio (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, y^2 - 5*y + 2 = 0 ∧ x = y^2) →
  c / b = -4 / 21 := by
sorry

end NUMINAMATH_CALUDE_square_roots_ratio_l1638_163852


namespace NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l1638_163827

/-- Calculates the percentage reduction in alcohol concentration when water is added to an alcohol solution. -/
theorem alcohol_concentration_reduction 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := initial_alcohol / final_volume
  let reduction := (initial_concentration - final_concentration) / initial_concentration * 100
  by
    -- Proof goes here
    sorry

/-- The specific case of adding 26 liters of water to 14 liters of 20% alcohol solution results in a 65% reduction in concentration. -/
theorem specific_alcohol_reduction : 
  alcohol_concentration_reduction 14 0.20 26 = 65 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l1638_163827


namespace NUMINAMATH_CALUDE_special_function_properties_l1638_163828

/-- A function satisfying the given functional equation -/
structure SpecialFunction where
  f : ℝ → ℝ
  eq : ∀ x y, f (x + y) * f (x - y) = f x + f y
  nonzero : f 0 ≠ 0

/-- Properties of the special function -/
theorem special_function_properties (F : SpecialFunction) :
  (F.f 0 = 2) ∧
  (∀ x, F.f x = F.f (-x)) ∧
  (∀ x, F.f (2 * x) = F.f x) := by
  sorry


end NUMINAMATH_CALUDE_special_function_properties_l1638_163828


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1638_163838

theorem expand_and_simplify (x y : ℝ) : 
  (2*x + 3*y)^2 - 2*x*(2*x - 3*y) = 18*x*y + 9*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1638_163838


namespace NUMINAMATH_CALUDE_f_monotone_intervals_f_B_range_l1638_163821

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem f_monotone_intervals (k : ℤ) :
  is_monotone_increasing f (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) := by sorry

theorem f_B_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  b * Real.cos (2 * A) = b * Real.cos A - a * Real.sin B →
  ∃ x, f B = x ∧ -Real.sqrt 2 / 2 ≤ x ∧ x ≤ Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_f_monotone_intervals_f_B_range_l1638_163821


namespace NUMINAMATH_CALUDE_min_box_value_l1638_163893

theorem min_box_value (a b : ℤ) (box : ℤ) :
  (∀ x : ℝ, (a * x + b) * (b * x + a) = 30 * x^2 + box * x + 30) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ (min_box : ℤ), (min_box = 61 ∧ box ≥ min_box) := by
  sorry

end NUMINAMATH_CALUDE_min_box_value_l1638_163893


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_and_terminating_l1638_163814

/-- A function that checks if a number contains the digit 7 -/
def contains_seven (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < n ∧ n % 10^(d+1) / 10^d = 7

/-- A function that checks if a fraction 1/n is a terminating decimal -/
def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

/-- Theorem stating that 128 is the smallest positive integer n such that
    1/n is a terminating decimal and n contains the digit 7 -/
theorem smallest_n_with_seven_and_terminating :
  ∀ n : ℕ, n > 0 → is_terminating_decimal n → contains_seven n → n ≥ 128 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_and_terminating_l1638_163814


namespace NUMINAMATH_CALUDE_strawberry_vs_cabbage_l1638_163898

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Represents the result of cutting an isosceles right triangle -/
structure CutTriangle where
  original : Triangle
  cut1 : ℝ  -- Position of first cut (0 ≤ cut1 ≤ 1)
  cut2 : ℝ  -- Position of second cut (0 ≤ cut2 ≤ 1)

/-- Calculates the area of the rectangle formed by the cuts -/
def rectangleArea (ct : CutTriangle) : ℝ := sorry

/-- Calculates the sum of areas of the two smaller triangles formed by the cuts -/
def smallTrianglesArea (ct : CutTriangle) : ℝ := sorry

/-- Theorem: The area of the rectangle is always less than or equal to 
    the sum of the areas of the two smaller triangles -/
theorem strawberry_vs_cabbage (ct : CutTriangle) : 
  rectangleArea ct ≤ smallTrianglesArea ct := by
  sorry

end NUMINAMATH_CALUDE_strawberry_vs_cabbage_l1638_163898


namespace NUMINAMATH_CALUDE_log_expression_equality_l1638_163875

theorem log_expression_equality : 
  2 * Real.log 10 / Real.log 5 + Real.log (1/4) / Real.log 5 + (2 : ℝ) ^ (Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1638_163875


namespace NUMINAMATH_CALUDE_overtime_calculation_l1638_163823

/-- A worker's pay structure and hours worked --/
structure WorkerPay where
  ordinary_rate : ℚ  -- Rate for ordinary time in cents per hour
  overtime_rate : ℚ  -- Rate for overtime in cents per hour
  total_pay : ℚ      -- Total pay for the week in cents
  total_hours : ℕ    -- Total hours worked in the week

/-- Calculate the number of overtime hours --/
def overtime_hours (w : WorkerPay) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the overtime hours are 8 --/
theorem overtime_calculation (w : WorkerPay) 
  (h1 : w.ordinary_rate = 60)
  (h2 : w.overtime_rate = 90)
  (h3 : w.total_pay = 3240)
  (h4 : w.total_hours = 50) :
  overtime_hours w = 8 := by
  sorry

end NUMINAMATH_CALUDE_overtime_calculation_l1638_163823


namespace NUMINAMATH_CALUDE_sofia_shopping_cost_l1638_163844

def shirt_cost : ℕ := 7
def shoes_cost : ℕ := shirt_cost + 3
def total_shirts_shoes : ℕ := 2 * shirt_cost + shoes_cost
def bag_cost : ℕ := total_shirts_shoes / 2
def total_cost : ℕ := 2 * shirt_cost + shoes_cost + bag_cost

theorem sofia_shopping_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_sofia_shopping_cost_l1638_163844


namespace NUMINAMATH_CALUDE_negation_of_statement_l1638_163839

theorem negation_of_statement :
  ¬(∀ x : ℝ, x ≠ 0 → x^2 - 4 > 0) ↔ 
  (∃ x : ℝ, x ≠ 0 ∧ x^2 - 4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_statement_l1638_163839


namespace NUMINAMATH_CALUDE_ellipse_properties_l1638_163809

/-- Definition of an ellipse passing through a point with given foci -/
def is_ellipse_through_point (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f2.1 - f1.1)^2 + (f2.2 - f1.2)^2) / 2
  ∃ a : ℝ, a > c ∧ d1 + d2 = 2 * a

/-- The equation of an ellipse in standard form -/
def ellipse_equation (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  let f1 : ℝ × ℝ := (0, 0)
  let f2 : ℝ × ℝ := (0, 8)
  let p : ℝ × ℝ := (7, 4)
  let a : ℝ := 8 * Real.sqrt 2
  let b : ℝ := 8 * Real.sqrt 7
  let h : ℝ := 0
  let k : ℝ := 4
  is_ellipse_through_point f1 f2 p →
  (∀ x y : ℝ, ellipse_equation a b h k x y ↔ 
    ((x - 0)^2 / (8 * Real.sqrt 2)^2 + (y - 4)^2 / (8 * Real.sqrt 7)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1638_163809


namespace NUMINAMATH_CALUDE_most_accurate_reading_l1638_163831

def scale_reading : ℝ → Prop :=
  λ x => 3.25 < x ∧ x < 3.5

def closer_to_3_3 (x : ℝ) : Prop :=
  |x - 3.3| < |x - 3.375|

def options : Set ℝ :=
  {3.05, 3.15, 3.25, 3.3, 3.6}

theorem most_accurate_reading (x : ℝ) 
  (h1 : scale_reading x) 
  (h2 : closer_to_3_3 x) : 
  ∀ y ∈ options, |x - 3.3| ≤ |x - y| :=
by sorry

end NUMINAMATH_CALUDE_most_accurate_reading_l1638_163831


namespace NUMINAMATH_CALUDE_goose_egg_count_l1638_163891

theorem goose_egg_count (
  hatch_rate : ℚ)
  (first_month_survival : ℚ)
  (next_three_months_survival : ℚ)
  (following_six_months_survival : ℚ)
  (first_half_second_year_survival : ℚ)
  (second_year_survival : ℚ)
  (final_survivors : ℕ)
  (h1 : hatch_rate = 4 / 7)
  (h2 : first_month_survival = 3 / 5)
  (h3 : next_three_months_survival = 7 / 10)
  (h4 : following_six_months_survival = 5 / 8)
  (h5 : first_half_second_year_survival = 2 / 3)
  (h6 : second_year_survival = 4 / 5)
  (h7 : final_survivors = 200) :
  ∃ (original_eggs : ℕ), original_eggs = 2503 ∧
  (↑final_survivors : ℚ) = ↑original_eggs * hatch_rate * first_month_survival *
    next_three_months_survival * following_six_months_survival *
    first_half_second_year_survival * second_year_survival :=
by sorry

end NUMINAMATH_CALUDE_goose_egg_count_l1638_163891


namespace NUMINAMATH_CALUDE_ant_journey_l1638_163826

-- Define the plane and points A and B
variable (Plane : Type) (A B : Plane)

-- Define the distance functions from A and B
variable (distA distB : ℝ → ℝ)

-- Define the conditions
variable (h1 : distA 7 = 5)
variable (h2 : distB 7 = 3)
variable (h3 : distB 0 = 0)
variable (h4 : distA 0 = 4)

-- Define the distance between A and B
def dist_AB : ℝ := 4

-- Define the theorem
theorem ant_journey :
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 0 ≤ t1 ∧ t1 ≤ 9 ∧ 0 ≤ t2 ∧ t2 ≤ 9 ∧ distA t1 = distB t1 ∧ distA t2 = distB t2) ∧
  (dist_AB = 4) ∧
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    0 ≤ t1 ∧ t1 ≤ 9 ∧ 0 ≤ t2 ∧ t2 ≤ 9 ∧ 0 ≤ t3 ∧ t3 ≤ 9 ∧
    distA t1 + distB t1 = dist_AB ∧
    distA t2 + distB t2 = dist_AB ∧
    distA t3 + distB t3 = dist_AB) ∧
  (∃ d : ℝ, d = 8 ∧ 
    d = |distA 3 - distA 0| + |distA 5 - distA 3| + |distA 7 - distA 5| + |distA 9 - distA 7|) :=
by sorry

end NUMINAMATH_CALUDE_ant_journey_l1638_163826


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1638_163804

/-- Given two digits A and B in base d > 7 such that AB + AA = 174 in base d,
    prove that A - B = 3 in base d, assuming A > B. -/
theorem digit_difference_in_base_d (d A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  A > B →
  (A * d + B) + (A * d + A) = 1 * d * d + 7 * d + 4 →
  A - B = 3 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1638_163804


namespace NUMINAMATH_CALUDE_prob_difference_games_l1638_163851

/-- Probability of getting heads on a single toss of the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails on a single toss of the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game A -/
def p_win_game_a : ℚ := p_heads^4 + p_tails^4

/-- Probability of winning Game C -/
def p_win_game_c : ℚ := p_heads^5 + p_tails^5 + p_heads^3 * p_tails^2 + p_tails^3 * p_heads^2

/-- The difference in probabilities between winning Game A and Game C -/
theorem prob_difference_games : p_win_game_a - p_win_game_c = 3/64 := by sorry

end NUMINAMATH_CALUDE_prob_difference_games_l1638_163851


namespace NUMINAMATH_CALUDE_select_teachers_eq_140_l1638_163886

/-- The number of ways to select 6 out of 10 teachers, where two specific teachers cannot be selected together -/
def select_teachers : ℕ :=
  let total_teachers : ℕ := 10
  let teachers_to_invite : ℕ := 6
  let remaining_teachers : ℕ := 8  -- Excluding A and B
  let case1 : ℕ := 2 * Nat.choose remaining_teachers (teachers_to_invite - 1)
  let case2 : ℕ := Nat.choose remaining_teachers teachers_to_invite
  case1 + case2

theorem select_teachers_eq_140 : select_teachers = 140 := by
  sorry

end NUMINAMATH_CALUDE_select_teachers_eq_140_l1638_163886


namespace NUMINAMATH_CALUDE_mean_median_sum_l1638_163843

theorem mean_median_sum (m n : ℕ) (h1 : m + 8 < n) 
  (h2 : (m + (m + 3) + (m + 8) + n + (n + 3) + (2 * n - 1)) / 6 = n + 1)
  (h3 : (m + 8 + n) / 2 = n + 1) : m + n = 16 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_sum_l1638_163843


namespace NUMINAMATH_CALUDE_intersection_M_N_l1638_163819

def M : Set ℝ := {y | ∃ x, y = x^2}

def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

theorem intersection_M_N :
  (M.prod Set.univ) ∩ N = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ Real.sqrt 2 ∧ p.2 = p.1^2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1638_163819


namespace NUMINAMATH_CALUDE_square_of_binomial_p_l1638_163880

/-- If 9x^2 + 24x + p is the square of a binomial, then p = 16 -/
theorem square_of_binomial_p (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 + 24*x + p = (a*x + b)^2) → p = 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_p_l1638_163880


namespace NUMINAMATH_CALUDE_book_collection_problem_l1638_163878

/-- The number of books in either Jessica's or Tina's collection, but not both -/
def unique_books (shared : ℕ) (jessica_total : ℕ) (tina_unique : ℕ) : ℕ :=
  (jessica_total - shared) + tina_unique

theorem book_collection_problem :
  unique_books 12 22 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_book_collection_problem_l1638_163878


namespace NUMINAMATH_CALUDE_zongzi_sales_theorem_l1638_163850

/-- Represents the sales and profit model for zongzi boxes -/
structure ZongziSales where
  cost : ℝ             -- Cost per box
  min_price : ℝ        -- Minimum selling price
  base_sales : ℝ       -- Base sales at minimum price
  price_sensitivity : ℝ -- Decrease in sales per unit price increase
  max_price : ℝ        -- Maximum allowed selling price
  min_profit : ℝ       -- Minimum desired daily profit

/-- The main theorem about zongzi sales and profit -/
theorem zongzi_sales_theorem (z : ZongziSales)
  (h_cost : z.cost = 40)
  (h_min_price : z.min_price = 45)
  (h_base_sales : z.base_sales = 700)
  (h_price_sensitivity : z.price_sensitivity = 20)
  (h_max_price : z.max_price = 58)
  (h_min_profit : z.min_profit = 6000) :
  (∃ (sales_eq : ℝ → ℝ),
    (∀ x, sales_eq x = -20 * x + 1600) ∧
    (∃ (optimal_price : ℝ) (max_profit : ℝ),
      optimal_price = 60 ∧
      max_profit = 8000 ∧
      (∀ p, z.min_price ≤ p → p ≤ z.max_price →
        (p - z.cost) * (sales_eq p) ≤ max_profit)) ∧
    (∃ (min_boxes : ℝ),
      min_boxes = 440 ∧
      (z.max_price - z.cost) * min_boxes ≥ z.min_profit)) :=
sorry

end NUMINAMATH_CALUDE_zongzi_sales_theorem_l1638_163850


namespace NUMINAMATH_CALUDE_first_number_proof_l1638_163871

theorem first_number_proof (x : ℕ) : 
  (∃ k : ℕ, x = 144 * k + 23) ∧ 
  (∃ m : ℕ, 7373 = 144 * m + 29) ∧
  (∀ d : ℕ, d > 144 → ¬(∃ r₁ r₂ : ℕ, x = d * k + r₁ ∧ 7373 = d * m + r₂)) →
  x = 7361 :=
by sorry

end NUMINAMATH_CALUDE_first_number_proof_l1638_163871


namespace NUMINAMATH_CALUDE_flower_bed_distance_l1638_163849

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- The total distance walked around a rectangle multiple times -/
def total_distance (length width : ℝ) (times : ℕ) : ℝ :=
  (rectangle_perimeter length width) * times

theorem flower_bed_distance :
  total_distance 5 3 3 = 30 := by sorry

end NUMINAMATH_CALUDE_flower_bed_distance_l1638_163849


namespace NUMINAMATH_CALUDE_cosine_function_parameters_l1638_163856

/-- Proves that for y = a cos(bx), if max is 3 at x=0 and first zero at x=π/6, then a=3 and b=3 -/
theorem cosine_function_parameters (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, a * Real.cos (b * x) ≤ 3) → 
  a * Real.cos 0 = 3 → 
  a * Real.cos (b * (π / 6)) = 0 → 
  a = 3 ∧ b = 3 := by
  sorry


end NUMINAMATH_CALUDE_cosine_function_parameters_l1638_163856


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_l1638_163813

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if four consecutive natural numbers are all prime -/
def fourConsecutivePrimes (n : ℕ) : Prop :=
  isPrime n ∧ isPrime (n + 1) ∧ isPrime (n + 2) ∧ isPrime (n + 3)

/-- The sum of four consecutive natural numbers starting from n -/
def sumFourConsecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3)

/-- The theorem stating that the smallest sum of four consecutive positive prime numbers
    that is divisible by 3 is 36 -/
theorem smallest_sum_four_consecutive_primes :
  ∃ n : ℕ, fourConsecutivePrimes n ∧ sumFourConsecutive n % 3 = 0 ∧
  sumFourConsecutive n = 36 ∧
  ∀ m : ℕ, m < n → ¬(fourConsecutivePrimes m ∧ sumFourConsecutive m % 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_l1638_163813


namespace NUMINAMATH_CALUDE_product_of_ratios_l1638_163812

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2007 ∧ y₁^3 - 3*x₁^2*y₁ = 2006)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2007 ∧ y₂^3 - 3*x₂^2*y₂ = 2006)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2007 ∧ y₃^3 - 3*x₃^2*y₃ = 2006)
  (h₄ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 1 / 1003 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l1638_163812


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1638_163874

-- Problem 1
theorem problem_1 : Real.sqrt 48 / Real.sqrt 3 * (1/4) = 1 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 12 - Real.sqrt 3 + Real.sqrt (1/3) = (4 * Real.sqrt 3) / 3 := by sorry

-- Problem 3
theorem problem_3 : (2 + Real.sqrt 3) * (2 - Real.sqrt 3) + Real.sqrt 3 * (2 - Real.sqrt 3) = 2 * Real.sqrt 3 - 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1638_163874


namespace NUMINAMATH_CALUDE_game_winning_strategy_l1638_163800

/-- Represents the players in the game -/
inductive Player
| A
| B

/-- Represents the result of the game -/
inductive GameResult
| AWins
| BWins

/-- Represents the game state -/
structure GameState where
  n : ℕ
  k : ℕ
  grid : Fin n → Fin n → Bool
  currentPlayer : Player

/-- Defines the winning strategy for the game -/
def winningStrategy (n k : ℕ) : GameResult :=
  if n ≤ 2 * k - 1 then
    GameResult.AWins
  else if n % 2 = 1 then
    GameResult.AWins
  else
    GameResult.BWins

/-- The main theorem stating the winning strategy for the game -/
theorem game_winning_strategy (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  (winningStrategy n k = GameResult.AWins ∧ 
   (n ≤ 2 * k - 1 ∨ (n ≥ 2 * k ∧ n % 2 = 1))) ∨
  (winningStrategy n k = GameResult.BWins ∧ 
   n ≥ 2 * k ∧ n % 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_game_winning_strategy_l1638_163800


namespace NUMINAMATH_CALUDE_linear_function_straight_line_l1638_163860

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the property of having a straight line graph
def HasStraightLineGraph (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x y : ℝ, f y - f x = a * (y - x)

-- Define our specific function
def f (x : ℝ) : ℝ := 2 * x + 5

-- State the theorem
theorem linear_function_straight_line :
  (∀ g : ℝ → ℝ, LinearFunction g → HasStraightLineGraph g) →
  LinearFunction f →
  HasStraightLineGraph f :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_straight_line_l1638_163860


namespace NUMINAMATH_CALUDE_power_of_negative_product_l1638_163881

theorem power_of_negative_product (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l1638_163881


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l1638_163890

theorem parabola_tangent_hyperbola (m : ℝ) : 
  (∃ x y : ℝ, y = x^2 + 5 ∧ y^2 - m*x^2 = 4 ∧ 
   ∀ x' y' : ℝ, y' = x'^2 + 5 → y'^2 - m*x'^2 ≥ 4) →
  (m = 10 + 2*Real.sqrt 21 ∨ m = 10 - 2*Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l1638_163890


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1638_163842

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, y - 1]
  (∃ (k : ℝ), a = k • b) →
  (3 / x + 2 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1638_163842


namespace NUMINAMATH_CALUDE_min_value_theorem_l1638_163817

def f (x : ℝ) := |2*x + 1| + |x - 1|

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (1/2)*a + b + 2*c = 3/2) : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (a^2 + b^2 + c^2 ≥ 3/7) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1638_163817


namespace NUMINAMATH_CALUDE_min_moves_for_target_vectors_l1638_163815

/-- A tuple of 31 integers -/
def Tuple31 := Fin 31 → ℤ

/-- The set of standard basis vectors -/
def StandardBasis : Set Tuple31 :=
  {v | ∃ i, ∀ j, v j = if i = j then 1 else 0}

/-- The set of target vectors -/
def TargetVectors : Set Tuple31 :=
  {v | ∀ i, v i = if i = 0 then 0 else 1} ∪
  {v | ∀ i, v i = if i = 1 then 0 else 1} ∪
  {v | ∀ i, v i = if i = 30 then 0 else 1}

/-- The operation of adding two vectors -/
def AddVectors (v w : Tuple31) : Tuple31 :=
  λ i => v i + w i

/-- The set of vectors that can be generated in n moves -/
def GeneratedVectors (n : ℕ) : Set Tuple31 :=
  sorry

/-- The theorem statement -/
theorem min_moves_for_target_vectors :
  (∃ n, TargetVectors ⊆ GeneratedVectors n) ∧
  (∀ m, m < 87 → ¬(TargetVectors ⊆ GeneratedVectors m)) :=
sorry

end NUMINAMATH_CALUDE_min_moves_for_target_vectors_l1638_163815


namespace NUMINAMATH_CALUDE_intersection_M_N_l1638_163863

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1638_163863


namespace NUMINAMATH_CALUDE_product_mod_thirteen_l1638_163868

theorem product_mod_thirteen : (1501 * 1502 * 1503 * 1504 * 1505) % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_thirteen_l1638_163868


namespace NUMINAMATH_CALUDE_square_sum_pairs_l1638_163862

theorem square_sum_pairs : 
  {(a, b) : ℕ × ℕ | ∃ (m n : ℕ), a^2 + 3*b = m^2 ∧ b^2 + 3*a = n^2} = 
  {(1, 1), (11, 11), (16, 11)} := by
sorry

end NUMINAMATH_CALUDE_square_sum_pairs_l1638_163862


namespace NUMINAMATH_CALUDE_lcm_48_75_l1638_163833

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_75_l1638_163833


namespace NUMINAMATH_CALUDE_cubic_function_root_product_l1638_163835

/-- A cubic function with specific properties -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  x₁ : ℝ
  x₂ : ℝ
  root_zero : d = 0
  root_x₁ : a * x₁^3 + b * x₁^2 + c * x₁ + d = 0
  root_x₂ : a * x₂^3 + b * x₂^2 + c * x₂ + d = 0
  extreme_value_1 : (3 * a * ((3 - Real.sqrt 3) / 3)^2 + 2 * b * ((3 - Real.sqrt 3) / 3) + c) = 0
  extreme_value_2 : (3 * a * ((3 + Real.sqrt 3) / 3)^2 + 2 * b * ((3 + Real.sqrt 3) / 3) + c) = 0
  a_nonzero : a ≠ 0

/-- The product of non-zero roots of the cubic function is 2 -/
theorem cubic_function_root_product (f : CubicFunction) : f.x₁ * f.x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_root_product_l1638_163835


namespace NUMINAMATH_CALUDE_power_calculation_l1638_163846

theorem power_calculation : 2^345 - 8^3 / 8^2 + 3^2 = 2^345 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1638_163846


namespace NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l1638_163832

theorem base_10_to_base_8_conversion : 
  (2 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0 : ℕ) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l1638_163832


namespace NUMINAMATH_CALUDE_refrigerator_installment_l1638_163885

/-- Calculates the monthly installment amount for a purchase --/
def monthly_installment (cash_price deposit num_installments cash_savings : ℕ) : ℕ :=
  ((cash_price + cash_savings - deposit) / num_installments)

theorem refrigerator_installment :
  monthly_installment 8000 3000 30 4000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_installment_l1638_163885


namespace NUMINAMATH_CALUDE_raul_money_left_l1638_163848

/-- Calculates the money left after buying comics -/
def money_left (initial_money : ℕ) (num_comics : ℕ) (cost_per_comic : ℕ) : ℕ :=
  initial_money - (num_comics * cost_per_comic)

/-- Proves that Raul's remaining money is correct -/
theorem raul_money_left :
  money_left 87 8 4 = 55 := by
  sorry

end NUMINAMATH_CALUDE_raul_money_left_l1638_163848


namespace NUMINAMATH_CALUDE_abigail_saving_period_l1638_163841

def saving_period (monthly_saving : ℕ) (total_saved : ℕ) : ℕ :=
  total_saved / monthly_saving

theorem abigail_saving_period :
  let monthly_saving : ℕ := 4000
  let total_saved : ℕ := 48000
  saving_period monthly_saving total_saved = 12 := by
  sorry

end NUMINAMATH_CALUDE_abigail_saving_period_l1638_163841


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l1638_163825

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 collinear dots from horizontal or vertical lines -/
def horizontalVerticalSets : ℕ := 2 * gridSize

/-- The number of ways to choose 4 collinear dots from diagonal lines -/
def diagonalSets : ℕ := 2 * (Nat.choose gridSize chosenDots)

/-- The total number of ways to choose 4 collinear dots -/
def totalCollinearSets : ℕ := horizontalVerticalSets + diagonalSets

/-- Theorem: The probability of selecting 4 collinear dots from a 5x5 grid 
    when choosing 4 dots at random is 4/2530 -/
theorem collinear_dots_probability : 
  (totalCollinearSets : ℚ) / (Nat.choose totalDots chosenDots) = 4 / 2530 := by
  sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l1638_163825


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l1638_163896

/-- Given a complex number z = (1 + 2i^3) / (2 + i), prove that its coordinates in the complex plane are (0, -1) -/
theorem complex_number_coordinates :
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 2 * i^3) / (2 + i)
  z.re = 0 ∧ z.im = -1 := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l1638_163896


namespace NUMINAMATH_CALUDE_remainder_two_power_33_minus_one_mod_9_l1638_163845

theorem remainder_two_power_33_minus_one_mod_9 : 2^33 - 1 ≡ 7 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_33_minus_one_mod_9_l1638_163845


namespace NUMINAMATH_CALUDE_average_monthly_balance_l1638_163811

def monthly_balances : List ℝ := [200, 300, 250, 350, 300]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 280 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l1638_163811


namespace NUMINAMATH_CALUDE_outermost_to_innermost_ratio_l1638_163810

/-- A sequence of alternating inscribed squares and circles -/
structure SquareCircleSequence where
  S1 : Real  -- Side length of innermost square
  C1 : Real  -- Diameter of circle inscribing S1
  S2 : Real  -- Side length of square inscribing C1
  C2 : Real  -- Diameter of circle inscribing S2
  S3 : Real  -- Side length of square inscribing C2
  C3 : Real  -- Diameter of circle inscribing S3
  S4 : Real  -- Side length of outermost square

/-- Properties of the SquareCircleSequence -/
axiom sequence_properties (seq : SquareCircleSequence) :
  seq.C1 = seq.S1 * Real.sqrt 2 ∧
  seq.S2 = seq.C1 ∧
  seq.C2 = seq.S2 * Real.sqrt 2 ∧
  seq.S3 = seq.C2 ∧
  seq.C3 = seq.S3 * Real.sqrt 2 ∧
  seq.S4 = seq.C3

/-- The ratio of the outermost square's side length to the innermost square's side length is 2√2 -/
theorem outermost_to_innermost_ratio (seq : SquareCircleSequence) :
  seq.S4 / seq.S1 = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_outermost_to_innermost_ratio_l1638_163810


namespace NUMINAMATH_CALUDE_union_when_m_4_intersection_condition_l1638_163824

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part (1)
theorem union_when_m_4 : A ∪ B 4 = {x | -2 ≤ x ∧ x ≤ 7} := by sorry

-- Theorem for part (2)
theorem intersection_condition (m : ℝ) : B m ∩ A = B m ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_when_m_4_intersection_condition_l1638_163824


namespace NUMINAMATH_CALUDE_vasya_promotion_higher_revenue_l1638_163889

/-- Represents the revenue from candy box sales under different promotions -/
def candy_revenue (normal_revenue : ℝ) : Prop :=
  let vasya_revenue := normal_revenue * 2 * 0.8
  let kolya_revenue := normal_revenue * (8/3)
  (vasya_revenue = 16000) ∧ 
  (kolya_revenue = 13333.33333333333) ∧ 
  (vasya_revenue - normal_revenue = 6000)

/-- Theorem stating that Vasya's promotion leads to higher revenue -/
theorem vasya_promotion_higher_revenue :
  candy_revenue 10000 :=
sorry

end NUMINAMATH_CALUDE_vasya_promotion_higher_revenue_l1638_163889


namespace NUMINAMATH_CALUDE_ratio_to_eight_l1638_163847

theorem ratio_to_eight : ∃ x : ℚ, (5 : ℚ) / 1 = x / 8 ∧ x = 40 := by sorry

end NUMINAMATH_CALUDE_ratio_to_eight_l1638_163847


namespace NUMINAMATH_CALUDE_total_questions_answered_l1638_163876

/-- Represents a tour group with the number of tourists asking different amounts of questions -/
structure TourGroup where
  usual : ℕ  -- number of tourists asking the usual 2 questions
  zero : ℕ   -- number of tourists asking 0 questions
  one : ℕ    -- number of tourists asking 1 question
  three : ℕ  -- number of tourists asking 3 questions
  five : ℕ   -- number of tourists asking 5 questions
  double : ℕ -- number of tourists asking double the usual (4 questions)
  triple : ℕ -- number of tourists asking triple the usual (6 questions)
  quad : ℕ   -- number of tourists asking quadruple the usual (8 questions)

/-- Calculates the total number of questions for a tour group -/
def questionsForGroup (g : TourGroup) : ℕ :=
  2 * g.usual + 0 * g.zero + 1 * g.one + 3 * g.three + 5 * g.five +
  4 * g.double + 6 * g.triple + 8 * g.quad

/-- The six tour groups as described in the problem -/
def tourGroups : List TourGroup := [
  ⟨3, 0, 2, 0, 1, 0, 0, 0⟩,  -- Group A
  ⟨4, 1, 0, 6, 0, 0, 0, 0⟩,  -- Group B
  ⟨4, 2, 1, 0, 0, 0, 1, 0⟩,  -- Group C
  ⟨3, 1, 0, 0, 0, 0, 0, 1⟩,  -- Group D
  ⟨3, 2, 0, 0, 1, 3, 0, 0⟩,  -- Group E
  ⟨4, 1, 0, 2, 0, 0, 0, 0⟩   -- Group F
]

theorem total_questions_answered (groups := tourGroups) :
  (groups.map questionsForGroup).sum = 105 := by sorry

end NUMINAMATH_CALUDE_total_questions_answered_l1638_163876


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1638_163897

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (1 - m) * x₁^2 + 2 * x₁ - 2 = 0 ∧ 
   (1 - m) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (m < 3/2 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1638_163897


namespace NUMINAMATH_CALUDE_milk_dilution_l1638_163836

theorem milk_dilution (initial_volume : ℝ) (initial_milk_percentage : ℝ) (water_added : ℝ) :
  initial_volume = 60 →
  initial_milk_percentage = 0.84 →
  water_added = 18.75 →
  let initial_milk_volume := initial_volume * initial_milk_percentage
  let final_volume := initial_volume + water_added
  let final_milk_percentage := initial_milk_volume / final_volume
  final_milk_percentage = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l1638_163836


namespace NUMINAMATH_CALUDE_cube_edge_sum_l1638_163866

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The length of one edge of the cube in centimeters -/
def edge_length : ℝ := 15

/-- The sum of the lengths of all edges of the cube in centimeters -/
def sum_of_edges : ℝ := cube_edges * edge_length

theorem cube_edge_sum :
  sum_of_edges = 180 := by sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l1638_163866


namespace NUMINAMATH_CALUDE_tetrahedron_altitude_impossibility_l1638_163877

theorem tetrahedron_altitude_impossibility : ∀ (S₁ S₂ S₃ S₄ : ℝ),
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 →
  ¬ ∃ (h₁ h₂ h₃ h₄ : ℝ),
    h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ h₄ > 0 ∧
    (S₁ * h₁ = S₂ * h₂) ∧ (S₁ * h₁ = S₃ * h₃) ∧ (S₁ * h₁ = S₄ * h₄) ∧
    h₁ = 4 ∧ h₂ = 25 * Real.sqrt 3 / 3 ∧ h₃ = 25 * Real.sqrt 3 / 3 ∧ h₄ = 25 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_tetrahedron_altitude_impossibility_l1638_163877


namespace NUMINAMATH_CALUDE_two_noncongruent_triangles_l1638_163882

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∨
  (t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∨
  (t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

/-- The set of all triangles with integer side lengths and perimeter 9 -/
def triangles_with_perimeter_9 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 9}

/-- There are exactly 2 non-congruent triangles with integer side lengths and perimeter 9 -/
theorem two_noncongruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ triangles_with_perimeter_9 ∧
    t2 ∈ triangles_with_perimeter_9 ∧
    ¬congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_9 →
      (congruent t t1 ∨ congruent t t2) :=
sorry

end NUMINAMATH_CALUDE_two_noncongruent_triangles_l1638_163882


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1638_163807

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1638_163807


namespace NUMINAMATH_CALUDE_decaf_coffee_percentage_l1638_163853

/-- Proves that the percentage of decaffeinated coffee in the second batch is 70% --/
theorem decaf_coffee_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (second_batch : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : second_batch = 100)
  (h4 : final_decaf_percent = 30)
  (h5 : initial_stock * initial_decaf_percent / 100 + second_batch * x / 100 = 
        (initial_stock + second_batch) * final_decaf_percent / 100) :
  x = 70 := by
  sorry

#check decaf_coffee_percentage

end NUMINAMATH_CALUDE_decaf_coffee_percentage_l1638_163853


namespace NUMINAMATH_CALUDE_x_over_y_value_l1638_163805

theorem x_over_y_value (x y : ℝ) (h : 16 * x = 0.24 * 90 * y) : x / y = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_value_l1638_163805


namespace NUMINAMATH_CALUDE_factorization_of_4_minus_4x_squared_l1638_163855

theorem factorization_of_4_minus_4x_squared (x : ℝ) : 4 - 4*x^2 = 4*(1+x)*(1-x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4_minus_4x_squared_l1638_163855


namespace NUMINAMATH_CALUDE_allowance_calculation_l1638_163873

theorem allowance_calculation (card_cost sticker_box_cost : ℚ) 
  (total_sticker_packs : ℕ) (h1 : card_cost = 10) 
  (h2 : sticker_box_cost = 2) (h3 : total_sticker_packs = 4) : 
  (card_cost + sticker_box_cost * (total_sticker_packs / 2)) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_allowance_calculation_l1638_163873


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1638_163858

theorem absolute_value_equation_solutions :
  {x : ℝ | |x - 2| = |x - 3| + |x - 6| + 2} = {-9, 9} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1638_163858


namespace NUMINAMATH_CALUDE_select_four_boots_from_five_pairs_l1638_163857

/-- The number of ways to select 4 boots from 5 pairs, including exactly one pair -/
def select_boots (n : ℕ) : ℕ :=
  let total_pairs := 5
  let pairs_to_choose := 1
  let remaining_pairs := total_pairs - pairs_to_choose
  let boots_to_choose := n - 2 * pairs_to_choose
  (total_pairs.choose pairs_to_choose) * 
  (remaining_pairs.choose (boots_to_choose / 2)) * 
  2^(boots_to_choose)

/-- Theorem stating that there are 120 ways to select 4 boots from 5 pairs, including exactly one pair -/
theorem select_four_boots_from_five_pairs : select_boots 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_four_boots_from_five_pairs_l1638_163857


namespace NUMINAMATH_CALUDE_brown_sugar_amount_l1638_163818

-- Define the amount of white sugar used
def white_sugar : ℝ := 0.25

-- Define the additional amount of brown sugar compared to white sugar
def additional_brown_sugar : ℝ := 0.38

-- Theorem stating the amount of brown sugar used
theorem brown_sugar_amount : 
  white_sugar + additional_brown_sugar = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_brown_sugar_amount_l1638_163818


namespace NUMINAMATH_CALUDE_scooter_initial_value_l1638_163829

/-- 
Given a scooter whose value depreciates to 3/4 of its value each year, 
if its value after one year is 30000, then its initial value was 40000.
-/
theorem scooter_initial_value (initial_value : ℝ) : 
  (3 / 4 : ℝ) * initial_value = 30000 → initial_value = 40000 := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l1638_163829


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l1638_163808

/-- A type representing lines in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A type representing planes in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem perpendicular_parallel_implies_perpendicular 
  (b c : Line3D) (α : Plane3D) :
  perpendicular_line_plane b α → 
  parallel_line_plane c α → 
  perpendicular_lines b c :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l1638_163808


namespace NUMINAMATH_CALUDE_flight_time_sum_l1638_163830

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

def Time.toMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

theorem flight_time_sum (departure : Time) (arrival : Time) (layover : ℕ) 
  (h m : ℕ) (hm : 0 < m ∧ m < 60) :
  departure.hours = 15 ∧ departure.minutes = 45 →
  arrival.hours = 20 ∧ arrival.minutes = 2 →
  layover = 25 →
  arrival.toMinutes - departure.toMinutes - layover = h * 60 + m →
  h + m = 55 := by
sorry

end NUMINAMATH_CALUDE_flight_time_sum_l1638_163830


namespace NUMINAMATH_CALUDE_circle_through_ellipse_vertices_l1638_163834

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point lies on an ellipse -/
def Point.onEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem to prove -/
theorem circle_through_ellipse_vertices (e : Ellipse) (c : Circle) : 
  e.a = 4 ∧ e.b = 2 ∧ 
  c.center.x = 3/2 ∧ c.center.y = 0 ∧ c.radius = 5/2 →
  (∃ (p1 p2 p3 : Point), 
    p1.onEllipse e ∧ p2.onEllipse e ∧ p3.onEllipse e ∧
    p1.onCircle c ∧ p2.onCircle c ∧ p3.onCircle c) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_through_ellipse_vertices_l1638_163834


namespace NUMINAMATH_CALUDE_tan_roots_expression_value_l1638_163884

theorem tan_roots_expression_value (α β : ℝ) :
  (∃ x y : ℝ, x^2 - 4*x - 2 = 0 ∧ y^2 - 4*y - 2 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.cos (α + β)^2 + 2 * Real.sin (α + β) * Real.cos (α + β) - 3 * Real.sin (α + β)^2 = -3/5 :=
by sorry

end NUMINAMATH_CALUDE_tan_roots_expression_value_l1638_163884


namespace NUMINAMATH_CALUDE_best_of_three_max_value_l1638_163867

/-- The maximum value of 8q - 9p in a best-of-three table tennis match -/
theorem best_of_three_max_value (p : ℝ) (q : ℝ) 
  (h1 : 0 < p) (h2 : p < 1) (h3 : q = 3 * p^2 - 2 * p^3) : 
  ∃ (max_val : ℝ), ∀ (p' : ℝ) (q' : ℝ), 
    0 < p' → p' < 1 → q' = 3 * p'^2 - 2 * p'^3 → 
    8 * q' - 9 * p' ≤ max_val ∧ max_val = 0 := by
  sorry

end NUMINAMATH_CALUDE_best_of_three_max_value_l1638_163867


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1638_163854

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  m = (Real.sqrt 3, 1) →
  n = (Real.cos A + 1, Real.sin A) →
  m.1 * n.1 + m.2 * n.2 = 2 + Real.sqrt 3 →
  a = Real.sqrt 3 →
  Real.cos B = Real.sqrt 3 / 3 →
  A = π / 6 ∧ 
  (1 / 2 : ℝ) * a * b * Real.sin C = Real.sqrt 2 / 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1638_163854


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1638_163816

theorem midpoint_sum_equals_vertex_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (a + c) / 2 + (b + c) / 2
  vertex_sum = midpoint_sum := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1638_163816


namespace NUMINAMATH_CALUDE_corrected_mean_l1638_163879

/-- Given 100 observations with an initial mean of 45, and three incorrect recordings
    (60 as 35, 52 as 25, and 85 as 40), the corrected mean is 45.97. -/
theorem corrected_mean (n : ℕ) (initial_mean : ℝ) 
  (error1 error2 error3 : ℝ) (h1 : n = 100) (h2 : initial_mean = 45)
  (h3 : error1 = 60 - 35) (h4 : error2 = 52 - 25) (h5 : error3 = 85 - 40) :
  let total_error := error1 + error2 + error3
  let initial_sum := n * initial_mean
  let corrected_sum := initial_sum + total_error
  corrected_sum / n = 45.97 := by
sorry

end NUMINAMATH_CALUDE_corrected_mean_l1638_163879
