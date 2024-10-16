import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_l656_65656

/-- A rectangle with length 6 and width 4 can be rolled into a cylinder. -/
structure RectangleCylinder where
  length : ℝ
  width : ℝ
  volume : ℝ

/-- The volume of the cylinder is either 24/π or 36/π. -/
theorem rectangle_cylinder_volume (rc : RectangleCylinder) 
  (h1 : rc.length = 6) 
  (h2 : rc.width = 4) : 
  rc.volume = 24 / Real.pi ∨ rc.volume = 36 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_rectangle_cylinder_volume_l656_65656


namespace NUMINAMATH_CALUDE_f_passes_through_point_f_has_max_at_one_f_is_unique_l656_65661

/-- A quadratic function that passes through (2, -6) and has a maximum of -4 at x = 1 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 6

/-- The function f passes through the point (2, -6) -/
theorem f_passes_through_point : f 2 = -6 := by sorry

/-- The function f has a maximum value of -4 when x = 1 -/
theorem f_has_max_at_one :
  (∀ x, f x ≤ f 1) ∧ f 1 = -4 := by sorry

/-- The function f is the unique quadratic function satisfying the given conditions -/
theorem f_is_unique (g : ℝ → ℝ) :
  (g 2 = -6) →
  ((∀ x, g x ≤ g 1) ∧ g 1 = -4) →
  (∃ a b c, ∀ x, g x = a * x^2 + b * x + c) →
  (∀ x, g x = f x) := by sorry

end NUMINAMATH_CALUDE_f_passes_through_point_f_has_max_at_one_f_is_unique_l656_65661


namespace NUMINAMATH_CALUDE_man_rowing_speed_l656_65683

/-- Given a man's downstream speed and speed in still water, calculate his upstream speed -/
theorem man_rowing_speed (downstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 31)
  (h2 : still_water_speed = 28) :
  still_water_speed - (downstream_speed - still_water_speed) = 25 := by
  sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l656_65683


namespace NUMINAMATH_CALUDE_circle_properties_l656_65618

noncomputable def circle_equation (x y : ℝ) : ℝ := (x - Real.sqrt 3)^2 + (y - 1)^2

theorem circle_properties :
  ∃ (c : ℝ × ℝ),
    (∀ x y : ℝ, circle_equation x y = 1 → ‖(x, y) - c‖ = 1) ∧
    (∃ x : ℝ, circle_equation x 0 = 1) ∧
    (∃ x y : ℝ, circle_equation x y = 1 ∧ y = Real.sqrt 3 * x) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l656_65618


namespace NUMINAMATH_CALUDE_square_floor_tiles_l656_65629

theorem square_floor_tiles (black_tiles : ℕ) (h : black_tiles = 57) :
  ∃ (side_length : ℕ),
    (2 * side_length - 1 = black_tiles) ∧
    (side_length * side_length = 841) :=
by sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l656_65629


namespace NUMINAMATH_CALUDE_lake_distance_proof_l656_65664

def lake_distance : Set ℝ := {d | d > 9 ∧ d < 10}

theorem lake_distance_proof (d : ℝ) :
  (¬ (d ≥ 10)) ∧ (¬ (d ≤ 9)) ∧ (d ≠ 7) ↔ d ∈ lake_distance := by
  sorry

end NUMINAMATH_CALUDE_lake_distance_proof_l656_65664


namespace NUMINAMATH_CALUDE_python_eating_theorem_l656_65693

/-- The number of days in the given time period -/
def total_days : ℕ := 616

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The rate at which the python eats alligators (alligators per week) -/
def eating_rate : ℕ := 1

/-- The maximum number of alligators the python can eat in the given time period -/
def max_alligators_eaten : ℕ := total_days / days_per_week

theorem python_eating_theorem :
  max_alligators_eaten = eating_rate * (total_days / days_per_week) :=
by sorry

end NUMINAMATH_CALUDE_python_eating_theorem_l656_65693


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l656_65675

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 4500 →
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 1350 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l656_65675


namespace NUMINAMATH_CALUDE_odd_function_zero_l656_65660

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_l656_65660


namespace NUMINAMATH_CALUDE_min_rooms_sufficient_l656_65689

/-- The minimum number of hotel rooms required for 100 tourists given k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  let m := k / 2
  if k % 2 = 0 then 100 * (m + 1) else 100 * (m + 1) + 1

/-- Theorem stating that min_rooms provides sufficient rooms for 100 tourists -/
theorem min_rooms_sufficient (k : ℕ) :
  ∀ (arrangement : Fin k → Fin (min_rooms k)),
  ∃ (allocation : Fin 100 → Fin (min_rooms k)),
  (∀ i j, i ≠ j → allocation i ≠ allocation j) ∧
  (∀ i, allocation i ∉ Set.range arrangement) :=
sorry

end NUMINAMATH_CALUDE_min_rooms_sufficient_l656_65689


namespace NUMINAMATH_CALUDE_max_x_value_l656_65640

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7) 
  (sum_prod_eq : x * y + x * z + y * z = 11) : 
  x ≤ (7 + Real.sqrt 34) / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l656_65640


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l656_65653

-- Problem 1
theorem problem_1 (m n : ℝ) (hm : m ≠ 0) :
  (2 * m * n) / (3 * m^2) * (6 * m * n) / (5 * n) = 4 * n / 5 :=
sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x ≠ 0) (hy : x ≠ y) :
  (5 * x - 5 * y) / (3 * x^2 * y) * (9 * x * y^2) / (x^2 - y^2) = 15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem problem_3 (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  ((x^3 * y^2) / z)^2 * ((y * z) / x^2)^3 = y^7 * z :=
sorry

-- Problem 4
theorem problem_4 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 2*x + y ≠ 0) (hxy2 : 4*x^2 - y^2 ≠ 0) :
  (4 * x^2 * y^2) / (2*x + y) * (4*x^2 + 4*x*y + y^2) / (2*x + y) / ((2*x*y * (2*x - y)) / (4*x^2 - y^2)) = 4*x^2*y + 2*x*y^2 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l656_65653


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l656_65650

theorem quadratic_form_minimum : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9 ≥ -10 ∧ 
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 2 * y₀^2 - 6 * x₀ + 8 * y₀ + 9 = -10 := by
  sorry

#check quadratic_form_minimum

end NUMINAMATH_CALUDE_quadratic_form_minimum_l656_65650


namespace NUMINAMATH_CALUDE_three_zeros_condition_l656_65688

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - a * x
  else -x^2 - (a + 2) * x + 1

/-- The number of zeros of f(x) -/
def number_of_zeros (a : ℝ) : ℕ := sorry

/-- Theorem stating the condition for f(x) to have exactly 3 zeros -/
theorem three_zeros_condition (a : ℝ) :
  number_of_zeros a = 3 ↔ a > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l656_65688


namespace NUMINAMATH_CALUDE_tim_reading_time_l656_65694

/-- Given that Tim spends 1 hour a day meditating and twice as much time reading,
    prove that he spends 14 hours a week reading. -/
theorem tim_reading_time (meditation_time : ℝ) (reading_time : ℝ) (days_in_week : ℕ) :
  meditation_time = 1 →
  reading_time = 2 * meditation_time →
  days_in_week = 7 →
  reading_time * days_in_week = 14 := by
  sorry

end NUMINAMATH_CALUDE_tim_reading_time_l656_65694


namespace NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l656_65641

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  carbon : Nat
  hydrogen : Nat
  oxygen : Nat

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : Real
  hydrogen : Real
  oxygen : Real

/-- Calculates the molecular weight of a compound -/
def molecularWeight (formula : MolecularFormula) (weights : AtomicWeights) : Real :=
  formula.carbon * weights.carbon + formula.hydrogen * weights.hydrogen + formula.oxygen * weights.oxygen

/-- Theorem stating that the value of y in C6HyO7 is 8 for a molecular weight of 192 g/mol -/
theorem hydrogen_atoms_in_compound (weights : AtomicWeights) 
    (h_carbon : weights.carbon = 12.01)
    (h_hydrogen : weights.hydrogen = 1.01)
    (h_oxygen : weights.oxygen = 16.00) :
  ∃ y : Nat, y = 8 ∧ 
    molecularWeight { carbon := 6, hydrogen := y, oxygen := 7 } weights = 192 := by
  sorry

end NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l656_65641


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l656_65628

/-- Calculates the total cost of typing a manuscript given the page counts and rates. -/
def manuscript_typing_cost (
  total_pages : ℕ) 
  (revised_once : ℕ) 
  (revised_twice : ℕ) 
  (revised_twice_sets : ℕ) 
  (revised_thrice : ℕ) 
  (revised_thrice_sets : ℕ) 
  (initial_rate : ℕ) 
  (revision_rate : ℕ) 
  (set_rate_thrice : ℕ) 
  (set_rate_twice : ℕ) : ℕ :=
  sorry

theorem manuscript_cost_calculation :
  manuscript_typing_cost 
    250  -- total pages
    80   -- pages revised once
    95   -- pages revised twice
    2    -- sets of 20 pages revised twice
    50   -- pages revised thrice
    3    -- sets of 10 pages revised thrice
    5    -- initial typing rate
    3    -- revision rate
    10   -- flat fee for set of 10 pages revised 3+ times
    15   -- flat fee for set of 20 pages revised 2 times
  = 1775 := by sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l656_65628


namespace NUMINAMATH_CALUDE_candy_game_theorem_l656_65601

/-- The maximum number of candies that can be eaten in the candy-eating game. -/
def max_candies (n : ℕ) : ℕ :=
  n.choose 2

/-- The candy-eating game theorem. -/
theorem candy_game_theorem :
  max_candies 27 = 351 :=
by sorry

end NUMINAMATH_CALUDE_candy_game_theorem_l656_65601


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l656_65620

theorem polynomial_multiplication (a b : ℝ) :
  (3 * a^4 - 7 * b^3) * (9 * a^8 + 21 * a^4 * b^3 + 49 * b^6 + 6 * a^2 * b^2) =
  27 * a^12 + 18 * a^6 * b^2 - 42 * a^2 * b^5 - 343 * b^9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l656_65620


namespace NUMINAMATH_CALUDE_enclosing_triangle_sides_l656_65611

/-- An isosceles triangle enclosing a circle -/
structure EnclosingTriangle where
  /-- Radius of the enclosed circle -/
  r : ℝ
  /-- Acute angle at the base of the isosceles triangle in radians -/
  θ : ℝ
  /-- Length of the equal sides of the isosceles triangle -/
  a : ℝ
  /-- Length of the base of the isosceles triangle -/
  b : ℝ

/-- The theorem stating the side lengths of the enclosing isosceles triangle -/
theorem enclosing_triangle_sides (t : EnclosingTriangle) 
  (h_r : t.r = 3)
  (h_θ : t.θ = π/6) -- 30° in radians
  : t.a = 4 * Real.sqrt 3 + 6 ∧ t.b = 6 * Real.sqrt 3 + 12 := by
  sorry


end NUMINAMATH_CALUDE_enclosing_triangle_sides_l656_65611


namespace NUMINAMATH_CALUDE_largest_two_digit_number_l656_65698

def digits : Finset Nat := {1, 2, 4, 6}

def valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

theorem largest_two_digit_number :
  ∀ n, valid_number n → n ≤ 64 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_l656_65698


namespace NUMINAMATH_CALUDE_rental_fee_calculation_l656_65666

/-- Rental fee calculation for comic books -/
theorem rental_fee_calculation 
  (rental_fee_per_30min : ℕ) 
  (num_students : ℕ) 
  (num_books : ℕ) 
  (rental_duration_hours : ℕ) 
  (h1 : rental_fee_per_30min = 4000)
  (h2 : num_students = 6)
  (h3 : num_books = 4)
  (h4 : rental_duration_hours = 3)
  : (rental_fee_per_30min * (rental_duration_hours * 2) * num_books) / num_students = 16000 := by
  sorry

#check rental_fee_calculation

end NUMINAMATH_CALUDE_rental_fee_calculation_l656_65666


namespace NUMINAMATH_CALUDE_xy_value_l656_65672

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l656_65672


namespace NUMINAMATH_CALUDE_sum_f_2015_is_zero_l656_65614

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

def sum_f (f : ℝ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_f f n + f (n + 1)

theorem sum_f_2015_is_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_sym : is_symmetric_about_one f) 
  (h_f_neg_one : f (-1) = 1) : 
  sum_f f 2015 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_2015_is_zero_l656_65614


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l656_65676

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem for the first line
theorem line_through_P_and_origin :
  ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (x + y = 0 ↔ (∃ t : ℝ, x = t * P.1 ∧ y = t * P.2)) :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ :
  ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (2 * x + y + 2 = 0 ↔ 
    (∃ t : ℝ, x = P.1 + t * 1 ∧ y = P.2 + t * (-2) ∧ 
    (1 * (-2) + 2 * 1 = 0))) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l656_65676


namespace NUMINAMATH_CALUDE_rhombus_side_length_l656_65649

/-- Given a rhombus with area K and diagonals d and 3d, prove its side length. -/
theorem rhombus_side_length (K d : ℝ) (h1 : K > 0) (h2 : d > 0) : ∃ s : ℝ,
  (K = (3 * d^2) / 2) →  -- Area formula for rhombus
  (s^2 = (d^2 / 4) + ((3 * d)^2 / 4)) →  -- Pythagorean theorem for side length
  s = Real.sqrt ((5 * K) / 3) :=
sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l656_65649


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l656_65670

theorem certain_fraction_proof (x : ℚ) : 
  (2 / 5) / x = (7 / 15) / (1 / 2) → x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l656_65670


namespace NUMINAMATH_CALUDE_jeans_prices_l656_65600

/-- Represents the shopping scenario with Mary and her children --/
structure ShoppingScenario where
  coat_original_price : ℝ
  coat_discount_rate : ℝ
  backpack_cost : ℝ
  shoes_cost : ℝ
  subtotal : ℝ
  jeans_price_difference : ℝ
  sales_tax_rate : ℝ

/-- Theorem stating the prices of Jamie's jeans --/
theorem jeans_prices (scenario : ShoppingScenario)
  (h_coat : scenario.coat_original_price = 50)
  (h_discount : scenario.coat_discount_rate = 0.1)
  (h_backpack : scenario.backpack_cost = 25)
  (h_shoes : scenario.shoes_cost = 30)
  (h_subtotal : scenario.subtotal = 139)
  (h_difference : scenario.jeans_price_difference = 15)
  (h_tax : scenario.sales_tax_rate = 0.07) :
  ∃ (cheap_jeans expensive_jeans : ℝ),
    cheap_jeans = 12 ∧
    expensive_jeans = 27 ∧
    cheap_jeans + expensive_jeans = scenario.subtotal -
      (scenario.coat_original_price * (1 - scenario.coat_discount_rate) +
       scenario.backpack_cost + scenario.shoes_cost) ∧
    expensive_jeans - cheap_jeans = scenario.jeans_price_difference :=
by sorry

end NUMINAMATH_CALUDE_jeans_prices_l656_65600


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l656_65665

theorem smallest_integer_in_set (n : ℤ) : 
  (7 * n + 21 > 4 * n) → (∀ m : ℤ, m < n → ¬(7 * m + 21 > 4 * m)) → n = -6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l656_65665


namespace NUMINAMATH_CALUDE_route_down_length_is_15_l656_65679

/-- Represents a hiking trip up and down a mountain -/
structure HikingTrip where
  rateUp : ℝ        -- Rate of hiking up the mountain in miles per day
  timeUp : ℝ        -- Time taken to hike up in days
  rateDownFactor : ℝ -- Factor by which the rate down is faster than the rate up

/-- Calculates the length of the route down the mountain -/
def routeDownLength (trip : HikingTrip) : ℝ :=
  trip.rateUp * trip.rateDownFactor * trip.timeUp

/-- Theorem stating that for the given conditions, the route down is 15 miles long -/
theorem route_down_length_is_15 : 
  ∀ (trip : HikingTrip), 
  trip.rateUp = 5 ∧ 
  trip.timeUp = 2 ∧ 
  trip.rateDownFactor = 1.5 → 
  routeDownLength trip = 15 := by
  sorry


end NUMINAMATH_CALUDE_route_down_length_is_15_l656_65679


namespace NUMINAMATH_CALUDE_curve_intersection_and_tangent_l656_65690

noncomputable section

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := Real.exp x * (c*x + d)

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 2*x + a

-- Define the derivative of g
def g' (c d x : ℝ) : ℝ := Real.exp x * (c*x + d + c)

-- State the theorem
theorem curve_intersection_and_tangent (a b c d : ℝ) :
  (f a b 0 = 2) →
  (g c d 0 = 2) →
  (f' a 0 = 4) →
  (g' c d 0 = 4) →
  (a = 4 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∧
  (∀ k, (∀ x, x ≥ -2 → f 4 2 x ≤ k * g 2 2 x) ↔ (1 ≤ k ∧ k ≤ Real.exp 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_curve_intersection_and_tangent_l656_65690


namespace NUMINAMATH_CALUDE_taco_cost_l656_65682

-- Define the cost of a taco and an enchilada
variable (T E : ℚ)

-- Define the conditions from the problem
def condition1 : Prop := 2 * T + 3 * E = 390 / 50
def condition2 : Prop := 3 * T + 5 * E = 635 / 50

-- Theorem to prove
theorem taco_cost (h1 : condition1 T E) (h2 : condition2 T E) : T = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_taco_cost_l656_65682


namespace NUMINAMATH_CALUDE_square_sum_inequality_l656_65607

theorem square_sum_inequality {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) ∧
  ((a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l656_65607


namespace NUMINAMATH_CALUDE_color_tape_overlap_l656_65663

theorem color_tape_overlap (total_length : ℝ) (tape_length : ℝ) (num_tapes : ℕ) 
  (h1 : total_length = 50.5)
  (h2 : tape_length = 18)
  (h3 : num_tapes = 3) :
  (num_tapes * tape_length - total_length) / 2 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_color_tape_overlap_l656_65663


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l656_65627

/-- Represents the delivery rates for different weight ranges -/
structure DeliveryRates :=
  (kg3to5 : ℝ)
  (kg6to8 : ℝ)
  (kg9to12 : ℝ)
  (kg13to15 : ℝ)

/-- Represents the daily deliveries -/
structure DailyDeliveries :=
  (kg5 : ℕ)
  (kg8 : ℕ)
  (kg10 : ℕ)
  (kg14 : ℕ)

def weekdayRates : DeliveryRates :=
  { kg3to5 := 2.5, kg6to8 := 4, kg9to12 := 6, kg13to15 := 8 }

def weekendRates : DeliveryRates :=
  { kg3to5 := 3, kg6to8 := 5, kg9to12 := 7.5, kg13to15 := 10 }

def weekdayDeliveries : DailyDeliveries :=
  { kg5 := 4, kg8 := 2, kg10 := 3, kg14 := 1 }

def weekendDeliveries : DailyDeliveries :=
  { kg5 := 2, kg8 := 3, kg10 := 0, kg14 := 2 }

def weekdaysInWeek : ℕ := 5
def weekendDaysInWeek : ℕ := 2

/-- Calculates the daily earnings based on rates and deliveries -/
def dailyEarnings (rates : DeliveryRates) (deliveries : DailyDeliveries) : ℝ :=
  rates.kg3to5 * deliveries.kg5 +
  rates.kg6to8 * deliveries.kg8 +
  rates.kg9to12 * deliveries.kg10 +
  rates.kg13to15 * deliveries.kg14

/-- Calculates the total weekly earnings -/
def weeklyEarnings : ℝ :=
  weekdaysInWeek * dailyEarnings weekdayRates weekdayDeliveries +
  weekendDaysInWeek * dailyEarnings weekendRates weekendDeliveries

theorem cherry_weekly_earnings :
  weeklyEarnings = 302 := by sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l656_65627


namespace NUMINAMATH_CALUDE_system_solution_l656_65674

theorem system_solution (w x y z : ℚ) 
  (eq1 : 2*w + x + y + z = 1)
  (eq2 : w + 2*x + y + z = 2)
  (eq3 : w + x + 2*y + z = 2)
  (eq4 : w + x + y + 2*z = 1) :
  w = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l656_65674


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_and_ln_l656_65692

theorem tangent_line_to_exp_and_ln (a b : ℝ) : 
  (∃ x₁ : ℝ, (x₁ + b = Real.exp x₁) ∧ (1 = Real.exp x₁)) →
  (∃ x₂ : ℝ, (x₂ + b = Real.log (x₂ + a)) ∧ (1 = 1 / (x₂ + a))) →
  a = 2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_and_ln_l656_65692


namespace NUMINAMATH_CALUDE_investment_equation_l656_65647

/-- Proves that the total amount invested satisfies the given equation based on the problem conditions -/
theorem investment_equation (total_interest : ℝ) (higher_rate_fraction : ℝ) 
  (lower_rate : ℝ) (higher_rate : ℝ) :
  total_interest = 1440 →
  higher_rate_fraction = 0.55 →
  lower_rate = 0.06 →
  higher_rate = 0.09 →
  ∃ T : ℝ, 0.0765 * T = 1440 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_equation_l656_65647


namespace NUMINAMATH_CALUDE_end_time_calculation_l656_65605

-- Define the structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define the problem parameters
def glowInterval : Nat := 17
def startTime : Time := { hours := 1, minutes := 57, seconds := 58 }
def glowCount : Float := 292.29411764705884

-- Define the function to calculate the ending time
def calculateEndTime (start : Time) (interval : Nat) (count : Float) : Time :=
  sorry

-- Theorem statement
theorem end_time_calculation :
  calculateEndTime startTime glowInterval glowCount = { hours := 3, minutes := 20, seconds := 42 } :=
sorry

end NUMINAMATH_CALUDE_end_time_calculation_l656_65605


namespace NUMINAMATH_CALUDE_product_zero_l656_65621

/-- Given two real numbers x and y satisfying x - y = 6 and x³ - y³ = 162, their product xy equals 0. -/
theorem product_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l656_65621


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l656_65604

-- Define proposition P
def P : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 1/x₀ > 3

-- Define proposition q
def q : Prop := ∀ x : ℝ, x > 2 → x^2 > 2^x

-- Theorem statement
theorem p_true_and_q_false : P ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l656_65604


namespace NUMINAMATH_CALUDE_largest_and_smallest_numbers_l656_65695

-- Define the numbers in their respective bases
def num1 : ℕ := 63  -- 111111₂ in decimal
def num2 : ℕ := 78  -- 210₆ in decimal
def num3 : ℕ := 64  -- 1000₄ in decimal
def num4 : ℕ := 65  -- 81₈ in decimal

-- Theorem statement
theorem largest_and_smallest_numbers :
  (num2 = max num1 (max num2 (max num3 num4))) ∧
  (num1 = min num1 (min num2 (min num3 num4))) := by
  sorry

end NUMINAMATH_CALUDE_largest_and_smallest_numbers_l656_65695


namespace NUMINAMATH_CALUDE_power_sum_zero_l656_65671

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2^(3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l656_65671


namespace NUMINAMATH_CALUDE_total_blocks_l656_65612

theorem total_blocks (red : ℕ) (yellow : ℕ) (green : ℕ) (blue : ℕ) (orange : ℕ) (purple : ℕ)
  (h1 : red = 24)
  (h2 : yellow = red + 8)
  (h3 : green = yellow - 10)
  (h4 : blue = 2 * green)
  (h5 : orange = blue + 15)
  (h6 : purple = red + orange - 7) :
  red + yellow + green + blue + orange + purple = 257 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_l656_65612


namespace NUMINAMATH_CALUDE_largest_x_floor_fraction_l656_65615

theorem largest_x_floor_fraction : 
  ∃ (x : ℝ), x = 120/11 ∧ 
  (∀ (y : ℝ), (↑(⌊y⌋) : ℝ) / y = 11/12 → y ≤ x) ∧ 
  (↑(⌊x⌋) : ℝ) / x = 11/12 :=
sorry

end NUMINAMATH_CALUDE_largest_x_floor_fraction_l656_65615


namespace NUMINAMATH_CALUDE_income_comparison_l656_65636

theorem income_comparison (Tim Mary Juan : ℝ) 
  (h1 : Mary = 1.60 * Tim) 
  (h2 : Mary = 1.28 * Juan) : 
  Tim = 0.80 * Juan := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l656_65636


namespace NUMINAMATH_CALUDE_tangent_curves_a_value_l656_65658

theorem tangent_curves_a_value (a : ℝ) : 
  let f (x : ℝ) := x + Real.log x
  let g (x : ℝ) := a * x^2 + (a + 2) * x + 1
  let f' (x : ℝ) := 1 + 1 / x
  let g' (x : ℝ) := 2 * a * x + (a + 2)
  (f 1 = g 1) ∧ 
  (f' 1 = g' 1) ∧ 
  (∀ x ≠ 1, f x ≠ g x) →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_tangent_curves_a_value_l656_65658


namespace NUMINAMATH_CALUDE_jeff_probability_multiple_of_four_l656_65609

/-- The number of cards --/
def num_cards : ℕ := 12

/-- The probability of moving left on a single spin --/
def prob_left : ℚ := 1/2

/-- The probability of moving right on a single spin --/
def prob_right : ℚ := 1/2

/-- The number of spaces moved left --/
def spaces_left : ℕ := 1

/-- The number of spaces moved right --/
def spaces_right : ℕ := 2

/-- The probability of ending up at a multiple of 4 --/
def prob_multiple_of_four : ℚ := 5/32

theorem jeff_probability_multiple_of_four :
  let start_at_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let start_two_more_than_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let start_two_less_than_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let end_at_multiple_of_four_from_multiple_of_four := prob_left * prob_right + prob_right * prob_left
  let end_at_multiple_of_four_from_two_more := prob_right * prob_right
  let end_at_multiple_of_four_from_two_less := prob_left * prob_left
  start_at_multiple_of_four * end_at_multiple_of_four_from_multiple_of_four +
  start_two_more_than_multiple_of_four * end_at_multiple_of_four_from_two_more +
  start_two_less_than_multiple_of_four * end_at_multiple_of_four_from_two_less =
  prob_multiple_of_four := by
  sorry

end NUMINAMATH_CALUDE_jeff_probability_multiple_of_four_l656_65609


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l656_65626

-- Define the complex number z
variable (z : ℂ)

-- Define the equation z ⋅ (1+2i)² = 3+4i
def equation (z : ℂ) : Prop := z * (1 + 2*Complex.I)^2 = 3 + 4*Complex.I

-- Theorem statement
theorem z_in_fourth_quadrant (h : equation z) : 
  0 < z.re ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l656_65626


namespace NUMINAMATH_CALUDE_gcd_of_4536_13440_216_l656_65669

theorem gcd_of_4536_13440_216 : Nat.gcd 4536 (Nat.gcd 13440 216) = 216 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_4536_13440_216_l656_65669


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l656_65684

def cupcake_problem (packages : ℕ) (cupcakes_per_package : ℕ) (eaten : ℕ) : Prop :=
  let total := packages * cupcakes_per_package
  let remaining := total - eaten
  remaining = 7

theorem cupcakes_remaining :
  cupcake_problem 3 4 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l656_65684


namespace NUMINAMATH_CALUDE_stamp_difference_l656_65619

theorem stamp_difference (p q : ℕ) (h1 : p * 4 = q * 7) 
  (h2 : (p - 8) * 5 = (q + 8) * 6) : p - q = 8 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l656_65619


namespace NUMINAMATH_CALUDE_parabola_shift_right_parabola_shift_result_l656_65685

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

theorem parabola_shift_right (x : ℝ) :
  let original := Parabola.mk 1 6 0
  let shifted := shift_parabola original 4
  (x^2 + 6*x) = ((x-4)^2 + 6*(x-4)) :=
by sorry

theorem parabola_shift_result :
  let original := Parabola.mk 1 6 0
  let shifted := shift_parabola original 4
  shifted.a * x^2 + shifted.b * x + shifted.c = (x - 1)^2 - 9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_right_parabola_shift_result_l656_65685


namespace NUMINAMATH_CALUDE_horse_tile_problem_representation_l656_65697

/-- Represents the equation for the horse and tile problem -/
def horse_tile_equation (x : ℝ) : Prop :=
  3 * x + (1/3) * (100 - x) = 100

/-- The total number of horses -/
def total_horses : ℝ := 100

/-- The total number of tiles -/
def total_tiles : ℝ := 100

/-- The number of tiles a big horse can pull -/
def big_horse_capacity : ℝ := 3

/-- The number of small horses needed to pull one tile -/
def small_horses_per_tile : ℝ := 3

/-- Theorem stating that the equation correctly represents the problem -/
theorem horse_tile_problem_representation :
  ∀ x, x ≥ 0 ∧ x ≤ total_horses →
  horse_tile_equation x ↔
    (x * big_horse_capacity + (total_horses - x) / small_horses_per_tile = total_tiles) :=
by sorry

end NUMINAMATH_CALUDE_horse_tile_problem_representation_l656_65697


namespace NUMINAMATH_CALUDE_smallest_lucky_integer_l656_65668

/-- An integer is lucky if there exist several consecutive integers, including itself, that add up to 2023. -/
def IsLucky (n : ℤ) : Prop :=
  ∃ k : ℕ, ∃ m : ℤ, (m + k : ℤ) = n ∧ (k + 1) * (2 * m + k) / 2 = 2023

/-- The smallest lucky integer -/
def SmallestLuckyInteger : ℤ := -2022

theorem smallest_lucky_integer :
  IsLucky SmallestLuckyInteger ∧
  ∀ n : ℤ, n < SmallestLuckyInteger → ¬IsLucky n :=
by sorry

end NUMINAMATH_CALUDE_smallest_lucky_integer_l656_65668


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l656_65645

/-- The cubic polynomial q(x) that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (51/13) * x^3 - (31/13) * x^2 + (16/13) * x + (3/13)

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l656_65645


namespace NUMINAMATH_CALUDE_max_k_value_l656_65631

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1 - 2*m) ≥ k) → k ≤ 8) ∧ 
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1 - 2*m) ≥ k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l656_65631


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l656_65602

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c (x + 1) - f a b c x = 2 * x) →
  f a b c 0 = 1 →
  (∃ m : ℝ, 
    (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a b c x ≥ 2 * x + m) ∧
    (∀ m' : ℝ, m' > m → ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a b c x < 2 * x + m')) →
  (∀ x : ℝ, f a b c x = x^2 - x + 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a b c x ≥ 2 * x + (-1)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l656_65602


namespace NUMINAMATH_CALUDE_four_roots_condition_l656_65696

/-- If the equation x^2 - 4|x| + 5 = m has four distinct real roots, then 1 < m < 5 -/
theorem four_roots_condition (m : ℝ) : 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^2 - 4*|x| + 5 = m ↔ (x = a ∨ x = b ∨ x = c ∨ x = d))) →
  1 < m ∧ m < 5 := by
sorry


end NUMINAMATH_CALUDE_four_roots_condition_l656_65696


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l656_65634

theorem fixed_point_of_exponential_translation (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x-3) + 3
  f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l656_65634


namespace NUMINAMATH_CALUDE_helga_shopping_items_l656_65652

def shopping_trip (store1_shoes store1_bags : ℕ) : Prop :=
  let store2_shoes := 2 * store1_shoes
  let store2_bags := store1_bags + 6
  let store3_shoes := 0
  let store3_bags := 0
  let store4_shoes := store1_bags + store2_bags
  let store4_bags := 0
  let store5_shoes := store4_shoes / 2
  let store5_bags := 8
  let store6_shoes := Int.floor (Real.sqrt (store2_shoes + store5_shoes))
  let store6_bags := store1_bags + store2_bags + store5_bags + 5
  let total_shoes := store1_shoes + store2_shoes + store3_shoes + store4_shoes + store5_shoes + store6_shoes
  let total_bags := store1_bags + store2_bags + store3_bags + store4_bags + store5_bags + store6_bags
  total_shoes + total_bags = 95

theorem helga_shopping_items :
  shopping_trip 7 4 := by
  sorry

end NUMINAMATH_CALUDE_helga_shopping_items_l656_65652


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l656_65686

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l656_65686


namespace NUMINAMATH_CALUDE_tony_puzzle_solution_l656_65646

/-- The number of puzzles Tony solved after the warm-up puzzle -/
def puzzles_after_warmup : ℕ := 2

/-- The time taken for the warm-up puzzle in minutes -/
def warmup_time : ℕ := 10

/-- The total time Tony spent solving puzzles in minutes -/
def total_time : ℕ := 70

/-- Each puzzle after the warm-up takes this many times longer than the warm-up -/
def puzzle_time_multiplier : ℕ := 3

theorem tony_puzzle_solution :
  warmup_time + puzzles_after_warmup * (puzzle_time_multiplier * warmup_time) = total_time :=
by sorry

end NUMINAMATH_CALUDE_tony_puzzle_solution_l656_65646


namespace NUMINAMATH_CALUDE_right_triangle_set_l656_65623

theorem right_triangle_set : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = Real.sqrt 2 ∧ b = 5 ∧ c = 2 * Real.sqrt 7) ∨
   (a = 6 ∧ b = 9 ∧ c = 15) ∨
   (a = 4 ∧ b = 12 ∧ c = 13)) ∧
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_set_l656_65623


namespace NUMINAMATH_CALUDE_monotonic_range_of_a_l656_65648

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * Real.exp (a * x)

/-- The function f is monotonic on ℝ -/
def is_monotonic (a : ℝ) : Prop :=
  Monotone (f a) ∨ StrictMono (f a)

/-- The theorem stating the range of a for which f is monotonic -/
theorem monotonic_range_of_a :
  ∀ a : ℝ, is_monotonic a ↔ a ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ioo 1 (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_range_of_a_l656_65648


namespace NUMINAMATH_CALUDE_divisors_of_720_l656_65616

theorem divisors_of_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_720_l656_65616


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_sugar_mixed_number_l656_65655

theorem sugar_recipe_reduction : 
  let original_sugar : ℚ := 31/4
  let reduced_sugar : ℚ := (1/3) * original_sugar
  reduced_sugar = 31/12 := by sorry

theorem sugar_mixed_number :
  let reduced_sugar : ℚ := 31/12
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    reduced_sugar = whole + (numerator : ℚ) / denominator ∧
    whole = 2 ∧ numerator = 7 ∧ denominator = 12 := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_sugar_mixed_number_l656_65655


namespace NUMINAMATH_CALUDE_nap_hours_in_70_days_l656_65651

/-- Calculates the total hours of naps taken in a given number of days -/
def total_nap_hours (days : ℕ) (naps_per_week : ℕ) (hours_per_nap : ℕ) : ℕ :=
  let weeks : ℕ := days / 7
  let total_naps : ℕ := weeks * naps_per_week
  total_naps * hours_per_nap

/-- Theorem stating that 70 days of naps results in 60 hours of nap time -/
theorem nap_hours_in_70_days :
  total_nap_hours 70 3 2 = 60 := by
  sorry

#eval total_nap_hours 70 3 2

end NUMINAMATH_CALUDE_nap_hours_in_70_days_l656_65651


namespace NUMINAMATH_CALUDE_point_on_graph_l656_65632

/-- The function f(x) = -3x + 3 -/
def f (x : ℝ) : ℝ := -3 * x + 3

/-- The point p = (-2, 9) -/
def p : ℝ × ℝ := (-2, 9)

/-- Theorem: The point p lies on the graph of f -/
theorem point_on_graph : f p.1 = p.2 := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l656_65632


namespace NUMINAMATH_CALUDE_not_false_is_true_l656_65691

theorem not_false_is_true (p q : Prop) (hp : p) (hq : ¬q) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_false_is_true_l656_65691


namespace NUMINAMATH_CALUDE_road_length_probability_l656_65606

/-- The probability of a road from A to B being at least 5 miles long -/
def prob_ab : ℚ := 2/3

/-- The probability of a road from B to C being at least 5 miles long -/
def prob_bc : ℚ := 3/4

/-- The probability that at least one of two randomly picked roads
    (one from A to B, one from B to C) is at least 5 miles long -/
def prob_at_least_one : ℚ := 1 - (1 - prob_ab) * (1 - prob_bc)

theorem road_length_probability : prob_at_least_one = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_road_length_probability_l656_65606


namespace NUMINAMATH_CALUDE_inequality_system_solution_l656_65617

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 4) / 3 > x / 2 + 1 ∧ x + a < 0 → x < 2) →
  (∀ x : ℝ, x < 2 → x + a < 0) →
  a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l656_65617


namespace NUMINAMATH_CALUDE_f_neg_two_eq_nine_l656_65625

/-- The function f(x) = x^5 + ax^3 + x^2 + bx + 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + x^2 + b*x + 2

/-- Theorem: If f(2) = 3, then f(-2) = 9 -/
theorem f_neg_two_eq_nine {a b : ℝ} (h : f a b 2 = 3) : f a b (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_nine_l656_65625


namespace NUMINAMATH_CALUDE_daniella_savings_l656_65673

/-- Daniella's savings amount -/
def D : ℝ := 400

/-- Ariella's initial savings amount -/
def A : ℝ := D + 200

/-- Interest rate per annum (as a decimal) -/
def r : ℝ := 0.1

/-- Time period in years -/
def t : ℝ := 2

/-- Ariella's final amount after interest -/
def F : ℝ := 720

theorem daniella_savings : 
  (A + A * r * t = F) → D = 400 := by
  sorry

end NUMINAMATH_CALUDE_daniella_savings_l656_65673


namespace NUMINAMATH_CALUDE_distance_representation_l656_65603

theorem distance_representation (A B : ℝ) (hA : A = 3) (hB : B = -2) :
  |A - B| = |3 - (-2)| := by sorry

end NUMINAMATH_CALUDE_distance_representation_l656_65603


namespace NUMINAMATH_CALUDE_remainder_x_50_divided_by_x2_minus_4x_plus_3_l656_65659

theorem remainder_x_50_divided_by_x2_minus_4x_plus_3 (x : ℝ) :
  ∃ (Q : ℝ → ℝ), x^50 = (x^2 - 4*x + 3) * Q x + ((3^50 - 1)/2 * x + (5 - 3^50)/2) :=
by sorry

end NUMINAMATH_CALUDE_remainder_x_50_divided_by_x2_minus_4x_plus_3_l656_65659


namespace NUMINAMATH_CALUDE_solve_equation_l656_65630

theorem solve_equation (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l656_65630


namespace NUMINAMATH_CALUDE_stratified_sampling_suitable_l656_65633

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a school population -/
structure SchoolPopulation where
  total_students : Nat
  boys : Nat
  girls : Nat
  sample_size : Nat

/-- Determines if a sampling method is suitable for a given school population -/
def is_suitable_sampling_method (population : SchoolPopulation) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified ∧
  population.total_students = population.boys + population.girls ∧
  population.sample_size < population.total_students

/-- Theorem stating that stratified sampling is suitable for the given school population -/
theorem stratified_sampling_suitable (population : SchoolPopulation) 
  (h1 : population.total_students = 1000)
  (h2 : population.boys = 520)
  (h3 : population.girls = 480)
  (h4 : population.sample_size = 100) :
  is_suitable_sampling_method population SamplingMethod.Stratified :=
by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_suitable_l656_65633


namespace NUMINAMATH_CALUDE_abs_is_even_l656_65677

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def abs_function (x : ℝ) : ℝ := |x|

theorem abs_is_even : is_even_function abs_function := by
  sorry

end NUMINAMATH_CALUDE_abs_is_even_l656_65677


namespace NUMINAMATH_CALUDE_incorrect_statement_l656_65678

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) :
  ¬(∀ (α β : Plane) (m n : Line),
    perp m n → perpPlane n α → parallel n β → perpPlanes α β) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l656_65678


namespace NUMINAMATH_CALUDE_pizza_and_burgers_theorem_l656_65654

/-- The number of pupils who like both pizza and burgers -/
def both_pizza_and_burgers (total : ℕ) (pizza : ℕ) (burgers : ℕ) : ℕ :=
  pizza + burgers - total

/-- Theorem: Given 200 total pupils, 125 who like pizza, and 115 who like burgers,
    40 pupils like both pizza and burgers. -/
theorem pizza_and_burgers_theorem :
  both_pizza_and_burgers 200 125 115 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pizza_and_burgers_theorem_l656_65654


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_specific_l656_65608

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (start : Int) (diff : Int) (endInclusive : Int) : Nat :=
  if start > endInclusive then 0
  else ((endInclusive - start) / diff).toNat + 1

theorem arithmetic_sequence_length_specific :
  arithmeticSequenceLength (-48) 7 119 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_specific_l656_65608


namespace NUMINAMATH_CALUDE_cookie_box_weight_limit_l656_65610

/-- The weight limit of a cookie box in pounds, given the weight of each cookie and the number of cookies it can hold. -/
theorem cookie_box_weight_limit (cookie_weight : ℚ) (box_capacity : ℕ) : 
  cookie_weight = 2 → box_capacity = 320 → (cookie_weight * box_capacity) / 16 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cookie_box_weight_limit_l656_65610


namespace NUMINAMATH_CALUDE_range_of_t_l656_65639

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ -1}
def B (t : ℝ) : Set ℝ := {y : ℝ | y ≥ t}

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem range_of_t (t : ℝ) :
  (∀ x ∈ A, f x ∈ B t) → t ≤ 0 := by
  sorry

-- Define the final result
def result : Set ℝ := {t : ℝ | t ≤ 0}

end NUMINAMATH_CALUDE_range_of_t_l656_65639


namespace NUMINAMATH_CALUDE_cupcake_difference_l656_65657

theorem cupcake_difference (morning_cupcakes afternoon_cupcakes total_cupcakes : ℕ) : 
  morning_cupcakes = 20 →
  total_cupcakes = 55 →
  afternoon_cupcakes = total_cupcakes - morning_cupcakes →
  afternoon_cupcakes - morning_cupcakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_difference_l656_65657


namespace NUMINAMATH_CALUDE_degree_of_g_given_f_plus_g_l656_65681

/-- Given two polynomials f and g, where f(x) = -3x^5 + 2x^4 + x^2 - 6 and the degree of f + g is 2, the degree of g is 5. -/
theorem degree_of_g_given_f_plus_g (f g : Polynomial ℝ) : 
  f = -3 * X^5 + 2 * X^4 + X^2 - 6 →
  Polynomial.degree (f + g) = 2 →
  Polynomial.degree g = 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_g_given_f_plus_g_l656_65681


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_less_than_zero_negation_of_cubic_inequality_l656_65637

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem negation_of_less_than_zero (x : ℝ) :
  ¬(x < 0) ↔ (x ≥ 0) :=
by sorry

theorem negation_of_cubic_inequality :
  (¬∀ x : ℝ, x^3 + 2 < 0) ↔ (∃ x : ℝ, x^3 + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_less_than_zero_negation_of_cubic_inequality_l656_65637


namespace NUMINAMATH_CALUDE_intersection_equiv_open_interval_l656_65662

def set_A : Set ℝ := {x | x / (x - 1) ≤ 0}
def set_B : Set ℝ := {x | x^2 < 2*x}

theorem intersection_equiv_open_interval : 
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ x ∈ Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equiv_open_interval_l656_65662


namespace NUMINAMATH_CALUDE_intersection_distance_l656_65667

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  Real.sqrt 3 * x - y + 3 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*y - 2*Real.sqrt 3*x = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l656_65667


namespace NUMINAMATH_CALUDE_difference_from_averages_l656_65642

theorem difference_from_averages (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50) 
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_difference_from_averages_l656_65642


namespace NUMINAMATH_CALUDE_boys_to_total_ratio_l656_65680

theorem boys_to_total_ratio 
  (b g : ℕ) -- number of boys and girls
  (h1 : b > 0 ∧ g > 0) -- ensure non-empty class
  (h2 : (b : ℚ) / (b + g) = 4/5 * (g : ℚ) / (b + g)) -- probability condition
  : (b : ℚ) / (b + g) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_total_ratio_l656_65680


namespace NUMINAMATH_CALUDE_range_of_f_l656_65638

-- Define the function f
def f (x : ℝ) : ℝ := (x^3 - 3*x + 1)^2

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l656_65638


namespace NUMINAMATH_CALUDE_one_solution_less_than_two_l656_65624

def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 14*x^6 + 1429*x^5 - 1279*x^4

theorem one_solution_less_than_two :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_solution_less_than_two_l656_65624


namespace NUMINAMATH_CALUDE_line_through_focus_line_intersects_ellipse_l656_65699

/-- The equation of the line l -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 4

/-- The equation of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

/-- The x-coordinate of the left focus of the ellipse -/
def left_focus : ℝ := -2

/-- Theorem: When the line passes through the left focus of the ellipse, k = 2 -/
theorem line_through_focus (k : ℝ) : 
  line k left_focus = 0 → k = 2 :=
sorry

/-- Theorem: The line intersects the ellipse if and only if k is in the specified range -/
theorem line_intersects_ellipse (k : ℝ) : 
  (∃ x y, ellipse x y ∧ y = line k x) ↔ k ≤ -Real.sqrt 3 ∨ k ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_line_through_focus_line_intersects_ellipse_l656_65699


namespace NUMINAMATH_CALUDE_units_digit_of_p_l656_65644

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  p > 0 → 
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  units_digit (p + 1) = 7 →
  units_digit p = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l656_65644


namespace NUMINAMATH_CALUDE_f_two_equals_six_l656_65613

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem f_two_equals_six (a b : ℝ) (h : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_six_l656_65613


namespace NUMINAMATH_CALUDE_innocent_statement_l656_65643

/-- Represents the type of person making a statement --/
inductive PersonType
| Knight
| Liar
| Normal

/-- Represents a statement that can be made --/
inductive Statement
| IAmALiar

/-- Defines whether a statement is true or false --/
def isTrue : PersonType → Statement → Prop
| PersonType.Knight, Statement.IAmALiar => False
| PersonType.Liar, Statement.IAmALiar => False
| PersonType.Normal, Statement.IAmALiar => True

theorem innocent_statement :
  ∀ (p : PersonType), p ≠ PersonType.Normal → ¬(isTrue p Statement.IAmALiar) := by
  sorry

end NUMINAMATH_CALUDE_innocent_statement_l656_65643


namespace NUMINAMATH_CALUDE_solutions_not_real_root_loci_l656_65622

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℂ) : ℂ := x^2 + a*x + 1

-- Theorem for the interval of a where solutions are not real
theorem solutions_not_real (a : ℝ) :
  (∀ x : ℂ, quadratic a x = 0 → x.im ≠ 0) ↔ a ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

-- Define the ellipse
def ellipse (z : ℂ) : Prop := 4 * z.re^2 + z.im^2 = 4

-- Theorem for the loci of roots
theorem root_loci (a : ℝ) (z : ℂ) :
  a ∈ Set.Ioo (-2 : ℝ) 2 →
  (quadratic a z = 0 ↔ (ellipse z ∧ z ≠ -1 ∧ z ≠ 1)) :=
sorry

end NUMINAMATH_CALUDE_solutions_not_real_root_loci_l656_65622


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l656_65687

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l656_65687


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l656_65635

/-- The y-intercept of the line 2x + 7y = 35 is (0, 5) -/
theorem y_intercept_of_line (x y : ℝ) :
  2 * x + 7 * y = 35 → y = 5 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l656_65635
