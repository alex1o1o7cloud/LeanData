import Mathlib

namespace NUMINAMATH_CALUDE_equation_value_l3248_324823

theorem equation_value (x y : ℚ) 
  (eq1 : 5 * x + 6 * y = 7) 
  (eq2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_value_l3248_324823


namespace NUMINAMATH_CALUDE_total_cats_l3248_324807

/-- The number of cats that can jump -/
def jump : ℕ := 60

/-- The number of cats that can fetch -/
def fetch : ℕ := 35

/-- The number of cats that can meow on command -/
def meow : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and meow -/
def fetch_meow : ℕ := 15

/-- The number of cats that can jump and meow -/
def jump_meow : ℕ := 25

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 11

/-- The number of cats that can do none of the tricks -/
def no_tricks : ℕ := 10

/-- Theorem stating the total number of cats in the training center -/
theorem total_cats : 
  jump + fetch + meow - jump_fetch - fetch_meow - jump_meow + all_three + no_tricks = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l3248_324807


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3248_324873

/-- Given a geometric sequence {a_n} where a_1 and a_5 are the positive roots of x^2 - 10x + 16 = 0, prove that a_3 = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (a 1 * a 1 - 10 * a 1 + 16 = 0) →  -- a_1 is a root of x^2 - 10x + 16 = 0
  (a 5 * a 5 - 10 * a 5 + 16 = 0) →  -- a_5 is a root of x^2 - 10x + 16 = 0
  (0 < a 1) →  -- a_1 is positive
  (0 < a 5) →  -- a_5 is positive
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3248_324873


namespace NUMINAMATH_CALUDE_jack_daily_reading_rate_l3248_324872

-- Define the number of books Jack reads in a year
def books_per_year : ℕ := 3285

-- Define the number of days in a year
def days_per_year : ℕ := 365

-- State the theorem
theorem jack_daily_reading_rate :
  books_per_year / days_per_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_jack_daily_reading_rate_l3248_324872


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l3248_324801

theorem roots_of_quadratic (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 6 → 
  x^2 - 10*x + 16 = 0 ∧ y^2 - 10*y + 16 = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l3248_324801


namespace NUMINAMATH_CALUDE_complex_modulus_l3248_324814

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3248_324814


namespace NUMINAMATH_CALUDE_smallest_b_value_l3248_324874

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 10) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 25) :
  ∀ b' : ℕ+, b'.val < b.val → 
    ¬(∃ a' : ℕ+, a'.val - b'.val = 10 ∧ 
      Nat.gcd ((a'.val^3 + b'.val^3) / (a'.val + b'.val)) (a'.val * b'.val) = 25) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3248_324874


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3248_324884

-- Define the triangle DEF
def triangle_DEF : Set (ℝ × ℝ) := sorry

-- Define that the triangle is isosceles
def is_isosceles (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define that the triangle is acute
def is_acute (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define the measure of an angle
def angle_measure (t : Set (ℝ × ℝ)) (v : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem largest_angle_in_special_triangle :
  ∀ (DEF : Set (ℝ × ℝ)),
    is_isosceles DEF →
    is_acute DEF →
    angle_measure DEF (0, 0) = 30 →
    (∃ (v : ℝ × ℝ), v ∈ DEF ∧ angle_measure DEF v = 75 ∧
      ∀ (w : ℝ × ℝ), w ∈ DEF → angle_measure DEF w ≤ 75) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3248_324884


namespace NUMINAMATH_CALUDE_colored_ball_probability_l3248_324819

/-- The probability of drawing a colored ball from an urn -/
theorem colored_ball_probability (total : ℕ) (blue green white : ℕ)
  (h_total : total = blue + green + white)
  (h_blue : blue = 15)
  (h_green : green = 5)
  (h_white : white = 20) :
  (blue + green : ℚ) / total = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_colored_ball_probability_l3248_324819


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3248_324850

theorem sqrt_expression_equality : 
  Real.sqrt 6 * (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6) - abs (3 * Real.sqrt 2 - 6) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3248_324850


namespace NUMINAMATH_CALUDE_D_144_l3248_324869

/-- D(n) represents the number of ways to write a positive integer n as a product of 
    integers strictly greater than 1, where the order of factors matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(144) = 45 -/
theorem D_144 : D 144 = 45 := by sorry

end NUMINAMATH_CALUDE_D_144_l3248_324869


namespace NUMINAMATH_CALUDE_largest_digit_change_l3248_324854

def incorrect_sum : ℕ := 2456
def num1 : ℕ := 641
def num2 : ℕ := 852
def num3 : ℕ := 973

theorem largest_digit_change :
  ∃ (d : ℕ), d ≤ 9 ∧
  (num1 + num2 + (num3 - 10) = incorrect_sum) ∧
  (∀ (d' : ℕ), d' ≤ 9 → 
    (num1 - 10 * d' + num2 + num3 = incorrect_sum ∨
     num1 + (num2 - 10 * d') + num3 = incorrect_sum) → 
    d' ≤ d) ∧
  d = 7 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_change_l3248_324854


namespace NUMINAMATH_CALUDE_stock_price_increase_l3248_324885

theorem stock_price_increase (opening_price : ℝ) (increase_percentage : ℝ) : 
  opening_price = 10 → increase_percentage = 0.5 → 
  opening_price * (1 + increase_percentage) = 15 := by
  sorry

#check stock_price_increase

end NUMINAMATH_CALUDE_stock_price_increase_l3248_324885


namespace NUMINAMATH_CALUDE_volume_of_specific_prism_l3248_324833

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculate the volume of a prism formed by slicing a rectangular solid -/
def volumeOfSlicedPrism (solid : RectangularSolid) (plane : Plane3D) (p1 p2 p3 vertex : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific prism -/
theorem volume_of_specific_prism :
  let solid := RectangularSolid.mk 4 3 3
  let p1 := Point3D.mk 0 0 3
  let p2 := Point3D.mk 4 0 3
  let p3 := Point3D.mk 0 3 1.5
  let vertex := Point3D.mk 4 3 0
  let plane := Plane3D.mk (-0.75) 0.75 1 (-3)
  volumeOfSlicedPrism solid plane p1 p2 p3 vertex = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_prism_l3248_324833


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3248_324860

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x > 2 ∨ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3248_324860


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3248_324853

theorem quadratic_inequality_solution (c : ℝ) : 
  (∃ x ∈ Set.Ioo (-2 : ℝ) 1, x^2 + x - c < 0) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3248_324853


namespace NUMINAMATH_CALUDE_parking_lot_cars_parking_lot_problem_l3248_324890

theorem parking_lot_cars (total_wheels : ℕ) (num_bikes : ℕ) : ℕ :=
  let car_wheels := 4
  let bike_wheels := 2
  let num_cars := (total_wheels - num_bikes * bike_wheels) / car_wheels
  num_cars

theorem parking_lot_problem :
  parking_lot_cars 44 2 = 10 := by sorry

end NUMINAMATH_CALUDE_parking_lot_cars_parking_lot_problem_l3248_324890


namespace NUMINAMATH_CALUDE_simplify_square_root_difference_l3248_324846

theorem simplify_square_root_difference : (Real.sqrt 8 - Real.sqrt (4 + 1/2))^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_difference_l3248_324846


namespace NUMINAMATH_CALUDE_salt_concentration_proof_l3248_324843

/-- Proves that adding 66.67 gallons of 25% salt solution to 100 gallons of pure water results in a 10% salt solution -/
theorem salt_concentration_proof (initial_water : ℝ) (saline_volume : ℝ) (salt_percentage : ℝ) :
  initial_water = 100 →
  saline_volume = 66.67 →
  salt_percentage = 0.25 →
  (salt_percentage * saline_volume) / (initial_water + saline_volume) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_salt_concentration_proof_l3248_324843


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3248_324809

theorem repeating_decimal_sum : ∃ (a b : ℚ), 
  (∀ n : ℕ, a = 2 / 10^n + a / 10^n) ∧ 
  (∀ m : ℕ, b = 3 / 100^m + b / 100^m) ∧ 
  (a + b = 25 / 99) := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3248_324809


namespace NUMINAMATH_CALUDE_min_side_length_with_integer_altitude_l3248_324838

theorem min_side_length_with_integer_altitude (a b c h x y : ℕ) :
  -- Triangle with integer side lengths
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Altitude h divides side b into segments x and y
  (x + y = b) →
  -- Difference between segments is 7
  (y = x + 7) →
  -- Pythagorean theorem for altitude
  (a^2 - y^2 = c^2 - x^2) →
  -- Altitude is an integer
  (h^2 = a^2 - y^2) →
  -- b is the minimum side length
  (∀ b' : ℕ, b' < b → ¬∃ a' c' h' x' y' : ℕ,
    (a' > 0) ∧ (b' > 0) ∧ (c' > 0) ∧
    (x' + y' = b') ∧ (y' = x' + 7) ∧
    (a'^2 - y'^2 = c'^2 - x'^2) ∧
    (h'^2 = a'^2 - y'^2)) →
  -- Conclusion: minimum side length is 25
  b = 25 := by sorry

end NUMINAMATH_CALUDE_min_side_length_with_integer_altitude_l3248_324838


namespace NUMINAMATH_CALUDE_correct_average_l3248_324821

/-- Given 10 numbers with an initial average of 40.2, if one number is 17 greater than
    it should be and another number is 13 instead of 31, then the correct average is 40.3. -/
theorem correct_average (n : ℕ) (initial_avg : ℚ) (error1 error2 : ℚ) (correct2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 17 →
  error2 = 13 →
  correct2 = 31 →
  (n : ℚ) * initial_avg - error1 - error2 + correct2 = n * 40.3 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3248_324821


namespace NUMINAMATH_CALUDE_star_not_associative_l3248_324883

-- Define the set T of non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b

-- Theorem: * is not associative over T
theorem star_not_associative :
  ∃ (a b c : T), star (star a b) c ≠ star a (star b c) := by
  sorry

end NUMINAMATH_CALUDE_star_not_associative_l3248_324883


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_six_implies_product_l3248_324897

theorem sqrt_sum_equals_six_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (15 - x) = 6 →
  (8 + x) * (15 - x) = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_six_implies_product_l3248_324897


namespace NUMINAMATH_CALUDE_total_cost_for_20_products_l3248_324837

/-- The total cost function for producing products -/
def total_cost (fixed_cost marginal_cost : ℝ) (n : ℕ) : ℝ :=
  fixed_cost + marginal_cost * n

/-- Theorem: The total cost for producing 20 products is $16000 -/
theorem total_cost_for_20_products
  (fixed_cost : ℝ)
  (marginal_cost : ℝ)
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200) :
  total_cost fixed_cost marginal_cost 20 = 16000 := by
  sorry

#eval total_cost 12000 200 20

end NUMINAMATH_CALUDE_total_cost_for_20_products_l3248_324837


namespace NUMINAMATH_CALUDE_plates_problem_l3248_324803

theorem plates_problem (total_days : ℕ) (plates_two_people : ℕ) (plates_four_people : ℕ) (total_plates : ℕ) :
  total_days = 7 →
  plates_two_people = 2 →
  plates_four_people = 8 →
  total_plates = 38 →
  ∃ (days_two_people : ℕ),
    days_two_people * plates_two_people + (total_days - days_two_people) * plates_four_people = total_plates ∧
    days_two_people = 3 :=
by sorry

end NUMINAMATH_CALUDE_plates_problem_l3248_324803


namespace NUMINAMATH_CALUDE_three_integers_difference_l3248_324820

theorem three_integers_difference (x y z : ℕ+) 
  (sum_xy : x + y = 998)
  (sum_xz : x + z = 1050)
  (sum_yz : y + z = 1234) :
  max x (max y z) - min x (min y z) = 236 := by
sorry

end NUMINAMATH_CALUDE_three_integers_difference_l3248_324820


namespace NUMINAMATH_CALUDE_train_speed_l3248_324812

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 360) 
  (h2 : bridge_length = 140) (h3 : time = 40) : 
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3248_324812


namespace NUMINAMATH_CALUDE_b_payment_l3248_324893

theorem b_payment (total_payment : ℚ) (ac_portion : ℚ) (b_payment : ℚ) : 
  total_payment = 529 →
  ac_portion = 19/23 →
  b_payment = (1 - ac_portion) * total_payment →
  b_payment = 92 := by sorry

end NUMINAMATH_CALUDE_b_payment_l3248_324893


namespace NUMINAMATH_CALUDE_wheat_mixture_problem_arun_wheat_problem_l3248_324825

/-- Calculates the rate of the second wheat purchase given the conditions of Arun's wheat mixture problem -/
theorem wheat_mixture_problem (first_quantity : ℝ) (first_rate : ℝ) (second_quantity : ℝ) (selling_rate : ℝ) (profit_percentage : ℝ) : ℝ :=
  let total_quantity := first_quantity + second_quantity
  let first_cost := first_quantity * first_rate
  let total_selling_price := total_quantity * selling_rate
  let total_cost := total_selling_price / (1 + profit_percentage / 100)
  (total_cost - first_cost) / second_quantity

/-- The rate of the second wheat purchase in Arun's problem is 14.25 -/
theorem arun_wheat_problem : 
  wheat_mixture_problem 30 11.50 20 15.75 25 = 14.25 := by
  sorry

end NUMINAMATH_CALUDE_wheat_mixture_problem_arun_wheat_problem_l3248_324825


namespace NUMINAMATH_CALUDE_infinitely_many_increasing_largest_prime_factors_l3248_324804

/-- h(n) denotes the largest prime factor of the natural number n -/
def h (n : ℕ) : ℕ := sorry

/-- There exist infinitely many natural numbers n such that 
    the largest prime factor of n is less than the largest prime factor of n+1, 
    which is less than the largest prime factor of n+2 -/
theorem infinitely_many_increasing_largest_prime_factors :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, h n < h (n + 1) ∧ h (n + 1) < h (n + 2) := by sorry

end NUMINAMATH_CALUDE_infinitely_many_increasing_largest_prime_factors_l3248_324804


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3248_324836

theorem quadratic_inequality (x : ℝ) : x^2 - 4*x - 21 < 0 ↔ -3 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3248_324836


namespace NUMINAMATH_CALUDE_odd_mult_odd_is_odd_l3248_324877

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def P : Set ℕ := {n : ℕ | is_odd n}

theorem odd_mult_odd_is_odd (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P := by
  sorry

end NUMINAMATH_CALUDE_odd_mult_odd_is_odd_l3248_324877


namespace NUMINAMATH_CALUDE_heroes_on_large_sheets_l3248_324839

/-- Represents the number of pictures that can be drawn on a sheet of paper. -/
structure SheetCapacity where
  small : ℕ
  large : ℕ
  large_twice_small : large = 2 * small

/-- Represents the distribution of pictures drawn during the lunch break. -/
structure PictureDistribution where
  total : ℕ
  on_back : ℕ
  on_front : ℕ
  total_sum : total = on_back + on_front
  half_on_back : on_back = total / 2

/-- Represents the time spent drawing during the lunch break. -/
structure DrawingTime where
  break_duration : ℕ
  time_per_drawing : ℕ
  time_left : ℕ
  total_drawing_time : ℕ
  drawing_time_calc : total_drawing_time = break_duration - time_left

/-- The main theorem to prove. -/
theorem heroes_on_large_sheets
  (sheet_capacity : SheetCapacity)
  (picture_dist : PictureDistribution)
  (drawing_time : DrawingTime)
  (h1 : picture_dist.total = 20)
  (h2 : drawing_time.break_duration = 75)
  (h3 : drawing_time.time_per_drawing = 5)
  (h4 : drawing_time.time_left = 5)
  : ∃ (n : ℕ), n = 6 ∧ n * sheet_capacity.small = picture_dist.on_front / 2 :=
sorry

end NUMINAMATH_CALUDE_heroes_on_large_sheets_l3248_324839


namespace NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l3248_324867

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit inside two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ := sorry

/-- The specific configuration of cones described in the problem -/
def problemCones : IntersectingCones :=
  { cone1 := { baseRadius := 3, height := 8 },
    cone2 := { baseRadius := 3, height := 8 },
    intersectionDistance := 3 }

theorem max_sphere_in_intersecting_cones :
  maxSphereRadiusSquared problemCones = 225 / 73 := by sorry

end NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l3248_324867


namespace NUMINAMATH_CALUDE_tan_sum_specific_angles_l3248_324870

theorem tan_sum_specific_angles (α β : Real) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_specific_angles_l3248_324870


namespace NUMINAMATH_CALUDE_simplify_fraction_l3248_324856

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3248_324856


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l3248_324880

theorem larger_number_of_pair (a b : ℝ) (h1 : a - b = 5) (h2 : a + b = 37) :
  max a b = 21 := by sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l3248_324880


namespace NUMINAMATH_CALUDE_min_vertical_distance_l3248_324894

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

-- Define the vertical distance between the two functions
def verticalDistance (x : ℝ) : ℝ := |f x - g x|

-- State the theorem
theorem min_vertical_distance :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ (x : ℝ), verticalDistance x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l3248_324894


namespace NUMINAMATH_CALUDE_unique_numbers_with_lcm_conditions_l3248_324811

theorem unique_numbers_with_lcm_conditions :
  ∃! (x y z : ℕ),
    x > y ∧ x > z ∧
    Nat.lcm x y = 200 ∧
    Nat.lcm y z = 300 ∧
    Nat.lcm x z = 120 ∧
    x = 40 ∧ y = 25 ∧ z = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_numbers_with_lcm_conditions_l3248_324811


namespace NUMINAMATH_CALUDE_candy_bar_earnings_difference_l3248_324840

/-- The problem of calculating the difference in earnings between Tina and Marvin from selling candy bars. -/
theorem candy_bar_earnings_difference : 
  let candy_bar_price : ℕ := 2
  let marvin_sales : ℕ := 35
  let tina_sales : ℕ := 3 * marvin_sales
  let marvin_earnings : ℕ := candy_bar_price * marvin_sales
  let tina_earnings : ℕ := candy_bar_price * tina_sales
  tina_earnings - marvin_earnings = 140 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_earnings_difference_l3248_324840


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3248_324864

def q (x : ℚ) : ℚ := -20/93 * x^3 - 110/93 * x^2 - 372/93 * x - 525/93

theorem cubic_polynomial_satisfies_conditions :
  q 1 = -11 ∧ q 2 = -15 ∧ q 3 = -25 ∧ q 5 = -65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3248_324864


namespace NUMINAMATH_CALUDE_village_population_l3248_324892

def initial_population : ℝ → ℝ → ℝ → Prop :=
  fun P rate years =>
    P * (1 - rate)^years = 4860

theorem village_population :
  ∃ P : ℝ, initial_population P 0.1 2 ∧ P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3248_324892


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3248_324881

/-- Given two parallel vectors a and b in R², prove that 3a + 2b equals (-1, -2) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  (a.1 * b.2 = a.2 * b.1) →  -- Parallel condition
  (3 * a.1 + 2 * b.1 = -1 ∧ 3 * a.2 + 2 * b.2 = -2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3248_324881


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l3248_324855

theorem six_digit_divisible_by_7_8_9 :
  ∃ (n₁ n₂ : ℕ),
    523000 ≤ n₁ ∧ n₁ < 524000 ∧
    523000 ≤ n₂ ∧ n₂ < 524000 ∧
    n₁ ≠ n₂ ∧
    n₁ % 7 = 0 ∧ n₁ % 8 = 0 ∧ n₁ % 9 = 0 ∧
    n₂ % 7 = 0 ∧ n₂ % 8 = 0 ∧ n₂ % 9 = 0 ∧
    n₁ = 523152 ∧ n₂ = 523656 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l3248_324855


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3248_324835

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (a^2 - 2*a) + (a^2 - a - 2)*I
  (z.re = 0) → (a = 0 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3248_324835


namespace NUMINAMATH_CALUDE_triangle_side_length_l3248_324852

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →
  (B = Real.pi / 3) →
  (a^2 + c^2 = 3 * a * c) →
  (b = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3248_324852


namespace NUMINAMATH_CALUDE_cave_depth_l3248_324899

/-- The depth of the cave given the current depth and remaining distance -/
theorem cave_depth (current_depth remaining_distance : ℕ) 
  (h1 : current_depth = 588)
  (h2 : remaining_distance = 386) : 
  current_depth + remaining_distance = 974 := by
  sorry

end NUMINAMATH_CALUDE_cave_depth_l3248_324899


namespace NUMINAMATH_CALUDE_tangency_points_form_cyclic_quadrilateral_l3248_324863

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define a point of tangency between two circles
def tangency_point (c1 c2 : Circle) : ℝ × ℝ :=
  sorry

-- Define the property of a quadrilateral being cyclic
def is_cyclic_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem tangency_points_form_cyclic_quadrilateral 
  (S1 S2 S3 S4 : Circle)
  (h12 : externally_tangent S1 S2)
  (h23 : externally_tangent S2 S3)
  (h34 : externally_tangent S3 S4)
  (h41 : externally_tangent S4 S1) :
  let p1 := tangency_point S1 S2
  let p2 := tangency_point S2 S3
  let p3 := tangency_point S3 S4
  let p4 := tangency_point S4 S1
  is_cyclic_quadrilateral p1 p2 p3 p4 :=
by
  sorry

end NUMINAMATH_CALUDE_tangency_points_form_cyclic_quadrilateral_l3248_324863


namespace NUMINAMATH_CALUDE_cereal_box_servings_l3248_324859

theorem cereal_box_servings (total_cups : ℕ) (serving_size : ℕ) (h1 : total_cups = 18) (h2 : serving_size = 2) :
  total_cups / serving_size = 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_servings_l3248_324859


namespace NUMINAMATH_CALUDE_quadrilateral_area_in_possible_areas_all_possible_areas_achievable_l3248_324849

/-- Represents a point on the side of a square --/
inductive SidePoint
| A1 | A2 | A3  -- Points on side AB
| B1 | B2 | B3  -- Points on side BC
| C1 | C2 | C3  -- Points on side CD
| D1 | D2 | D3  -- Points on side DA

/-- Represents a quadrilateral formed by choosing points from each side of a square --/
structure Quadrilateral :=
  (p1 : SidePoint)
  (p2 : SidePoint)
  (p3 : SidePoint)
  (p4 : SidePoint)

/-- Calculates the area of a quadrilateral formed by choosing points from each side of a square --/
def area (q : Quadrilateral) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The set of possible areas for quadrilaterals formed in the given square --/
def possible_areas : Set ℝ := {6, 7, 7.5, 8, 8.5, 9, 10}

/-- Theorem stating that the area of any quadrilateral formed in the given square
    must be one of the values in the possible_areas set --/
theorem quadrilateral_area_in_possible_areas (q : Quadrilateral) :
  area q ∈ possible_areas :=
sorry

/-- Theorem stating that every value in the possible_areas set
    is achievable by some quadrilateral in the given square --/
theorem all_possible_areas_achievable :
  ∀ a ∈ possible_areas, ∃ q : Quadrilateral, area q = a :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_in_possible_areas_all_possible_areas_achievable_l3248_324849


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l3248_324830

theorem percentage_of_red_non_honda_cars 
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (honda_red_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 9000)
  (h2 : honda_cars = 5000)
  (h3 : honda_red_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100)
  : (↑(total_cars * total_red_ratio - honda_cars * honda_red_ratio) / ↑(total_cars - honda_cars) : ℚ) = 225 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l3248_324830


namespace NUMINAMATH_CALUDE_power_three_mod_thirteen_l3248_324832

theorem power_three_mod_thirteen : 3^21 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_thirteen_l3248_324832


namespace NUMINAMATH_CALUDE_polygon_sides_diagonals_l3248_324845

theorem polygon_sides_diagonals : ∃ (n : ℕ), n > 2 ∧ 3 * n * (n * (n - 3)) = 300 := by
  use 10
  sorry

end NUMINAMATH_CALUDE_polygon_sides_diagonals_l3248_324845


namespace NUMINAMATH_CALUDE_cube_decomposition_smallest_l3248_324800

theorem cube_decomposition_smallest (m : ℕ) (h1 : m ≥ 2) : 
  (m^2 - m + 1 = 73) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_smallest_l3248_324800


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l3248_324858

theorem triangle_abc_problem (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  2 * Real.sin A ^ 2 + 3 * Real.cos (B + C) = 0 →
  S = 5 * Real.sqrt 3 →
  a = Real.sqrt 21 →
  A = π / 3 ∧ b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l3248_324858


namespace NUMINAMATH_CALUDE_number_difference_l3248_324805

theorem number_difference (L S : ℕ) (h1 : L = 1614) (h2 : L = 6 * S + 15) : L - S = 1348 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3248_324805


namespace NUMINAMATH_CALUDE_movie_change_theorem_l3248_324866

/-- The change received by two sisters after buying movie tickets -/
def change_received (ticket_price : ℕ) (money_brought : ℕ) : ℕ :=
  money_brought - (2 * ticket_price)

/-- Theorem: The change received is $9 when tickets cost $8 each and the sisters brought $25 -/
theorem movie_change_theorem : change_received 8 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_change_theorem_l3248_324866


namespace NUMINAMATH_CALUDE_area_of_fourth_square_l3248_324831

/-- Given two right triangles PQR and PRS sharing a common hypotenuse PR,
    where the squares on PQ, QR, and RS have areas 25, 49, and 64 square units respectively,
    prove that the area of the square on PS is 10 square units. -/
theorem area_of_fourth_square (P Q R S : ℝ × ℝ) : 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 25 →
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 49 →
  (R.1 - S.1)^2 + (R.2 - S.2)^2 = 64 →
  (P.1 - S.1)^2 + (P.2 - S.2)^2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_area_of_fourth_square_l3248_324831


namespace NUMINAMATH_CALUDE_third_term_is_six_l3248_324895

-- Define the sum of the first n terms
def S (n : ℕ) : ℕ := n^2 + n

-- Define the arithmetic sequence
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem third_term_is_six : a 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_six_l3248_324895


namespace NUMINAMATH_CALUDE_coal_reserve_duration_l3248_324847

theorem coal_reserve_duration 
  (Q a x : ℝ) 
  (h₁ : 0 < Q) 
  (h₂ : 0 < a) 
  (h₃ : 0 < x) 
  (h₄ : x < a) : 
  ∃ y : ℝ, y = Q / (a - x) - Q / a :=
by sorry

end NUMINAMATH_CALUDE_coal_reserve_duration_l3248_324847


namespace NUMINAMATH_CALUDE_power_of_i_sum_l3248_324878

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_sum : i^123 - i^321 + i^432 = -2*i + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_sum_l3248_324878


namespace NUMINAMATH_CALUDE_count_negative_rationals_l3248_324857

def rational_set : Finset ℚ := {-1/2, 5, 0, -(-3), -2, -|-25|}

theorem count_negative_rationals : 
  (rational_set.filter (λ x => x < 0)).card = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_rationals_l3248_324857


namespace NUMINAMATH_CALUDE_quadratic_properties_l3248_324848

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a < 0)
  (h_root : f a b c (-1) = 0)
  (h_sym : -b / (2 * a) = 1) :
  (a - b + c = 0) ∧
  (∀ m : ℝ, f a b c m ≤ -4 * a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a b c x₁ = -1 → f a b c x₂ = -1 → x₁ < -1 ∧ x₂ > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3248_324848


namespace NUMINAMATH_CALUDE_traveler_time_difference_l3248_324876

/-- Proof of the time difference between two travelers meeting at a point -/
theorem traveler_time_difference 
  (speed_A speed_B meeting_distance : ℝ) 
  (h1 : speed_A > 0)
  (h2 : speed_B > speed_A)
  (h3 : meeting_distance > 0) :
  meeting_distance / speed_A - meeting_distance / speed_B = 7 :=
by sorry

end NUMINAMATH_CALUDE_traveler_time_difference_l3248_324876


namespace NUMINAMATH_CALUDE_unique_poly_pair_l3248_324806

/-- A polynomial of degree 3 -/
def Poly3 (R : Type*) [CommRing R] := R → R

/-- The evaluation of a polynomial at a point -/
def eval (p : Poly3 ℝ) (x : ℝ) : ℝ := p x

/-- The composition of two polynomials -/
def comp (p q : Poly3 ℝ) : Poly3 ℝ := λ x ↦ p (q x)

/-- The cube of a polynomial -/
def cube (p : Poly3 ℝ) : Poly3 ℝ := λ x ↦ (p x)^3

theorem unique_poly_pair (f g : Poly3 ℝ) 
  (h1 : f ≠ g)
  (h2 : ∀ x, eval (comp f f) x = eval (cube g) x)
  (h3 : ∀ x, eval (comp f g) x = eval (cube f) x)
  (h4 : eval f 0 = 1) :
  (∀ x, f x = (1 - x)^3) ∧ (∀ x, g x = (x - 1)^3 + 1) := by
  sorry


end NUMINAMATH_CALUDE_unique_poly_pair_l3248_324806


namespace NUMINAMATH_CALUDE_johns_height_l3248_324828

/-- Given the heights of John, Lena, and Rebeca, prove John's height is 152 cm -/
theorem johns_height (john lena rebeca : ℕ) 
  (h1 : john = lena + 15)
  (h2 : john + 6 = rebeca)
  (h3 : lena + rebeca = 295) :
  john = 152 := by sorry

end NUMINAMATH_CALUDE_johns_height_l3248_324828


namespace NUMINAMATH_CALUDE_gumballs_per_package_l3248_324802

theorem gumballs_per_package (total_gumballs : ℕ) (total_boxes : ℕ) 
  (h1 : total_gumballs = 20) 
  (h2 : total_boxes = 4) 
  (h3 : total_gumballs > 0) 
  (h4 : total_boxes > 0) : 
  (total_gumballs / total_boxes : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_per_package_l3248_324802


namespace NUMINAMATH_CALUDE_firefighter_remaining_money_l3248_324813

-- Define the firefighter's financial parameters
def hourly_rate : ℚ := 30
def weekly_hours : ℚ := 48
def food_expense : ℚ := 500
def tax_expense : ℚ := 1000
def weeks_per_month : ℚ := 4

-- Calculate weekly and monthly earnings
def weekly_earnings : ℚ := hourly_rate * weekly_hours
def monthly_earnings : ℚ := weekly_earnings * weeks_per_month

-- Calculate monthly rent
def monthly_rent : ℚ := monthly_earnings / 3

-- Calculate total monthly expenses
def total_monthly_expenses : ℚ := monthly_rent + food_expense + tax_expense

-- Calculate remaining money after expenses
def remaining_money : ℚ := monthly_earnings - total_monthly_expenses

-- Theorem to prove
theorem firefighter_remaining_money :
  remaining_money = 2340 := by
  sorry

end NUMINAMATH_CALUDE_firefighter_remaining_money_l3248_324813


namespace NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_l3248_324886

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem perpendicular_to_same_plane_implies_parallel 
  (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular l1 p) (h2 : perpendicular l2 p) : 
  parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_l3248_324886


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l3248_324862

theorem geometric_sequence_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_geometric : b^2 = a * c) : 
  a^2 + b^2 + c^2 > (a - b + c)^2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l3248_324862


namespace NUMINAMATH_CALUDE_stock_percentage_calculation_l3248_324827

/-- Calculates the percentage of a stock given its yield and quoted price. -/
theorem stock_percentage_calculation (yield : ℝ) (quote : ℝ) :
  yield = 10 →
  quote = 160 →
  let face_value := 100
  let market_price := quote * face_value / 100
  let annual_income := yield * face_value / 100
  let stock_percentage := annual_income / market_price * 100
  stock_percentage = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_calculation_l3248_324827


namespace NUMINAMATH_CALUDE_power_zero_equals_one_l3248_324868

theorem power_zero_equals_one (x : ℝ) (hx : x ≠ 0) : x ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_equals_one_l3248_324868


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l3248_324879

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem for part (1)
theorem solution_part1 : {x : ℝ | f x > 3 - 4*x} = {x : ℝ | x > 3/5} := by sorry

-- Theorem for part (2)
theorem solution_part2 : 
  (∀ x : ℝ, f x + |1 - x| ≥ 6*m^2 - 5*m) → 
  -1/6 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l3248_324879


namespace NUMINAMATH_CALUDE_missing_donuts_percentage_l3248_324891

def initial_donuts : ℕ := 30
def remaining_donuts : ℕ := 9

theorem missing_donuts_percentage :
  (initial_donuts - remaining_donuts : ℚ) / initial_donuts * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_missing_donuts_percentage_l3248_324891


namespace NUMINAMATH_CALUDE_stifel_conjecture_counterexample_l3248_324842

theorem stifel_conjecture_counterexample : ∃ n : ℕ, ¬ Nat.Prime (2^(2*n + 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_stifel_conjecture_counterexample_l3248_324842


namespace NUMINAMATH_CALUDE_book_pages_theorem_l3248_324851

theorem book_pages_theorem (P : ℕ) 
  (h1 : P / 2 + P / 4 + P / 6 + 20 = P) : P = 240 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l3248_324851


namespace NUMINAMATH_CALUDE_equation_solution_l3248_324841

theorem equation_solution :
  ∃ x : ℝ, (3 * x + 9 = 0) ∧ (x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3248_324841


namespace NUMINAMATH_CALUDE_inequality_solution_l3248_324815

/-- The solution set of the inequality |ax-2|+|ax-a| ≥ 2 when a = 1 -/
def solution_set_a1 : Set ℝ := {x | x ≥ 2.5 ∨ x ≤ 0.5}

/-- The inequality |ax-2|+|ax-a| ≥ 2 -/
def inequality (a x : ℝ) : Prop := |a*x - 2| + |a*x - a| ≥ 2

theorem inequality_solution :
  (∀ x, inequality 1 x ↔ x ∈ solution_set_a1) ∧
  (∀ a, a > 0 → (∀ x, inequality a x) ↔ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3248_324815


namespace NUMINAMATH_CALUDE_solve_sues_library_problem_l3248_324817

/-- Represents the number of books and movies Sue has --/
structure LibraryItems where
  books : ℕ
  movies : ℕ

/-- The problem statement about Sue's library items --/
def sues_library_problem (initial_items : LibraryItems) 
  (books_checked_out : ℕ) (final_total : ℕ) : Prop :=
  let movies_returned := initial_items.movies / 3
  let final_movies := initial_items.movies - movies_returned
  let total_books_before_return := initial_items.books + books_checked_out
  let final_books := final_total - final_movies
  let books_returned := total_books_before_return - final_books
  books_returned = 8

/-- Theorem stating the solution to Sue's library problem --/
theorem solve_sues_library_problem :
  sues_library_problem ⟨15, 6⟩ 9 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_sues_library_problem_l3248_324817


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l3248_324808

theorem linear_systems_solutions :
  -- System 1
  let system1 (x y : ℚ) := (y = x - 5) ∧ (3 * x - y = 8)
  let solution1 := (3/2, -7/2)
  -- System 2
  let system2 (x y : ℚ) := (3 * x - 2 * y = 1) ∧ (7 * x + 4 * y = 11)
  let solution2 := (1, 1)
  -- Proof statements
  (∃! p : ℚ × ℚ, system1 p.1 p.2 ∧ p = solution1) ∧
  (∃! q : ℚ × ℚ, system2 q.1 q.2 ∧ q = solution2) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l3248_324808


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3248_324882

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 132 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 132 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3248_324882


namespace NUMINAMATH_CALUDE_f_continuous_at_2_delta_epsilon_relation_l3248_324816

def f (x : ℝ) : ℝ := -3 * x^2 - 5

theorem f_continuous_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 3 ∧
    ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_2_delta_epsilon_relation_l3248_324816


namespace NUMINAMATH_CALUDE_log_28_5_l3248_324861

theorem log_28_5 (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 7 = b) :
  (Real.log 5) / (Real.log 28) = (1 - a) / (2 * a + b) := by
  sorry

end NUMINAMATH_CALUDE_log_28_5_l3248_324861


namespace NUMINAMATH_CALUDE_fraction_division_problem_solution_l3248_324824

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem problem_solution :
  (5 : ℚ) / 6 / ((11 : ℚ) / 12) = 10 / 11 := by sorry

end NUMINAMATH_CALUDE_fraction_division_problem_solution_l3248_324824


namespace NUMINAMATH_CALUDE_inverse_proportional_properties_l3248_324887

/-- Given two inverse proportional functions y = k/x and y = 1/x, where k > 0,
    and a point P(a, k/a) on y = k/x, with a > 0, we define:
    C(a, 0), A(a, 1/a), D(0, k/a), B(a/k, k/a) -/
theorem inverse_proportional_properties (k a : ℝ) (hk : k > 0) (ha : a > 0) :
  let P := (a, k / a)
  let C := (a, 0)
  let A := (a, 1 / a)
  let D := (0, k / a)
  let B := (a / k, k / a)
  let triangle_area (p q r : ℝ × ℝ) := (abs ((p.1 - r.1) * (q.2 - r.2) - (q.1 - r.1) * (p.2 - r.2))) / 2
  let quadrilateral_area (p q r s : ℝ × ℝ) := triangle_area p q r + triangle_area p r s
  -- 1. The areas of triangles ODB and OCA are equal to 1/2
  (triangle_area (0, 0) D B = 1 / 2 ∧ triangle_area (0, 0) C A = 1 / 2) ∧
  -- 2. The area of quadrilateral OAPB is equal to k - 1
  (quadrilateral_area (0, 0) A P B = k - 1) ∧
  -- 3. If k = 2, then A is the midpoint of PC and B is the midpoint of PD
  (k = 2 → (A.2 - C.2 = P.2 - A.2 ∧ B.1 - D.1 = P.1 - B.1)) := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportional_properties_l3248_324887


namespace NUMINAMATH_CALUDE_power_five_minus_self_divisible_by_five_l3248_324844

theorem power_five_minus_self_divisible_by_five (a : ℤ) : ∃ k : ℤ, a^5 - a = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_power_five_minus_self_divisible_by_five_l3248_324844


namespace NUMINAMATH_CALUDE_gold_bars_lost_l3248_324829

theorem gold_bars_lost (initial_bars : ℕ) (num_friends : ℕ) (bars_per_friend : ℕ) : 
  initial_bars = 100 →
  num_friends = 4 →
  bars_per_friend = 20 →
  initial_bars - (num_friends * bars_per_friend) = 20 := by
  sorry

end NUMINAMATH_CALUDE_gold_bars_lost_l3248_324829


namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_exists_l3248_324818

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem 1: Inequality solution
theorem inequality_solution :
  ∀ x : ℝ, f (2 * x) ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem 2: Minimum value
theorem minimum_value :
  ∀ a b : ℝ, a + b = 2 → f (a^2) + f (b^2) ≥ 2 :=
sorry

-- Theorem 3: Existence of minimum
theorem minimum_exists :
  ∃ a b : ℝ, a + b = 2 ∧ f (a^2) + f (b^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_exists_l3248_324818


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3248_324889

/-- Given a rectangle with area 800 cm² and length twice its width, prove its perimeter is 120 cm. -/
theorem rectangle_perimeter (width : ℝ) (length : ℝ) :
  width > 0 →
  length > 0 →
  length = 2 * width →
  width * length = 800 →
  2 * (width + length) = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3248_324889


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3248_324865

theorem cubic_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 6 * y^(1/3) - 3 * (y / y^(2/3)) = 12 + 2 * y^(1/3) ∧ y = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3248_324865


namespace NUMINAMATH_CALUDE_no_prime_between_100_110_congruent_3_mod_6_l3248_324810

theorem no_prime_between_100_110_congruent_3_mod_6 : ¬ ∃ n : ℕ, 
  Nat.Prime n ∧ 100 < n ∧ n < 110 ∧ n % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_between_100_110_congruent_3_mod_6_l3248_324810


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l3248_324822

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (hn : n > 0) 
  (h_divisible : 72 ∣ n^2) : 
  ∀ m : ℕ, m ∣ n → m ≤ 12 ∧ 12 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l3248_324822


namespace NUMINAMATH_CALUDE_marble_probability_l3248_324896

/-- Represents a box containing marbles -/
structure Box where
  total : ℕ
  black : ℕ
  white : ℕ
  sum_constraint : total = black + white

/-- The probability of drawing a specific color from a box -/
def probability (box : Box) (color : ℕ) : ℚ :=
  color / box.total

theorem marble_probability (box1 box2 : Box) 
  (total_constraint : box1.total + box2.total = 30)
  (black_prob : probability box1 box1.black * probability box2 box2.black = 1/2) :
  probability box1 box1.white * probability box2 box2.white = 0 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3248_324896


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3248_324826

theorem arithmetic_calculations :
  (-(1/8 : ℚ) + 3/4 - (-(1/4)) - 5/8 = 1/4) ∧
  (-3^2 + 5 * (-6) - (-4)^2 / (-8) = -37) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3248_324826


namespace NUMINAMATH_CALUDE_transform_trig_function_l3248_324898

/-- Given a function f(x) = (√2/2)(sin x + cos x), 
    applying a horizontal stretch by a factor of 2 
    and a left shift by π/2 results in cos(x/2) -/
theorem transform_trig_function : 
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f x = (Real.sqrt 2 / 2) * (Real.sin x + Real.cos x)) ∧
    (∀ x, g x = f (x / 2 + π / 2)) ∧
    (∀ x, g x = Real.cos (x / 2)) := by
  sorry

end NUMINAMATH_CALUDE_transform_trig_function_l3248_324898


namespace NUMINAMATH_CALUDE_ratio_to_eleven_l3248_324875

theorem ratio_to_eleven : ∃ x : ℚ, (5 : ℚ) / 1 = x / 11 ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_eleven_l3248_324875


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l3248_324834

def prob_rain_friday : ℝ := 0.30
def prob_rain_saturday : ℝ := 0.45
def prob_rain_sunday : ℝ := 0.55

theorem weekend_rain_probability : 
  let prob_no_rain_friday := 1 - prob_rain_friday
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let prob_no_rain_weekend := prob_no_rain_friday * prob_no_rain_saturday * prob_no_rain_sunday
  1 - prob_no_rain_weekend = 0.82675 := by
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l3248_324834


namespace NUMINAMATH_CALUDE_kittens_per_female_cat_l3248_324888

theorem kittens_per_female_cat 
  (total_adult_cats : ℕ)
  (female_ratio : ℚ)
  (sold_kittens : ℕ)
  (kitten_ratio_after_sale : ℚ)
  (h1 : total_adult_cats = 6)
  (h2 : female_ratio = 1/2)
  (h3 : sold_kittens = 9)
  (h4 : kitten_ratio_after_sale = 67/100) :
  ∃ (kittens_per_female : ℕ),
    kittens_per_female = 7 ∧
    (female_ratio * total_adult_cats : ℚ) * kittens_per_female = 
      (1 - kitten_ratio_after_sale) * 
        ((total_adult_cats : ℚ) / (1 - kitten_ratio_after_sale) - total_adult_cats) +
      sold_kittens :=
by
  sorry

end NUMINAMATH_CALUDE_kittens_per_female_cat_l3248_324888


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3248_324871

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) := by sorry

theorem negation_of_quadratic_inequality : 
  (¬∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3248_324871
