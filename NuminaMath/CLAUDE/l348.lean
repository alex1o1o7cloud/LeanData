import Mathlib

namespace NUMINAMATH_CALUDE_l₂_passes_through_fixed_point_l348_34893

/-- A line in 2D space defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetryPoint : ℝ × ℝ := (2, 1)

/-- Line l₁ defined as y = k(x - 4) -/
def l₁ (k : ℝ) : Line :=
  { slope := k, point := (4, 0) }

/-- Reflect a point about the symmetry point -/
def reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * symmetryPoint.1 - p.1, 2 * symmetryPoint.2 - p.2)

/-- Line l₂ symmetric to l₁ about the symmetry point -/
def l₂ (k : ℝ) : Line :=
  { slope := -k, point := reflect (l₁ k).point }

theorem l₂_passes_through_fixed_point :
  ∀ k : ℝ, (l₂ k).point = (0, 2) := by sorry

end NUMINAMATH_CALUDE_l₂_passes_through_fixed_point_l348_34893


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l348_34849

theorem ice_cream_scoops (oli_scoops : ℕ) (victoria_scoops : ℕ) : 
  oli_scoops = 4 → 
  victoria_scoops = 2 * oli_scoops → 
  victoria_scoops - oli_scoops = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l348_34849


namespace NUMINAMATH_CALUDE_area_isosceles_right_triangle_radius_circumcircle_isosceles_right_triangle_l348_34854

/-- A right triangle with two equal angles and hypotenuse 8√2 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 8√2 -/
  hypotenuse_eq : hypotenuse = 8 * Real.sqrt 2

/-- The area of an isosceles right triangle with hypotenuse 8√2 is 32 -/
theorem area_isosceles_right_triangle (t : IsoscelesRightTriangle) :
    (1 / 2 : ℝ) * t.hypotenuse^2 / 2 = 32 := by sorry

/-- The radius of the circumcircle of an isosceles right triangle with hypotenuse 8√2 is 4√2 -/
theorem radius_circumcircle_isosceles_right_triangle (t : IsoscelesRightTriangle) :
    t.hypotenuse / 2 = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_area_isosceles_right_triangle_radius_circumcircle_isosceles_right_triangle_l348_34854


namespace NUMINAMATH_CALUDE_numbers_satisfying_conditions_l348_34895

def ends_with_196 (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 1000 * x + 196

def decreases_by_integer_factor (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ n / k = n - 196

def satisfies_conditions (n : ℕ) : Prop :=
  ends_with_196 n ∧ decreases_by_integer_factor n

theorem numbers_satisfying_conditions :
  {n : ℕ | satisfies_conditions n} = {1196, 2196, 4196, 7196, 14196, 49196, 98196} :=
by sorry

end NUMINAMATH_CALUDE_numbers_satisfying_conditions_l348_34895


namespace NUMINAMATH_CALUDE_cube_rect_surface_area_ratio_l348_34844

theorem cube_rect_surface_area_ratio (a b : ℝ) (h : a > 0) :
  2 * a^2 + 4 * a * b = 0.6 * (6 * a^2) → b = 0.6 * a := by
  sorry

end NUMINAMATH_CALUDE_cube_rect_surface_area_ratio_l348_34844


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l348_34888

def small_bottle_capacity : ℝ := 35
def large_bottle_capacity : ℝ := 500

theorem min_bottles_to_fill (small_cap large_cap : ℝ) (h1 : small_cap = small_bottle_capacity) (h2 : large_cap = large_bottle_capacity) :
  ⌈large_cap / small_cap⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l348_34888


namespace NUMINAMATH_CALUDE_unique_stickers_count_l348_34800

def emily_stickers : ℕ := 22
def mia_unique_stickers : ℕ := 10
def shared_stickers : ℕ := 12

theorem unique_stickers_count :
  (emily_stickers - shared_stickers) + mia_unique_stickers = 20 :=
by sorry

end NUMINAMATH_CALUDE_unique_stickers_count_l348_34800


namespace NUMINAMATH_CALUDE_sin_sum_alpha_beta_l348_34883

theorem sin_sum_alpha_beta (α β : Real) 
  (h1 : 13 * Real.sin α + 5 * Real.cos β = 9)
  (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_alpha_beta_l348_34883


namespace NUMINAMATH_CALUDE_total_cost_5_and_5_l348_34843

/-- The cost of a single room in yuan -/
def single_room_cost : ℝ := sorry

/-- The cost of a double room in yuan -/
def double_room_cost : ℝ := sorry

/-- The total cost of 3 single rooms and 6 double rooms is 1020 yuan -/
axiom cost_equation_1 : 3 * single_room_cost + 6 * double_room_cost = 1020

/-- The total cost of 1 single room and 5 double rooms is 700 yuan -/
axiom cost_equation_2 : single_room_cost + 5 * double_room_cost = 700

/-- The theorem states that the total cost of 5 single rooms and 5 double rooms is 1100 yuan -/
theorem total_cost_5_and_5 : 5 * single_room_cost + 5 * double_room_cost = 1100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_5_and_5_l348_34843


namespace NUMINAMATH_CALUDE_lenas_muffins_l348_34841

/-- Represents the cost of a single item -/
structure ItemCost where
  cake : ℚ
  muffin : ℚ
  bagel : ℚ

/-- Represents a purchase of items -/
structure Purchase where
  cakes : ℕ
  muffins : ℕ
  bagels : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (cost : ItemCost) (purchase : Purchase) : ℚ :=
  cost.cake * purchase.cakes + cost.muffin * purchase.muffins + cost.bagel * purchase.bagels

/-- The main theorem to prove -/
theorem lenas_muffins (cost : ItemCost) : 
  let petya := Purchase.mk 1 2 3
  let anya := Purchase.mk 3 0 1
  let kolya := Purchase.mk 0 6 0
  let lena := Purchase.mk 2 0 2
  totalCost cost petya = totalCost cost anya ∧ 
  totalCost cost anya = totalCost cost kolya →
  ∃ n : ℕ, totalCost cost lena = totalCost cost (Purchase.mk 0 n 0) ∧ n = 5 := by
  sorry


end NUMINAMATH_CALUDE_lenas_muffins_l348_34841


namespace NUMINAMATH_CALUDE_range_of_a_for_intersection_l348_34815

theorem range_of_a_for_intersection (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ 
   Real.cos (Real.pi * x₁) = 2^x₂ * a - 1/2) ↔ 
  a ∈ Set.Icc (-1/2) 0 ∪ Set.Ioc 0 (3/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_intersection_l348_34815


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l348_34879

theorem fraction_equals_zero (x : ℝ) : (x^2 - 4) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l348_34879


namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l348_34838

/-- The function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem: If f(x) = x^2 - 2x + 3 has a maximum of 3 and a minimum of 2 on [0, a], then a ∈ [1, 2] -/
theorem f_max_min_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 := by
  sorry

#check f_max_min_implies_a_range

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l348_34838


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l348_34886

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4*a*x + 2 else Real.log x / Real.log a

-- Define the property of f being decreasing on the entire real line
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- State the theorem
theorem decreasing_function_a_range (a : ℝ) :
  (is_decreasing (f a)) → (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_a_range_l348_34886


namespace NUMINAMATH_CALUDE_min_height_is_six_l348_34882

/-- Represents the dimensions of a rectangular box with square bases -/
structure BoxDimensions where
  base_side : ℝ
  height : ℝ

/-- The surface area of a rectangular box with square bases -/
def surface_area (d : BoxDimensions) : ℝ :=
  2 * d.base_side^2 + 4 * d.base_side * d.height

/-- The constraint that the height is 3 units greater than the base side -/
def height_constraint (d : BoxDimensions) : Prop :=
  d.height = d.base_side + 3

/-- The constraint that the surface area is at least 90 square units -/
def area_constraint (d : BoxDimensions) : Prop :=
  surface_area d ≥ 90

theorem min_height_is_six :
  ∃ (d : BoxDimensions),
    height_constraint d ∧
    area_constraint d ∧
    d.height = 6 ∧
    ∀ (d' : BoxDimensions),
      height_constraint d' → area_constraint d' → d'.height ≥ d.height :=
by sorry

end NUMINAMATH_CALUDE_min_height_is_six_l348_34882


namespace NUMINAMATH_CALUDE_power_sum_value_l348_34832

theorem power_sum_value (a : ℝ) (x y : ℝ) (h1 : a^x = 4) (h2 : a^y = 9) : a^(x+y) = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l348_34832


namespace NUMINAMATH_CALUDE_percentage_problem_l348_34848

theorem percentage_problem (number : ℝ) (excess : ℝ) (base_percentage : ℝ) (base_number : ℝ) (percentage : ℝ) : 
  number = 6400 →
  excess = 190 →
  base_percentage = 20 →
  base_number = 650 →
  percentage = 5 →
  percentage / 100 * number = base_percentage / 100 * base_number + excess :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l348_34848


namespace NUMINAMATH_CALUDE_union_equals_reals_l348_34847

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define set B
def B : Set ℝ := {x | x > -1}

-- Theorem statement
theorem union_equals_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_equals_reals_l348_34847


namespace NUMINAMATH_CALUDE_complement_of_union_l348_34894

def U : Set ℕ := {x | x ∈ Finset.range 6 \ {0}}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l348_34894


namespace NUMINAMATH_CALUDE_yellow_red_block_difference_l348_34816

/-- Given a toy bin with red, yellow, and blue blocks, prove the difference between yellow and red blocks -/
theorem yellow_red_block_difference 
  (red : ℕ) 
  (yellow : ℕ) 
  (blue : ℕ) 
  (h1 : red = 18) 
  (h2 : yellow > red) 
  (h3 : blue = red + 14) 
  (h4 : red + yellow + blue = 75) : 
  yellow - red = 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_block_difference_l348_34816


namespace NUMINAMATH_CALUDE_initial_charge_correct_l348_34878

/-- The initial charge for renting a bike at Oceanside Bike Rental Shop -/
def initial_charge : ℝ := 17

/-- The hourly rate for renting a bike -/
def hourly_rate : ℝ := 7

/-- The number of hours Tom rented the bike -/
def rental_hours : ℝ := 9

/-- The total cost Tom paid for renting the bike -/
def total_cost : ℝ := 80

/-- Theorem stating that the initial charge is correct given the conditions -/
theorem initial_charge_correct : 
  initial_charge + hourly_rate * rental_hours = total_cost :=
by sorry

end NUMINAMATH_CALUDE_initial_charge_correct_l348_34878


namespace NUMINAMATH_CALUDE_power_inequality_l348_34887

theorem power_inequality : 22^55 > 33^44 ∧ 33^44 > 55^33 ∧ 55^33 > 66^22 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l348_34887


namespace NUMINAMATH_CALUDE_cube_root_five_sixteenths_l348_34867

theorem cube_root_five_sixteenths :
  (5 / 16 : ℝ)^(1/3) = (5 : ℝ)^(1/3) / 2^(4/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_five_sixteenths_l348_34867


namespace NUMINAMATH_CALUDE_evenly_geometric_difference_l348_34874

/-- A 3-digit number is evenly geometric if it comprises 3 distinct even digits
    which form a geometric sequence when read from left to right. -/
def EvenlyGeometric (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧
                 a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                 Even a ∧ Even b ∧ Even c ∧
                 ∃ (r : ℚ), b = a * r ∧ c = a * r^2

theorem evenly_geometric_difference :
  ∃ (max min : ℕ),
    (∀ n, EvenlyGeometric n → n ≤ max) ∧
    (∀ n, EvenlyGeometric n → min ≤ n) ∧
    (EvenlyGeometric max) ∧
    (EvenlyGeometric min) ∧
    max - min = 0 :=
sorry

end NUMINAMATH_CALUDE_evenly_geometric_difference_l348_34874


namespace NUMINAMATH_CALUDE_car_mileage_l348_34833

theorem car_mileage (highway_miles_per_tank : ℕ) (city_mpg : ℕ) (mpg_difference : ℕ) :
  highway_miles_per_tank = 462 →
  city_mpg = 24 →
  mpg_difference = 9 →
  (highway_miles_per_tank / (city_mpg + mpg_difference)) * city_mpg = 336 :=
by sorry

end NUMINAMATH_CALUDE_car_mileage_l348_34833


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l348_34889

def triangle_area (a b : ℝ) (cos_C : ℝ) : ℝ :=
  6

theorem triangle_area_theorem (a b cos_C : ℝ) :
  a = 3 →
  b = 5 →
  5 * cos_C^2 - 7 * cos_C - 6 = 0 →
  triangle_area a b cos_C = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l348_34889


namespace NUMINAMATH_CALUDE_min_layoff_rounds_is_four_l348_34877

def initial_employees : ℕ := 1000
def layoff_rate : ℝ := 0.1
def total_layoffs : ℕ := 271

def remaining_employees (n : ℕ) : ℝ :=
  initial_employees * (1 - layoff_rate) ^ n

def layoffs_after_rounds (n : ℕ) : ℝ :=
  initial_employees - remaining_employees n

theorem min_layoff_rounds_is_four :
  (∀ k < 4, layoffs_after_rounds k < total_layoffs) ∧
  layoffs_after_rounds 4 ≥ total_layoffs := by sorry

end NUMINAMATH_CALUDE_min_layoff_rounds_is_four_l348_34877


namespace NUMINAMATH_CALUDE_ian_painted_48_faces_l348_34842

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Ian painted -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted by Ian -/
def total_faces_painted : ℕ := faces_per_cuboid * num_cuboids

/-- Theorem stating that the total number of faces painted by Ian is 48 -/
theorem ian_painted_48_faces : total_faces_painted = 48 := by
  sorry

end NUMINAMATH_CALUDE_ian_painted_48_faces_l348_34842


namespace NUMINAMATH_CALUDE_limit_of_r_as_m_approaches_zero_l348_34884

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 2*x - 6

-- Define L(m) as the smaller root of x^2 + 2x - (m + 6) = 0
noncomputable def L (m : ℝ) : ℝ := -1 - Real.sqrt (m + 7)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem limit_of_r_as_m_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ → |r m - 1 / Real.sqrt 7| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_r_as_m_approaches_zero_l348_34884


namespace NUMINAMATH_CALUDE_total_cost_is_30_l348_34881

-- Define the cost of silverware
def silverware_cost : ℝ := 20

-- Define the cost of dinner plates as 50% of silverware cost
def dinner_plates_cost : ℝ := silverware_cost * 0.5

-- Theorem: The total cost is $30
theorem total_cost_is_30 : silverware_cost + dinner_plates_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_30_l348_34881


namespace NUMINAMATH_CALUDE_seventeen_is_possible_result_l348_34858

def expression (op1 op2 op3 : ℕ → ℕ → ℕ) : ℕ :=
  op1 7 (op2 2 (op3 5 8))

def is_valid_operation (op : ℕ → ℕ → ℕ) : Prop :=
  (op = (·+·)) ∨ (op = (·-·)) ∨ (op = (·*·))

theorem seventeen_is_possible_result :
  ∃ (op1 op2 op3 : ℕ → ℕ → ℕ),
    is_valid_operation op1 ∧
    is_valid_operation op2 ∧
    is_valid_operation op3 ∧
    op1 ≠ op2 ∧ op2 ≠ op3 ∧ op1 ≠ op3 ∧
    expression op1 op2 op3 = 17 :=
by
  sorry

#check seventeen_is_possible_result

end NUMINAMATH_CALUDE_seventeen_is_possible_result_l348_34858


namespace NUMINAMATH_CALUDE_minimum_cost_for_boxes_l348_34863

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the problem parameters -/
structure ProblemParams where
  boxDims : BoxDimensions
  costPerBox : ℝ
  totalVolumeNeeded : ℝ

theorem minimum_cost_for_boxes (p : ProblemParams)
  (h1 : p.boxDims.length = 20)
  (h2 : p.boxDims.width = 20)
  (h3 : p.boxDims.height = 12)
  (h4 : p.costPerBox = 0.4)
  (h5 : p.totalVolumeNeeded = 2160000) :
  ⌈p.totalVolumeNeeded / boxVolume p.boxDims⌉ * p.costPerBox = 180 := by
  sorry

#check minimum_cost_for_boxes

end NUMINAMATH_CALUDE_minimum_cost_for_boxes_l348_34863


namespace NUMINAMATH_CALUDE_raspberry_carton_is_eight_ounces_l348_34823

/-- Represents the cost and size of fruit cartons, and the amount needed for muffins --/
structure FruitData where
  blueberry_cost : ℚ
  blueberry_size : ℚ
  raspberry_cost : ℚ
  batches : ℕ
  fruit_per_batch : ℚ
  savings : ℚ

/-- Calculates the size of a raspberry carton based on the given data --/
def raspberry_carton_size (data : FruitData) : ℚ :=
  sorry

/-- Theorem stating that the raspberry carton size is 8 ounces --/
theorem raspberry_carton_is_eight_ounces (data : FruitData)
    (h1 : data.blueberry_cost = 5)
    (h2 : data.blueberry_size = 6)
    (h3 : data.raspberry_cost = 3)
    (h4 : data.batches = 4)
    (h5 : data.fruit_per_batch = 12)
    (h6 : data.savings = 22) :
    raspberry_carton_size data = 8 := by
  sorry

end NUMINAMATH_CALUDE_raspberry_carton_is_eight_ounces_l348_34823


namespace NUMINAMATH_CALUDE_multiple_subtraction_problem_l348_34853

theorem multiple_subtraction_problem (n : ℝ) (m : ℝ) : 
  n = 6 → m * n - 6 = 2 * n → m * n = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiple_subtraction_problem_l348_34853


namespace NUMINAMATH_CALUDE_zoe_app_cost_l348_34856

/-- Calculates the total cost of an app and its associated expenses -/
def total_app_cost (initial_cost monthly_cost in_game_cost upgrade_cost months : ℕ) : ℕ :=
  initial_cost + (monthly_cost * months) + in_game_cost + upgrade_cost

/-- Theorem stating the total cost for Zoe's app usage -/
theorem zoe_app_cost : total_app_cost 5 8 10 12 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_zoe_app_cost_l348_34856


namespace NUMINAMATH_CALUDE_sixth_term_is_three_l348_34864

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_first_three : a 0 + a 1 + a 2 = 168
  diff_2_5 : a 1 - a 4 = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l348_34864


namespace NUMINAMATH_CALUDE_min_value_sum_l348_34809

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → 
    (a + 1/a) + (b + 1/b) ≤ (x + 1/x) + (y + 1/y) ∧
    (a + 1/a) + (b + 1/b) = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l348_34809


namespace NUMINAMATH_CALUDE_carla_marbles_l348_34839

/-- The number of marbles Carla has now, given her initial marbles and the number she bought -/
def total_marbles (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem stating that Carla now has 187 marbles -/
theorem carla_marbles : total_marbles 53 134 = 187 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l348_34839


namespace NUMINAMATH_CALUDE_factor_expression_l348_34817

theorem factor_expression (x : ℝ) : 84 * x^7 - 306 * x^13 = 6 * x^7 * (14 - 51 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l348_34817


namespace NUMINAMATH_CALUDE_line_transformation_l348_34885

-- Define the original line
def original_line (x : ℝ) : ℝ := x

-- Define rotation by 90 degrees counterclockwise
def rotate_90 (x y : ℝ) : ℝ × ℝ := (-y, x)

-- Define vertical shift by 1 unit
def shift_up (y : ℝ) : ℝ := y + 1

-- Theorem statement
theorem line_transformation :
  ∀ x : ℝ, 
  let (x', y') := rotate_90 x (original_line x)
  shift_up y' = -x' + 1 := by
  sorry

end NUMINAMATH_CALUDE_line_transformation_l348_34885


namespace NUMINAMATH_CALUDE_solve_for_x_l348_34836

theorem solve_for_x (x y : ℝ) (h1 : x - y = 15) (h2 : x + y = 9) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l348_34836


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l348_34869

/-- A function f(x) = x³ + ax - 2 that is increasing on (1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x - 2

/-- The derivative of f(x) -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x > 1, ∀ y > x, f a y > f a x) ↔ a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l348_34869


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l348_34831

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (3 * x + 15) = 12) ∧ (x = 43) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l348_34831


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l348_34870

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  (3/2 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l348_34870


namespace NUMINAMATH_CALUDE_equation_solution_l348_34827

theorem equation_solution :
  let a : ℝ := 9
  let b : ℝ := 4
  let c : ℝ := 3
  ∃ x : ℝ, (x^2 + c + b^2 = (a - x)^2 + c) ∧ (x = 65 / 18) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l348_34827


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l348_34890

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotation_90_degrees :
  rotate90 (8 - 5 * Complex.I) = 5 + 8 * Complex.I := by sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l348_34890


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l348_34892

theorem coefficient_x_squared_in_expansion : ∃ (a b c d e : ℤ), 
  (2 * X + 1)^2 * (X - 2)^3 = a * X^5 + b * X^4 + c * X^3 + 10 * X^2 + d * X + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l348_34892


namespace NUMINAMATH_CALUDE_total_red_balloons_l348_34857

theorem total_red_balloons (sam_initial : ℝ) (fred_received : ℝ) (dan_balloons : ℝ)
  (h1 : sam_initial = 46.0)
  (h2 : fred_received = 10.0)
  (h3 : dan_balloons = 16.0) :
  sam_initial - fred_received + dan_balloons = 52.0 := by
sorry

end NUMINAMATH_CALUDE_total_red_balloons_l348_34857


namespace NUMINAMATH_CALUDE_equality_of_fractions_l348_34862

theorem equality_of_fractions (x y z l : ℝ) :
  (9 / (x + y + 1) = l / (x + z - 1)) ∧
  (l / (x + z - 1) = 13 / (z - y + 2)) →
  l = 22 := by
sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l348_34862


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l348_34865

/-- Given a circle with equation (x-3)^2+(y+4)^2=2, 
    prove that its symmetric circle with respect to y=0 
    has the equation (x-3)^2+(y-4)^2=2 -/
theorem symmetric_circle_equation : 
  ∀ (x y : ℝ), 
  (∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = 2 ∧ x₀ = 3 ∧ y₀ = -4) →
  (∃ (x₁ y₁ : ℝ), (x - x₁)^2 + (y - y₁)^2 = 2 ∧ x₁ = 3 ∧ y₁ = 4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l348_34865


namespace NUMINAMATH_CALUDE_triangle_side_squares_sum_l348_34802

theorem triangle_side_squares_sum (a b c : ℝ) (h : a + b + c = 4) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^2 + b^2 + c^2 > 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_squares_sum_l348_34802


namespace NUMINAMATH_CALUDE_x_percent_of_x_squared_l348_34837

theorem x_percent_of_x_squared (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x^2 = 16) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_x_squared_l348_34837


namespace NUMINAMATH_CALUDE_cyclic_inequality_l348_34852

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 / (a^2 + a*b + b^2) + b^3 / (b^2 + b*c + c^2) + c^3 / (c^2 + c*a + a^2) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l348_34852


namespace NUMINAMATH_CALUDE_equal_sprocket_production_l348_34898

/-- Represents the production rates and times of two machines manufacturing sprockets -/
structure SprocketProduction where
  machine_a_rate : ℝ  -- Sprockets per hour for Machine A
  machine_b_rate : ℝ  -- Sprockets per hour for Machine B
  machine_b_time : ℝ  -- Time taken by Machine B in hours

/-- Theorem stating that both machines produce the same number of sprockets -/
theorem equal_sprocket_production (sp : SprocketProduction) 
  (h1 : sp.machine_a_rate = 4)  -- Machine A produces 4 sprockets per hour
  (h2 : sp.machine_b_rate = sp.machine_a_rate * 1.1)  -- Machine B is 10% faster
  (h3 : sp.machine_b_time * sp.machine_b_rate = (sp.machine_b_time + 10) * sp.machine_a_rate)  -- Total production is equal
  : sp.machine_a_rate * (sp.machine_b_time + 10) = 440 ∧ sp.machine_b_rate * sp.machine_b_time = 440 :=
by sorry

end NUMINAMATH_CALUDE_equal_sprocket_production_l348_34898


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l348_34851

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 24 years older than his son and the son's present age is 22 years. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 22 →
    man_age = son_age + 24 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l348_34851


namespace NUMINAMATH_CALUDE_blood_drops_per_liter_l348_34829

/-- The number of drops of blood sucked by one mosquito in a single feeding. -/
def drops_per_mosquito : ℕ := 20

/-- The number of liters of blood loss that is fatal. -/
def fatal_blood_loss : ℕ := 3

/-- The number of mosquitoes that would cause a fatal blood loss if they all fed. -/
def fatal_mosquito_count : ℕ := 750

/-- The number of drops of blood in one liter. -/
def drops_per_liter : ℕ := 5000

theorem blood_drops_per_liter :
  drops_per_liter = (drops_per_mosquito * fatal_mosquito_count) / fatal_blood_loss := by
  sorry

end NUMINAMATH_CALUDE_blood_drops_per_liter_l348_34829


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_1_l348_34828

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 9x + 1 = 0 -/
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

theorem discriminant_of_5x2_minus_9x_plus_1 :
  discriminant a b c = 61 := by sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_1_l348_34828


namespace NUMINAMATH_CALUDE_goldbach_2024_l348_34875

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem goldbach_2024 : ∃ p q : ℕ, 
  is_prime p ∧ 
  is_prime q ∧ 
  p ≠ q ∧ 
  p + q = 2024 :=
sorry

end NUMINAMATH_CALUDE_goldbach_2024_l348_34875


namespace NUMINAMATH_CALUDE_wendy_running_distance_l348_34845

theorem wendy_running_distance (ran walked : ℝ) (h1 : ran = 19.833333333333332) 
  (h2 : walked = 9.166666666666666) : 
  ran - walked = 10.666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_wendy_running_distance_l348_34845


namespace NUMINAMATH_CALUDE_special_choose_result_l348_34846

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 6 players from 16, with at most one from a special group of 4 -/
def specialChoose : ℕ :=
  choose 16 6 - (choose 4 2 * choose 12 4 + choose 4 3 * choose 12 3 + choose 4 4 * choose 12 2)

theorem special_choose_result : specialChoose = 4092 := by sorry

end NUMINAMATH_CALUDE_special_choose_result_l348_34846


namespace NUMINAMATH_CALUDE_camping_hike_distance_l348_34830

/-- The total distance hiked by Irwin's family during their camping trip -/
theorem camping_hike_distance 
  (car_to_stream : ℝ) 
  (stream_to_meadow : ℝ) 
  (meadow_to_campsite : ℝ)
  (h1 : car_to_stream = 0.2)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  car_to_stream + stream_to_meadow + meadow_to_campsite = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l348_34830


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l348_34818

/-- An arithmetic progression with a non-zero difference -/
def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Consecutive terms of a geometric progression -/
def geometric_progression (x y z : ℝ) : Prop :=
  y * y = x * z

/-- The main theorem -/
theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_progression a d)
  (h_geom : geometric_progression (a 10) (a 13) (a 19)) :
  (a 12) / (a 18) = 5 / 11 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l348_34818


namespace NUMINAMATH_CALUDE_weight_of_Na2Ca_CO3_2_l348_34820

-- Define molar masses of elements
def Na_mass : ℝ := 22.99
def Ca_mass : ℝ := 40.08
def C_mass : ℝ := 12.01
def O_mass : ℝ := 16.00

-- Define the number of atoms in Na2Ca(CO3)2
def Na_count : ℕ := 2
def Ca_count : ℕ := 1
def C_count : ℕ := 2
def O_count : ℕ := 6

-- Define the number of moles of Na2Ca(CO3)2
def moles : ℝ := 3.75

-- Define the molar mass of Na2Ca(CO3)2
def Na2Ca_CO3_2_mass : ℝ :=
  Na_count * Na_mass + Ca_count * Ca_mass + C_count * C_mass + O_count * O_mass

-- Theorem: The weight of 3.75 moles of Na2Ca(CO3)2 is 772.8 grams
theorem weight_of_Na2Ca_CO3_2 : moles * Na2Ca_CO3_2_mass = 772.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_Na2Ca_CO3_2_l348_34820


namespace NUMINAMATH_CALUDE_set_inclusion_implies_upper_bound_l348_34804

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Define the complement of B in ℝ
def C_R_B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- Theorem statement
theorem set_inclusion_implies_upper_bound (a : ℝ) :
  A ⊆ C_R_B a → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_upper_bound_l348_34804


namespace NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l348_34860

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 = 4

-- Define what it means for a point to be inside the circle
def point_inside_circle (x y a : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 < 4

-- Theorem statement
theorem point_inside_circle_implies_a_range :
  ∀ a : ℝ, point_inside_circle (-1) (-1) a → -1 < a ∧ a < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l348_34860


namespace NUMINAMATH_CALUDE_at_least_four_same_prob_l348_34821

-- Define the number of dice and sides
def num_dice : ℕ := 5
def num_sides : ℕ := 8

-- Define the probability of a specific outcome for a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of all five dice showing the same number
def all_same_prob : ℚ := single_prob ^ (num_dice - 1)

-- Define the probability of exactly four dice showing the same number
def four_same_prob : ℚ := 
  (num_dice : ℚ) * single_prob ^ (num_dice - 2) * (1 - single_prob)

-- State the theorem
theorem at_least_four_same_prob : 
  all_same_prob + four_same_prob = 9 / 1024 := by sorry

end NUMINAMATH_CALUDE_at_least_four_same_prob_l348_34821


namespace NUMINAMATH_CALUDE_max_product_constraint_l348_34861

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → a + 2 * b = 10 → ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + 2 * y = 10 → x * y ≤ m ∧ a * b = m :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l348_34861


namespace NUMINAMATH_CALUDE_sum_of_new_observations_l348_34819

/-- Given 10 observations with an average of 21, prove that adding two new observations
    that increase the average by 2 results in the sum of the two new observations being 66. -/
theorem sum_of_new_observations (initial_count : Nat) (initial_avg : ℝ) (new_count : Nat) (avg_increase : ℝ) :
  initial_count = 10 →
  initial_avg = 21 →
  new_count = initial_count + 2 →
  avg_increase = 2 →
  (new_count : ℝ) * (initial_avg + avg_increase) - (initial_count : ℝ) * initial_avg = 66 := by
  sorry

#check sum_of_new_observations

end NUMINAMATH_CALUDE_sum_of_new_observations_l348_34819


namespace NUMINAMATH_CALUDE_factorization_problem_multiplication_problem_l348_34866

variable (x y : ℝ)

theorem factorization_problem : x^5 - x^3 * y^2 = x^3 * (x - y) * (x + y) := by sorry

theorem multiplication_problem : (-2 * x^3 * y^2) * (3 * x^2 * y) = -6 * x^5 * y^3 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_multiplication_problem_l348_34866


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l348_34810

theorem smallest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = 60 →           -- One angle is 60°
  β = 75 →           -- Another angle is 75° (complementary to 45° + 60°)
  γ ≤ α ∧ γ ≤ β →    -- γ is the smallest angle
  γ = 45 :=          -- The smallest angle is 45°
by
  sorry

#check smallest_angle_in_triangle

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l348_34810


namespace NUMINAMATH_CALUDE_condition_equivalence_l348_34805

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Add necessary conditions for a valid triangle
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

-- Define the theorem
theorem condition_equivalence (t : Triangle) : 
  (t.a / Real.cos t.A = t.b / Real.cos t.B) ↔ (t.a = t.b) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l348_34805


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l348_34801

theorem opposite_reciprocal_sum (a b m n : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : m * n = 1)  -- m and n are reciprocals
  : 5*a + 5*b - m*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l348_34801


namespace NUMINAMATH_CALUDE_rotation_equivalence_l348_34891

theorem rotation_equivalence (y : ℝ) : 
  (330 : ℝ) = (360 - y) → y < 360 → y = 30 := by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l348_34891


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l348_34840

/-- Given a circle with area 16π, prove its diameter is 8 -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), π * r^2 = 16 * π → 2 * r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l348_34840


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l348_34855

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 13) + Real.sqrt (x^2 - 10*x + 26) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l348_34855


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_stock_price_calculation_l348_34859

theorem stock_price_after_two_years 
  (initial_price : ℝ) 
  (first_year_increase : ℝ) 
  (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price

theorem stock_price_calculation : 
  stock_price_after_two_years 120 1 0.3 = 168 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_stock_price_calculation_l348_34859


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l348_34812

theorem cube_root_unity_sum (w : ℂ) 
  (h1 : w^3 - 1 = 0) 
  (h2 : w^2 + w + 1 ≠ 0) : 
  w^105 + w^106 + w^107 + w^108 + w^109 + w^110 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l348_34812


namespace NUMINAMATH_CALUDE_perpendicular_line_modulus_l348_34835

/-- Given a line ax + y + 5 = 0 and points P and Q, prove the modulus of z = a + 4i -/
theorem perpendicular_line_modulus (a : ℝ) : 
  let P : ℝ × ℝ := (2, 4)
  let Q : ℝ × ℝ := (4, 3)
  let line (x y : ℝ) := a * x + y + 5 = 0
  let perpendicular (P Q : ℝ × ℝ) (line : ℝ → ℝ → Prop) := 
    (Q.2 - P.2) * a = -(Q.1 - P.1)  -- Perpendicular condition
  let z : ℂ := a + 4 * Complex.I
  perpendicular P Q line → Complex.abs z = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_modulus_l348_34835


namespace NUMINAMATH_CALUDE_waiting_time_theorem_l348_34880

/-- Represents a queue with Slowpokes and Quickies -/
structure Queue where
  m : ℕ  -- number of Slowpokes
  n : ℕ  -- number of Quickies
  a : ℕ  -- time taken by Quickies
  b : ℕ  -- time taken by Slowpokes

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Minimum total waiting time -/
def T_min (q : Queue) : ℕ :=
  q.a * choose q.n 2 + q.a * q.m * q.n + q.b * choose q.m 2

/-- Maximum total waiting time -/
def T_max (q : Queue) : ℕ :=
  q.a * choose q.n 2 + q.b * q.m * q.n + q.b * choose q.m 2

/-- Expected total waiting time -/
def E_T (q : Queue) : ℚ :=
  (choose (q.n + q.m) 2 : ℚ) * (q.b * q.m + q.a * q.n) / (q.m + q.n)

theorem waiting_time_theorem (q : Queue) :
  T_min q ≤ T_max q ∧ 
  (T_min q : ℚ) ≤ E_T q ∧ 
  E_T q ≤ (T_max q : ℚ) := by sorry

end NUMINAMATH_CALUDE_waiting_time_theorem_l348_34880


namespace NUMINAMATH_CALUDE_erased_digit_greater_than_original_l348_34872

-- Define the fraction
def fraction : Rat := 3 / 7

-- Define the number of digits after the decimal point
def num_digits : Nat := 1000

-- Define the position of the digit to be erased
def erased_position : Nat := 500

-- Function to get the nth digit after the decimal point
def nth_digit (n : Nat) : Nat :=
  sorry

-- Function to construct the number after erasing the 500th digit
def number_after_erasing : Rat :=
  sorry

-- Theorem statement
theorem erased_digit_greater_than_original :
  number_after_erasing > fraction :=
sorry

end NUMINAMATH_CALUDE_erased_digit_greater_than_original_l348_34872


namespace NUMINAMATH_CALUDE_tom_walking_speed_l348_34873

/-- Given the walking speeds of Max, Lila, and Tom, prove Tom's speed -/
theorem tom_walking_speed (max_speed : ℚ) (lila_ratio : ℚ) (tom_ratio : ℚ)
  (h1 : max_speed = 5)
  (h2 : lila_ratio = 4/5)
  (h3 : tom_ratio = 6/7) :
  tom_ratio * (lila_ratio * max_speed) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_tom_walking_speed_l348_34873


namespace NUMINAMATH_CALUDE_area_fold_points_specific_triangle_l348_34825

/-- Represents a right triangle ABC -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  angleB : ℝ

/-- Represents the area of fold points -/
def area_fold_points (t : RightTriangle) : ℝ := sorry

/-- Main theorem: Area of fold points for the given right triangle -/
theorem area_fold_points_specific_triangle :
  let t : RightTriangle := { AB := 45, AC := 90, angleB := 90 }
  area_fold_points t = 379 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_area_fold_points_specific_triangle_l348_34825


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l348_34850

/-- The area of a square with perimeter equal to that of a triangle with sides 7.3 cm, 8.6 cm, and 10.1 cm is 42.25 square centimeters. -/
theorem square_area_equal_perimeter_triangle (a b c : ℝ) (s : ℝ) :
  a = 7.3 ∧ b = 8.6 ∧ c = 10.1 →
  4 * s = a + b + c →
  s^2 = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l348_34850


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l348_34806

/-- Proves the distance traveled downstream by a boat -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : travel_time = 5) : 
  boat_speed + stream_speed * travel_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l348_34806


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l348_34897

theorem absolute_value_equation_solution_product : 
  (∀ x : ℝ, |x - 5| + 4 = 7 → x = 8 ∨ x = 2) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |x₁ - 5| + 4 = 7 ∧ |x₂ - 5| + 4 = 7) ∧
  (∀ x₁ x₂ : ℝ, |x₁ - 5| + 4 = 7 → |x₂ - 5| + 4 = 7 → x₁ * x₂ = 16) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l348_34897


namespace NUMINAMATH_CALUDE_chalk_boxes_l348_34808

theorem chalk_boxes (total_chalk : ℕ) (chalk_per_box : ℕ) (h1 : total_chalk = 3484) (h2 : chalk_per_box = 18) :
  (total_chalk + chalk_per_box - 1) / chalk_per_box = 194 := by
  sorry

end NUMINAMATH_CALUDE_chalk_boxes_l348_34808


namespace NUMINAMATH_CALUDE_spider_journey_l348_34871

theorem spider_journey (r : ℝ) (third_leg : ℝ) (h1 : r = 50) (h2 : third_leg = 70) :
  let diameter := 2 * r
  let second_leg := Real.sqrt (diameter^2 - third_leg^2)
  diameter + third_leg + second_leg = 170 + Real.sqrt 5100 := by
sorry

end NUMINAMATH_CALUDE_spider_journey_l348_34871


namespace NUMINAMATH_CALUDE_number_in_interval_l348_34803

theorem number_in_interval (x : ℝ) (h : x = (1/x) * (-x) + 2) :
  x = 1 ∧ 0 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_number_in_interval_l348_34803


namespace NUMINAMATH_CALUDE_backpack_cost_relationship_l348_34824

theorem backpack_cost_relationship (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for division
  (h2 : 810 > 0) -- Cost of type A backpacks is positive
  (h3 : 600 > 0) -- Cost of type B backpacks is positive
  (h4 : x + 20 > 0) -- Ensure denominator is positive
  : 
  810 / (x + 20) = (600 / x) * (1 - 0.1) :=
sorry

end NUMINAMATH_CALUDE_backpack_cost_relationship_l348_34824


namespace NUMINAMATH_CALUDE_vector_sum_equality_l348_34876

theorem vector_sum_equality (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-3, 4) →
  (3 : ℝ) • a + (4 : ℝ) • b = (-6, 19) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l348_34876


namespace NUMINAMATH_CALUDE_eight_fifteen_div_sixtyfour_six_l348_34899

theorem eight_fifteen_div_sixtyfour_six : 8^15 / 64^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_eight_fifteen_div_sixtyfour_six_l348_34899


namespace NUMINAMATH_CALUDE_no_three_digit_number_eight_times_smaller_l348_34822

theorem no_three_digit_number_eight_times_smaller : ¬ ∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (b ≤ 9) ∧ 
  (c ≤ 9) ∧ 
  (100 * a + 10 * b + c = 8 * (10 * b + c)) := by
sorry

end NUMINAMATH_CALUDE_no_three_digit_number_eight_times_smaller_l348_34822


namespace NUMINAMATH_CALUDE_combination_arrangement_equality_l348_34807

theorem combination_arrangement_equality (m : ℕ) : (Nat.choose m 3) = (m * (m - 1)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_arrangement_equality_l348_34807


namespace NUMINAMATH_CALUDE_correct_vote_distribution_l348_34813

/-- Represents the number of votes for each candidate -/
structure Votes where
  eliot : ℕ
  shaun : ℕ
  randy : ℕ
  lisa : ℕ

/-- Checks if the vote distribution satisfies the given conditions -/
def is_valid_vote_distribution (v : Votes) : Prop :=
  v.eliot = 2 * v.shaun ∧
  v.eliot = 4 * v.randy ∧
  v.shaun = 5 * v.randy ∧
  v.shaun = 3 * v.lisa ∧
  v.randy = 16

/-- The theorem stating that the given vote distribution is correct -/
theorem correct_vote_distribution :
  ∃ (v : Votes), is_valid_vote_distribution v ∧
    v.eliot = 64 ∧ v.shaun = 80 ∧ v.randy = 16 ∧ v.lisa = 27 :=
by
  sorry


end NUMINAMATH_CALUDE_correct_vote_distribution_l348_34813


namespace NUMINAMATH_CALUDE_solution_characterization_l348_34811

def is_solution (x y z w : ℝ) : Prop :=
  x + y = z^2 + w^2 + 6*z*w ∧
  x + z = y^2 + w^2 + 6*y*w ∧
  x + w = y^2 + z^2 + 6*y*z ∧
  y + z = x^2 + w^2 + 6*x*w ∧
  y + w = x^2 + z^2 + 6*x*z ∧
  z + w = x^2 + y^2 + 6*x*y

def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(0, 0, 0, 0), (1/4, 1/4, 1/4, 1/4), (-1/4, -1/4, 3/4, -1/4), (-1/2, -1/2, 5/2, -1/2)} ∪
  {(0, 0, 0, 0), (1/4, 1/4, 1/4, 1/4), (3/4, -1/4, -1/4, -1/4), (5/2, -1/2, -1/2, -1/2)} ∪
  {(0, 0, 0, 0), (1/4, 1/4, 1/4, 1/4), (-1/4, 3/4, -1/4, -1/4), (-1/2, 5/2, -1/2, -1/2)} ∪
  {(0, 0, 0, 0), (1/4, 1/4, 1/4, 1/4), (-1/4, -1/4, -1/4, 3/4), (-1/2, -1/2, -1/2, 5/2)}

theorem solution_characterization :
  ∀ x y z w : ℝ, is_solution x y z w ↔ (x, y, z, w) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l348_34811


namespace NUMINAMATH_CALUDE_solve_for_y_l348_34896

theorem solve_for_y (x y : ℝ) (hx : x = 51) (heq : x^3 * y^2 - 4 * x^2 * y^2 + 4 * x * y^2 = 100800) :
  y = 1/34 ∨ y = -1/34 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l348_34896


namespace NUMINAMATH_CALUDE_crayon_selection_proof_l348_34826

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem crayon_selection_proof :
  choose 12 4 = 495 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_proof_l348_34826


namespace NUMINAMATH_CALUDE_range_of_a_range_is_nonnegative_reals_l348_34814

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 = a}

-- State the theorem
theorem range_of_a (h : ∃ x, x ∈ A a) : a ≥ 0 := by
  sorry

-- Prove that this covers the entire range [0, +∞)
theorem range_is_nonnegative_reals : 
  ∀ a ≥ 0, ∃ x, x ∈ A a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_range_is_nonnegative_reals_l348_34814


namespace NUMINAMATH_CALUDE_min_value_product_l348_34868

theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (x / y + y) * (y / x + x) ≥ 4 ∧
  ((x / y + y) * (y / x + x) = 4 ↔ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l348_34868


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l348_34834

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 1) (hy : y > 2) (h : (x - 1) * (y - 2) = 4) :
  ∀ a b : ℝ, a > 1 → b > 2 → (a - 1) * (b - 2) = 4 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 1 ∧ y > 2 ∧ (x - 1) * (y - 2) = 4 ∧ x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l348_34834
