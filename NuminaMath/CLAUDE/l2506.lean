import Mathlib

namespace spiral_config_399_400_401_l2506_250619

/-- A function representing the spiral number sequence -/
def spiral_sequence : ℕ → ℕ := sorry

/-- Perfect squares are positioned at the center-bottom of their spiral layers -/
axiom perfect_square_position (n : ℕ) :
  ∃ (k : ℕ), k^2 = n → spiral_sequence n = spiral_sequence (n-1) + 1

/-- The vertical configuration of three consecutive numbers -/
def vertical_config (a b c : ℕ) : Prop :=
  spiral_sequence b = spiral_sequence a + 1 ∧
  spiral_sequence c = spiral_sequence b + 1

/-- Theorem stating the configuration of 399, 400, and 401 in the spiral -/
theorem spiral_config_399_400_401 :
  vertical_config 399 400 401 := by sorry

end spiral_config_399_400_401_l2506_250619


namespace sandy_watermelons_count_l2506_250663

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := 48

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := total_watermelons - jason_watermelons

theorem sandy_watermelons_count : sandy_watermelons = 11 := by
  sorry

end sandy_watermelons_count_l2506_250663


namespace system_solution_l2506_250666

/-- Given a system of equations with parameters a, b, and c, prove that the solutions for x, y, and z are as stated. -/
theorem system_solution (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  ∃ (x y z : ℝ),
    x ≠ y ∧
    (x - y) / (x + z) = a ∧
    (x^2 - y^2) / (x + z) = b ∧
    (x^3 + x^2*y - x*y^2 - y^3) / (x + z)^2 = b^2 / (a^2 * c) ∧
    x = (a^3 * c + b) / (2 * a) ∧
    y = (b - a^3 * c) / (2 * a) ∧
    z = (2 * a^2 * c - a^3 * c - b) / (2 * a) := by
  sorry

end system_solution_l2506_250666


namespace square_side_from_diagonal_difference_l2506_250696

/-- Given the difference between the diagonal and side of a square, 
    the side of the square can be uniquely determined. -/
theorem square_side_from_diagonal_difference (d_minus_a : ℝ) (d_minus_a_pos : 0 < d_minus_a) :
  ∃! a : ℝ, ∃ d : ℝ, 
    0 < a ∧ 
    d = Real.sqrt (2 * a ^ 2) ∧ 
    d - a = d_minus_a :=
by sorry

end square_side_from_diagonal_difference_l2506_250696


namespace sum_of_digits_power_6_13_l2506_250697

def power_6_13 : ℕ := 6^13

def ones_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_of_digits_power_6_13 :
  ones_digit power_6_13 + tens_digit power_6_13 = 13 := by
  sorry

end sum_of_digits_power_6_13_l2506_250697


namespace rectangle_area_change_l2506_250622

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 625) : 
  (1.2 * L) * (0.8 * W) = 600 := by sorry

end rectangle_area_change_l2506_250622


namespace diophantine_equation_solutions_l2506_250668

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m ≠ 0 ∧ n ≠ 0 →
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by sorry

end diophantine_equation_solutions_l2506_250668


namespace non_self_intersecting_chains_count_l2506_250669

/-- Represents a point on a circle -/
structure CirclePoint where
  label : ℕ

/-- Represents a polygonal chain on a circle -/
structure PolygonalChain where
  points : List CirclePoint
  is_non_self_intersecting : Bool

/-- The number of ways to form a non-self-intersecting polygonal chain -/
def count_non_self_intersecting_chains (n : ℕ) : ℕ :=
  n * 2^(n-2)

/-- Theorem stating the number of ways to form a non-self-intersecting polygonal chain -/
theorem non_self_intersecting_chains_count 
  (n : ℕ) 
  (h : n > 1) :
  (∀ (chain : PolygonalChain), 
    chain.points.length = n ∧ 
    chain.is_non_self_intersecting = true) →
  (∃! count : ℕ, count = count_non_self_intersecting_chains n) :=
sorry

end non_self_intersecting_chains_count_l2506_250669


namespace girls_count_l2506_250612

/-- The number of boys in the school -/
def num_boys : ℕ := 337

/-- The difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 402

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys + girl_boy_difference

theorem girls_count : num_girls = 739 := by
  sorry

end girls_count_l2506_250612


namespace sector_central_angle_l2506_250670

theorem sector_central_angle (area : Real) (radius : Real) (h1 : area = 3 / 8 * Real.pi) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 / 4 * Real.pi := by
  sorry

end sector_central_angle_l2506_250670


namespace nested_square_root_simplification_l2506_250689

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * (5 ^ (3/4)) := by
  sorry

end nested_square_root_simplification_l2506_250689


namespace grocery_payment_possible_l2506_250687

def soup_price : ℕ := 2
def bread_price : ℕ := 5
def cereal_price : ℕ := 3
def milk_price : ℕ := 4

def soup_quantity : ℕ := 6
def bread_quantity : ℕ := 2
def cereal_quantity : ℕ := 2
def milk_quantity : ℕ := 2

def total_cost : ℕ := 
  soup_price * soup_quantity + 
  bread_price * bread_quantity + 
  cereal_price * cereal_quantity + 
  milk_price * milk_quantity

def us_bill_denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

theorem grocery_payment_possible :
  ∃ (a b c d : ℕ), 
    a ∈ us_bill_denominations ∧ 
    b ∈ us_bill_denominations ∧ 
    c ∈ us_bill_denominations ∧ 
    d ∈ us_bill_denominations ∧ 
    a + b + c + d = total_cost :=
sorry

end grocery_payment_possible_l2506_250687


namespace fraction_square_equals_49_l2506_250695

theorem fraction_square_equals_49 : (3072 - 2993)^2 / 121 = 49 := by sorry

end fraction_square_equals_49_l2506_250695


namespace part_1_part_2_part_3_l2506_250649

-- Part 1
theorem part_1 (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 3 ∧ x₁ + p / x₁ = q ∧ x₂ + p / x₂ = q) →
  p = -6 ∧ q = 1 := by sorry

-- Part 2
theorem part_2 :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ + 7 / x₁ = 8 ∧ x₂ + 7 / x₂ = 8) →
  (∃ x : ℝ, x + 7 / x = 8 ∧ ∀ y : ℝ, y + 7 / y = 8 → y ≤ x) →
  (∃ x : ℝ, x + 7 / x = 8 ∧ x = 7) := by sorry

-- Part 3
theorem part_3 (n : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    2 * x₁ + (n^2 - n) / (2 * x₁ - 1) = 2 * n ∧
    2 * x₂ + (n^2 - n) / (2 * x₂ - 1) = 2 * n) →
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    2 * x₁ + (n^2 - n) / (2 * x₁ - 1) = 2 * n ∧
    2 * x₂ + (n^2 - n) / (2 * x₂ - 1) = 2 * n ∧
    (2 * x₁ - 1) / (2 * x₂) = (n - 1) / (n + 1)) := by sorry

end part_1_part_2_part_3_l2506_250649


namespace spatial_relations_l2506_250628

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular parallel subset : ∀ {T U : Type}, T → U → Prop)

-- Define the given conditions
variable (α β γ : Plane)
variable (m n : Line)
variable (h1 : m ≠ n)

-- Define the main theorem
theorem spatial_relations :
  (∀ α β γ : Plane, perpendicular α β → parallel α γ ∧ perpendicular α γ) →
  ((parallel m n ∧ subset n α) → parallel m α) ∧
  ((perpendicular m α ∧ parallel n α) → perpendicular α β) :=
sorry

end spatial_relations_l2506_250628


namespace symmetric_points_sum_power_l2506_250641

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that (m + n)^2023 = -1 --/
theorem symmetric_points_sum_power (m n : ℝ) : 
  (m = 3 ∧ n = -4) → (m + n)^2023 = -1 := by
  sorry

end symmetric_points_sum_power_l2506_250641


namespace smallest_perimeter_isosceles_triangle_l2506_250615

/-- Triangle with positive integer side lengths --/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Isosceles triangle where two sides are equal --/
def IsoscelesTriangle (t : IntegerTriangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Perimeter of a triangle --/
def Perimeter (t : IntegerTriangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

/-- Angle bisector theorem relation --/
def AngleBisectorRelation (t : IntegerTriangle) (bisectorLength : ℕ+) : Prop :=
  ∃ (x y : ℕ+), x + y = t.c ∧ bisectorLength * t.c = t.a * y

/-- Main theorem --/
theorem smallest_perimeter_isosceles_triangle :
  ∀ (t : IntegerTriangle),
    IsoscelesTriangle t →
    AngleBisectorRelation t 8 →
    (∀ (t' : IntegerTriangle),
      IsoscelesTriangle t' →
      AngleBisectorRelation t' 8 →
      Perimeter t ≤ Perimeter t') →
    Perimeter t = 108 := by
  sorry

end smallest_perimeter_isosceles_triangle_l2506_250615


namespace lunch_break_duration_l2506_250646

/-- Given the recess breaks and total time outside of class, prove the lunch break duration. -/
theorem lunch_break_duration 
  (recess1 recess2 recess3 total_outside : ℕ)
  (h1 : recess1 = 15)
  (h2 : recess2 = 15)
  (h3 : recess3 = 20)
  (h4 : total_outside = 80) :
  total_outside - (recess1 + recess2 + recess3) = 30 := by
  sorry

end lunch_break_duration_l2506_250646


namespace cost_of_dozen_pens_l2506_250667

/-- The cost of one pen in rupees -/
def pen_cost : ℕ := 65

/-- The ratio of the cost of one pen to one pencil -/
def pen_pencil_ratio : ℚ := 5/1

/-- The cost of 3 pens and some pencils in rupees -/
def total_cost : ℕ := 260

/-- The number of pens in a dozen -/
def dozen : ℕ := 12

/-- Theorem stating that the cost of one dozen pens is 780 rupees -/
theorem cost_of_dozen_pens : pen_cost * dozen = 780 := by
  sorry

end cost_of_dozen_pens_l2506_250667


namespace imo_2007_hktst_1_problem_6_l2506_250651

theorem imo_2007_hktst_1_problem_6 :
  ∀ x y : ℕ+, 
    (∃ k : ℕ+, x = 11 * k^2 ∧ y = 11 * k) ↔ 
    ∃ n : ℤ, (x.val^2 * y.val + x.val + y.val : ℤ) = n * (x.val * y.val^2 + y.val + 11) := by
  sorry

end imo_2007_hktst_1_problem_6_l2506_250651


namespace age_ratio_after_time_l2506_250634

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- Represents the ratio between two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

def Age.addYears (a : Age) (years : ℕ) : Age :=
  ⟨a.value + years⟩

def Age.subtractYears (a : Age) (years : ℕ) : Age :=
  ⟨a.value - years⟩

def AgeRatio.fromAges (a b : Age) : AgeRatio :=
  ⟨a.value, b.value⟩

theorem age_ratio_after_time (sandy_age molly_age : Age) 
    (h1 : AgeRatio.fromAges sandy_age molly_age = ⟨7, 2⟩)
    (h2 : (sandy_age.subtractYears 6).value = 78) :
    AgeRatio.fromAges (sandy_age.addYears 16) (molly_age.addYears 16) = ⟨5, 2⟩ := by
  sorry

end age_ratio_after_time_l2506_250634


namespace cost_of_450_candies_l2506_250682

/-- Represents the cost calculation for chocolate candies --/
def chocolate_cost (candies_per_box : ℕ) (discount_threshold : ℕ) (regular_price : ℚ) (discount_price : ℚ) (total_candies : ℕ) : ℚ :=
  let boxes := total_candies / candies_per_box
  if boxes ≥ discount_threshold then
    (boxes : ℚ) * discount_price
  else
    (boxes : ℚ) * regular_price

/-- Theorem stating the cost of 450 chocolate candies --/
theorem cost_of_450_candies :
  chocolate_cost 15 10 5 4 450 = 120 := by
  sorry

end cost_of_450_candies_l2506_250682


namespace wednesday_distance_l2506_250636

/-- Represents the distance Mona biked on each day of the week -/
structure BikeDistance where
  monday : ℕ
  wednesday : ℕ
  saturday : ℕ

/-- Defines the conditions of Mona's biking schedule -/
def validBikeSchedule (d : BikeDistance) : Prop :=
  d.monday + d.wednesday + d.saturday = 30 ∧
  d.monday = 6 ∧
  d.saturday = 2 * d.monday

theorem wednesday_distance (d : BikeDistance) (h : validBikeSchedule d) : d.wednesday = 12 := by
  sorry

end wednesday_distance_l2506_250636


namespace complex_number_location_l2506_250653

theorem complex_number_location (z : ℂ) : 
  z * Complex.I = 2015 - Complex.I → 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
by sorry

end complex_number_location_l2506_250653


namespace cookies_sold_l2506_250671

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def lottery_win : ℕ := 500
def sister_gift : ℕ := 500
def remaining_needed : ℕ := 3214

theorem cookies_sold :
  ∃ (n : ℕ), n * cookie_price = 
    trip_cost - 
    (hourly_wage * hours_worked + 
     lottery_win + 
     2 * sister_gift + 
     remaining_needed) :=
by sorry

end cookies_sold_l2506_250671


namespace sculpture_cost_in_cny_l2506_250694

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating the equivalent cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end sculpture_cost_in_cny_l2506_250694


namespace first_valid_year_is_2913_l2506_250645

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2100 ∧ sum_of_digits year = 15

theorem first_valid_year_is_2913 :
  (∀ y, 2100 < y ∧ y < 2913 → sum_of_digits y ≠ 15) ∧
  is_valid_year 2913 := by
  sorry

end first_valid_year_is_2913_l2506_250645


namespace six_balls_four_boxes_l2506_250613

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 257 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 257 := by sorry

end six_balls_four_boxes_l2506_250613


namespace unique_x_with_square_property_l2506_250656

theorem unique_x_with_square_property : ∃! x : ℕ+, 
  (∃ k : ℕ, (2 * x.val + 1 : ℕ) = k^2) ∧ 
  (∀ y : ℕ, (2 * x.val + 2 : ℕ) ≤ y ∧ y ≤ (3 * x.val + 2) → ¬∃ k : ℕ, y = k^2) ∧
  x = 4 := by
sorry

end unique_x_with_square_property_l2506_250656


namespace y_derivative_l2506_250698

noncomputable def y (x : ℝ) : ℝ := Real.tan (Real.sqrt (Real.cos (1/3))) + (Real.sin (31*x))^2 / (31 * Real.cos (62*x))

theorem y_derivative (x : ℝ) :
  deriv y x = (2 * (Real.sin (31*x) * Real.cos (31*x) * Real.cos (62*x) + Real.sin (31*x)^2 * Real.sin (62*x))) / Real.cos (62*x)^2 :=
sorry

end y_derivative_l2506_250698


namespace increasing_function_implies_a_geq_5_l2506_250604

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem increasing_function_implies_a_geq_5 (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 4 → f a x < f a y) →
  a ≥ 5 :=
sorry

end increasing_function_implies_a_geq_5_l2506_250604


namespace circle_radius_l2506_250600

/-- Circle with center (3, -5) and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 5)^2 = r^2}

/-- Line 4x - 3y - 2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 - 2 = 0}

/-- The shortest distance from a point on the circle to the line is 1 -/
def ShortestDistance (r : ℝ) : Prop :=
  ∃ p ∈ Circle r, ∀ q ∈ Circle r, ∀ l ∈ Line,
    dist p l ≤ dist q l ∧ dist p l = 1

theorem circle_radius (r : ℝ) :
  ShortestDistance r → r = 4 := by
  sorry

end circle_radius_l2506_250600


namespace vase_discount_percentage_l2506_250662

theorem vase_discount_percentage 
  (original_price : ℝ) 
  (total_payment : ℝ) 
  (sales_tax_rate : ℝ) 
  (h1 : original_price = 200)
  (h2 : total_payment = 165)
  (h3 : sales_tax_rate = 0.1)
  : ∃ (discount_percentage : ℝ), 
    discount_percentage = 25 ∧ 
    total_payment = (original_price * (1 - discount_percentage / 100)) * (1 + sales_tax_rate) := by
  sorry

end vase_discount_percentage_l2506_250662


namespace no_xyz_solution_l2506_250683

theorem no_xyz_solution : ¬∃ (x y z : ℕ), 
  0 ≤ x ∧ x ≤ 9 ∧
  0 ≤ y ∧ y ≤ 9 ∧
  0 ≤ z ∧ z ≤ 9 ∧
  100 * x + 10 * y + z = y * (10 * x + z) := by
  sorry

end no_xyz_solution_l2506_250683


namespace delivery_fee_percentage_l2506_250658

def toy_organizer_cost : ℝ := 78
def gaming_chair_cost : ℝ := 83
def toy_organizer_sets : ℕ := 3
def gaming_chairs : ℕ := 2
def total_paid : ℝ := 420

def total_before_fee : ℝ := toy_organizer_cost * toy_organizer_sets + gaming_chair_cost * gaming_chairs

def delivery_fee : ℝ := total_paid - total_before_fee

theorem delivery_fee_percentage : (delivery_fee / total_before_fee) * 100 = 5 := by
  sorry

end delivery_fee_percentage_l2506_250658


namespace cylinder_minus_cones_volume_l2506_250621

theorem cylinder_minus_cones_volume (r h₁ h₂ : ℝ) (hr : r = 10) (hh₁ : h₁ = 15) (hh₂ : h₂ = 30) :
  let v_cyl := π * r^2 * h₂
  let v_cone := (1/3) * π * r^2 * h₁
  v_cyl - 2 * v_cone = 2000 * π := by sorry

end cylinder_minus_cones_volume_l2506_250621


namespace range_of_special_set_l2506_250620

def is_valid_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ c = 10 ∧ (a + b + c) / 3 = 6 ∧ b = 6

theorem range_of_special_set :
  ∀ a b c : ℝ, is_valid_set a b c → c - a = 8 :=
by sorry

end range_of_special_set_l2506_250620


namespace quadratic_m_bounds_l2506_250684

open Complex

/-- Given a quadratic equation x^2 + z₁x + z₂ + m = 0 with complex coefficients,
    prove that under certain conditions, |m| has specific min and max values. -/
theorem quadratic_m_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  z₁^2 - 4*z₂ = 16 + 20*I →
  α^2 + z₁*α + z₂ + m = 0 →
  β^2 + z₁*β + z₂ + m = 0 →
  abs (α - β) = 2 * Real.sqrt 7 →
  (abs m = Real.sqrt 41 - 7 ∨ abs m = Real.sqrt 41 + 7) ∧
  ∀ m' : ℂ, (∃ α' β' : ℂ, α'^2 + z₁*α' + z₂ + m' = 0 ∧
                          β'^2 + z₁*β' + z₂ + m' = 0 ∧
                          abs (α' - β') = 2 * Real.sqrt 7) →
    Real.sqrt 41 - 7 ≤ abs m' ∧ abs m' ≤ Real.sqrt 41 + 7 :=
by sorry

end quadratic_m_bounds_l2506_250684


namespace x_minus_y_range_l2506_250616

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the range of x - y
def range_x_minus_y (x y : ℝ) : Prop :=
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10

-- Theorem statement
theorem x_minus_y_range :
  ∀ (x y ρ θ : ℝ), C ρ θ → x = ρ * Real.cos θ → y = ρ * Real.sin θ → range_x_minus_y x y :=
sorry

end x_minus_y_range_l2506_250616


namespace least_prime_for_integer_roots_l2506_250637

theorem least_prime_for_integer_roots : 
  ∃ (P : ℕ), 
    Prime P ∧ 
    (∃ (x : ℤ), x^2 + 2*(P+1)*x + P^2 - P - 14 = 0) ∧
    (∀ (Q : ℕ), Prime Q ∧ Q < P → ¬∃ (y : ℤ), y^2 + 2*(Q+1)*y + Q^2 - Q - 14 = 0) ∧
    P = 7 :=
sorry

end least_prime_for_integer_roots_l2506_250637


namespace z_value_range_l2506_250638

theorem z_value_range (x y z : ℝ) (sum_eq : x + y + z = 3) (sum_sq_eq : x^2 + y^2 + z^2 = 18) :
  ∃ (z_max z_min : ℝ), 
    (∀ z', (∃ x' y', x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
sorry

end z_value_range_l2506_250638


namespace triangle_equilateral_from_cosine_product_l2506_250675

theorem triangle_equilateral_from_cosine_product (A B C : ℝ) 
  (triangle_condition : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (angle_sum : A + B + C = π) 
  (cosine_product : Real.cos (A - B) * Real.cos (B - C) * Real.cos (C - A) = 1) : 
  A = B ∧ B = C := by
  sorry

end triangle_equilateral_from_cosine_product_l2506_250675


namespace polynomial_remainder_l2506_250690

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 6, if the remainder when q(x) is divided by (x - 2) is 14, 
    then the remainder when q(x) is divided by (x + 2) is also 14 -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^4 + E * x^2 + F * x + 6
  (q 2 = 14) → (q (-2) = 14) := by
  sorry

end polynomial_remainder_l2506_250690


namespace f_greater_than_g_l2506_250673

def f (x : ℝ) : ℝ := 3 * x^2 - x + 1

def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

theorem f_greater_than_g : ∀ x : ℝ, f x > g x := by
  sorry

end f_greater_than_g_l2506_250673


namespace distribute_nine_to_three_l2506_250629

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 9 distinct objects into 3 distinct containers,
    with each container receiving at least one object, is 504 -/
theorem distribute_nine_to_three : distribute 9 3 = 504 := by sorry

end distribute_nine_to_three_l2506_250629


namespace line_parallel_transitive_plane_parallel_transitive_l2506_250691

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallelLine : Line → Line → Prop)

-- Define the parallel relation for planes
variable (parallelPlane : Plane → Plane → Prop)

-- Theorem for lines
theorem line_parallel_transitive (a b c : Line) :
  parallelLine a b → parallelLine a c → parallelLine b c := by sorry

-- Theorem for planes
theorem plane_parallel_transitive (α β γ : Plane) :
  parallelPlane α β → parallelPlane α γ → parallelPlane β γ := by sorry

end line_parallel_transitive_plane_parallel_transitive_l2506_250691


namespace largest_package_size_l2506_250677

theorem largest_package_size (lucy_markers emma_markers : ℕ) 
  (h1 : lucy_markers = 54)
  (h2 : emma_markers = 36) :
  Nat.gcd lucy_markers emma_markers = 18 := by
  sorry

end largest_package_size_l2506_250677


namespace at_least_two_equal_l2506_250650

theorem at_least_two_equal (x y z : ℝ) : 
  (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0 →
  (x = y ∨ y = z ∨ z = x) :=
by sorry

end at_least_two_equal_l2506_250650


namespace weight_of_substance_a_l2506_250631

/-- Given a mixture of substances a and b in the ratio 9:11 with a total weight,
    calculate the weight of substance a in the mixture. -/
theorem weight_of_substance_a (total_weight : ℝ) : 
  total_weight = 58.00000000000001 →
  (9 : ℝ) / (9 + 11) * total_weight = 26.1 := by
  sorry

end weight_of_substance_a_l2506_250631


namespace may_birth_percentage_l2506_250633

def total_mathematicians : ℕ := 120
def may_births : ℕ := 15

theorem may_birth_percentage :
  (may_births : ℚ) / total_mathematicians * 100 = 12.5 := by
  sorry

end may_birth_percentage_l2506_250633


namespace factorial_fraction_equals_one_l2506_250664

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end factorial_fraction_equals_one_l2506_250664


namespace isosceles_triangle_construction_uniqueness_l2506_250611

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  radius : ℝ
  altitude : ℝ
  orthocenter : ℝ
  is_positive : base > 0 ∧ radius > 0 ∧ altitude > 0
  bisects_altitude : orthocenter = altitude / 2

/-- Theorem stating that an isosceles triangle can be uniquely constructed given the base, radius, and orthocenter condition -/
theorem isosceles_triangle_construction_uniqueness 
  (b r : ℝ) 
  (hb : b > 0) 
  (hr : r > 0) : 
  ∃! t : IsoscelesTriangle, t.base = b ∧ t.radius = r :=
sorry

end isosceles_triangle_construction_uniqueness_l2506_250611


namespace inequality_condition_l2506_250605

theorem inequality_condition (a b c : ℝ) :
  (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b → (a + Real.sqrt (b + c) > b + Real.sqrt (a + c))) ↔ c > (1/4 : ℝ) :=
by sorry

end inequality_condition_l2506_250605


namespace surface_area_difference_after_cube_removal_l2506_250674

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the surface area difference after cube removal -/
def surfaceAreaDifference (length width height cubeEdge : ℝ) : ℝ :=
  let originalArea := surfaceArea length width height
  let removedArea := 3 * cubeEdge ^ 2
  let addedArea := cubeEdge ^ 2
  originalArea - removedArea + addedArea - originalArea

theorem surface_area_difference_after_cube_removal :
  surfaceAreaDifference 5 4 3 2 = -8 := by sorry

end surface_area_difference_after_cube_removal_l2506_250674


namespace min_difference_theorem_l2506_250699

noncomputable def f (x : ℝ) : ℝ := Real.exp (4 * x - 1)

noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.log (2 * x)

theorem min_difference_theorem (m n : ℝ) (h : f m = g n) :
  (∀ m' n', f m' = g n' → n' - m' ≥ (1 + Real.log 2) / 4) ∧
  (∃ m₀ n₀, f m₀ = g n₀ ∧ n₀ - m₀ = (1 + Real.log 2) / 4) := by
  sorry

end min_difference_theorem_l2506_250699


namespace supplement_of_complement_of_36_l2506_250688

-- Define the original angle
def original_angle : ℝ := 36

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define the supplement of an angle
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_36 : 
  supplement (complement original_angle) = 126 := by
  sorry

end supplement_of_complement_of_36_l2506_250688


namespace max_geometric_mean_of_sequence_l2506_250643

theorem max_geometric_mean_of_sequence (A : ℝ) (a : Fin 6 → ℝ) :
  (∃ i, a i = 1) →
  (∀ i, i < 4 → (a i + a (i + 1) + a (i + 2)) / 3 = (a (i + 1) + a (i + 2) + a (i + 3)) / 3) →
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5) / 6 = A →
  (∃ i, i < 4 → 
    ∀ j, j < 4 → 
      (a j * a (j + 1) * a (j + 2)) ^ (1/3 : ℝ) ≤ ((3 * A - 1) ^ 2 / 4) ^ (1/3 : ℝ)) :=
by sorry

end max_geometric_mean_of_sequence_l2506_250643


namespace three_points_with_midpoint_l2506_250678

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points on a line
structure Point where
  position : ℝ
  color : Color

-- Define the theorem
theorem three_points_with_midpoint
  (line : Set Point)
  (h_nonempty : Set.Nonempty line)
  (h_two_colors : ∃ p q : Point, p ∈ line ∧ q ∈ line ∧ p.color ≠ q.color)
  (h_one_color : ∀ p : Point, p ∈ line → (p.color = Color.Red ∨ p.color = Color.Blue)) :
  ∃ p q r : Point,
    p ∈ line ∧ q ∈ line ∧ r ∈ line ∧
    p.color = q.color ∧ q.color = r.color ∧
    q.position = (p.position + r.position) / 2 :=
sorry

end three_points_with_midpoint_l2506_250678


namespace garden_breadth_l2506_250680

/-- Given a rectangular garden with perimeter 900 m and length 260 m, prove its breadth is 190 m. -/
theorem garden_breadth (perimeter length breadth : ℝ) : 
  perimeter = 900 ∧ length = 260 ∧ perimeter = 2 * (length + breadth) → breadth = 190 := by
  sorry

end garden_breadth_l2506_250680


namespace last_twelve_average_l2506_250632

theorem last_twelve_average (total_average : ℝ) (first_twelve_average : ℝ) (thirteenth_result : ℝ) :
  total_average = 20 →
  first_twelve_average = 14 →
  thirteenth_result = 128 →
  (25 * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 17 := by
sorry

end last_twelve_average_l2506_250632


namespace flowerbed_perimeter_l2506_250626

/-- The perimeter of a rectangular flowerbed with given dimensions -/
theorem flowerbed_perimeter : 
  let width : ℝ := 4
  let length : ℝ := 2 * width - 1
  2 * (length + width) = 22 := by sorry

end flowerbed_perimeter_l2506_250626


namespace extreme_values_when_a_neg_three_one_intersection_when_a_ge_one_l2506_250681

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x - a

-- Theorem for the extreme values when a = -3
theorem extreme_values_when_a_neg_three :
  (∃ x₁ x₂ : ℝ, f (-3) x₁ = 5 ∧ f (-3) x₂ = -6 ∧
    ∀ x : ℝ, f (-3) x ≤ 5 ∧ f (-3) x ≥ -6) :=
sorry

-- Theorem for the intersection with x-axis when a ≥ 1
theorem one_intersection_when_a_ge_one :
  ∀ a : ℝ, a ≥ 1 →
    ∃! x : ℝ, f a x = 0 :=
sorry

end extreme_values_when_a_neg_three_one_intersection_when_a_ge_one_l2506_250681


namespace legs_heads_difference_l2506_250642

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- The main theorem about the difference between legs and twice the heads -/
theorem legs_heads_difference (group : AnimalGroup) 
    (h : group.cows = 18) : 
    totalLegs group - 2 * totalHeads group = 36 := by
  sorry


end legs_heads_difference_l2506_250642


namespace no_prime_sum_47_l2506_250692

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem no_prime_sum_47 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 47 := by
  sorry

end no_prime_sum_47_l2506_250692


namespace speed_ratio_eddy_freddy_l2506_250644

/-- Proves that the ratio of Eddy's average speed to Freddy's average speed is 34:15 -/
theorem speed_ratio_eddy_freddy :
  let eddy_distance : ℝ := 510  -- km
  let eddy_time : ℝ := 3        -- hours
  let freddy_distance : ℝ := 300  -- km
  let freddy_time : ℝ := 4        -- hours
  let eddy_speed : ℝ := eddy_distance / eddy_time
  let freddy_speed : ℝ := freddy_distance / freddy_time
  (eddy_speed / freddy_speed) = 34 / 15 := by
  sorry


end speed_ratio_eddy_freddy_l2506_250644


namespace field_trip_students_l2506_250647

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 2

/-- The number of buses needed for the trip -/
def number_of_buses : ℕ := 7

/-- The total number of students going on the field trip -/
def total_students : ℕ := seats_per_bus * number_of_buses

theorem field_trip_students : total_students = 14 := by
  sorry

end field_trip_students_l2506_250647


namespace max_min_distance_difference_l2506_250665

/-- Two unit squares with horizontal and vertical sides -/
structure UnitSquare where
  bottomLeft : ℝ × ℝ

/-- The minimum distance between two points -/
def minDistance (s1 s2 : UnitSquare) : ℝ :=
  sorry

/-- The maximum distance between two points -/
def maxDistance (s1 s2 : UnitSquare) : ℝ :=
  sorry

/-- Theorem: The difference between max and min possible y values is 5 - 3√2 -/
theorem max_min_distance_difference (s1 s2 : UnitSquare) 
  (h : minDistance s1 s2 = 5) :
  ∃ (yMin yMax : ℝ),
    yMin ≤ maxDistance s1 s2 ∧ 
    maxDistance s1 s2 ≤ yMax ∧
    yMax - yMin = 5 - 3 * Real.sqrt 2 :=
  sorry

end max_min_distance_difference_l2506_250665


namespace lower_bound_k_squared_l2506_250635

theorem lower_bound_k_squared (k : ℤ) (V : ℤ) (h1 : k^2 > V) (h2 : k^2 < 225) 
  (h3 : ∃ (S : Finset ℤ), S.card ≤ 6 ∧ ∀ x, x ∈ S ↔ x^2 > V ∧ x^2 < 225) :
  81 ≤ k^2 := by
  sorry

end lower_bound_k_squared_l2506_250635


namespace coin_difference_l2506_250652

/-- Represents the number of coins of a specific denomination a person has -/
structure CoinCount where
  fiveRuble : ℕ
  twoRuble : ℕ

/-- Calculates the total value in rubles for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  5 * coins.fiveRuble + 2 * coins.twoRuble

/-- Represents the coin counts for Petya and Vanya -/
structure CoinDistribution where
  petya : CoinCount
  vanya : CoinCount

/-- Checks if the coin distribution satisfies the problem conditions -/
def isValidDistribution (dist : CoinDistribution) : Prop :=
  dist.vanya.fiveRuble = dist.petya.twoRuble ∧
  dist.vanya.twoRuble = dist.petya.fiveRuble ∧
  totalValue dist.petya = totalValue dist.vanya + 60

theorem coin_difference (dist : CoinDistribution) 
  (h : isValidDistribution dist) : 
  dist.petya.fiveRuble - dist.petya.twoRuble = 20 :=
sorry

end coin_difference_l2506_250652


namespace container_capacity_proof_l2506_250624

theorem container_capacity_proof :
  ∀ (C : ℝ),
    (C > 0) →
    (0.3 * C + 27 = 0.75 * C) →
    C = 60 :=
by
  sorry

end container_capacity_proof_l2506_250624


namespace sum_of_squared_differences_zero_l2506_250607

theorem sum_of_squared_differences_zero (x y z : ℝ) :
  (x - 4)^2 + (y - 5)^2 + (z - 6)^2 = 0 → x + y + z = 15 := by
sorry

end sum_of_squared_differences_zero_l2506_250607


namespace circle_placement_possible_l2506_250614

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Theorem: In a 20x25 rectangle with 120 unit squares, there exists a point
    that is at least 0.5 units away from any edge and at least √2/2 units
    away from the center of any unit square -/
theorem circle_placement_possible (rect : Rectangle) 
    (squares : Finset Point) : 
    rect.width = 20 ∧ rect.height = 25 ∧ squares.card = 120 →
    ∃ p : Point, 
      0.5 ≤ p.x ∧ p.x ≤ rect.width - 0.5 ∧
      0.5 ≤ p.y ∧ p.y ≤ rect.height - 0.5 ∧
      ∀ s ∈ squares, (p.x - s.x)^2 + (p.y - s.y)^2 ≥ 0.5 := by
  sorry

end circle_placement_possible_l2506_250614


namespace optimal_robot_purchase_l2506_250602

/-- Represents the robot purchase problem -/
structure RobotPurchase where
  cost_A : ℕ  -- Cost of A robot in yuan
  cost_B : ℕ  -- Cost of B robot in yuan
  capacity_A : ℕ  -- Daily capacity of A robot in tons
  capacity_B : ℕ  -- Daily capacity of B robot in tons
  total_robots : ℕ  -- Total number of robots to purchase
  min_capacity : ℕ  -- Minimum daily capacity required

/-- The optimal solution minimizes the total cost -/
def optimal_solution (rp : RobotPurchase) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating the optimal solution for the given problem -/
theorem optimal_robot_purchase :
  let rp : RobotPurchase := {
    cost_A := 12000,
    cost_B := 20000,
    capacity_A := 90,
    capacity_B := 100,
    total_robots := 30,
    min_capacity := 2830
  }
  let (num_A, num_B, total_cost) := optimal_solution rp
  num_A = 17 ∧ num_B = 13 ∧ total_cost = 464000 :=
by sorry

end optimal_robot_purchase_l2506_250602


namespace stratified_sample_size_l2506_250617

/-- Represents a school population with teachers and students. -/
structure SchoolPopulation where
  teachers : ℕ
  maleStudents : ℕ
  femaleStudents : ℕ

/-- Represents a stratified sample from the school population. -/
structure StratifiedSample where
  totalSize : ℕ
  femalesSampled : ℕ

/-- Theorem: Given the school population and number of females sampled, 
    the total sample size is 192. -/
theorem stratified_sample_size 
  (school : SchoolPopulation)
  (sample : StratifiedSample)
  (h1 : school.teachers = 200)
  (h2 : school.maleStudents = 1200)
  (h3 : school.femaleStudents = 1000)
  (h4 : sample.femalesSampled = 80) :
  sample.totalSize = 192 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l2506_250617


namespace yoga_time_l2506_250655

/-- Mandy's exercise routine -/
def exercise_routine (gym bicycle yoga : ℕ) : Prop :=
  -- Gym to bicycle ratio is 2:3
  3 * gym = 2 * bicycle ∧
  -- Yoga to total exercise ratio is 2:3
  3 * yoga = 2 * (gym + bicycle) ∧
  -- Mandy spends 30 minutes doing yoga
  yoga = 30

/-- Theorem stating that given the exercise routine, yoga time is 30 minutes -/
theorem yoga_time (gym bicycle yoga : ℕ) :
  exercise_routine gym bicycle yoga → yoga = 30 := by
  sorry

end yoga_time_l2506_250655


namespace expression_evaluation_l2506_250693

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^4 + 1) / x) * ((y^4 + 1) / y) + ((x^4 - 1) / y) * ((y^4 - 1) / x) = 2 * x^3 * y^3 + 2 / (x * y) := by
  sorry

end expression_evaluation_l2506_250693


namespace circle_properties_l2506_250661

/-- The circle equation --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y - 12 = 0

/-- The center of the circle --/
def circle_center : ℝ × ℝ := (-2, 3)

/-- The radius of the circle --/
def circle_radius : ℝ := 5

/-- Theorem stating that the given equation represents a circle with the specified center and radius --/
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end circle_properties_l2506_250661


namespace solution_equation_one_solution_equation_two_l2506_250685

-- Problem 1
theorem solution_equation_one (x : ℝ) : 
  3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 := by sorry

-- Problem 2
theorem solution_equation_two (x : ℝ) :
  2 * (x + 1)^3 + 54 = 0 ↔ x = -4 := by sorry

end solution_equation_one_solution_equation_two_l2506_250685


namespace ratio_problem_l2506_250654

theorem ratio_problem (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) :
  (x + y) / (y + z) = 26 / 53 := by
  sorry

end ratio_problem_l2506_250654


namespace min_value_expression_l2506_250660

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x * y = 1) :
  ∃ (t : ℝ), t = 25 ∧ ∀ (z : ℝ), (3 * x^3 + 125 * y^3) / (x - y) ≥ z := by
  sorry

end min_value_expression_l2506_250660


namespace dorothy_profit_l2506_250609

/-- Dorothy's doughnut business profit calculation -/
theorem dorothy_profit (ingredients_cost : ℕ) (num_doughnuts : ℕ) (price_per_doughnut : ℕ) :
  ingredients_cost = 53 →
  num_doughnuts = 25 →
  price_per_doughnut = 3 →
  num_doughnuts * price_per_doughnut - ingredients_cost = 22 :=
by
  sorry

#check dorothy_profit

end dorothy_profit_l2506_250609


namespace largest_four_digit_multiple_of_3_and_5_l2506_250676

theorem largest_four_digit_multiple_of_3_and_5 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ 9990 :=
by sorry

end largest_four_digit_multiple_of_3_and_5_l2506_250676


namespace arccos_of_one_eq_zero_l2506_250627

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by sorry

end arccos_of_one_eq_zero_l2506_250627


namespace circle_equation_implies_value_l2506_250659

theorem circle_equation_implies_value (x y : ℝ) : 
  x^2 + y^2 - 12*x + 16*y + 100 = 0 → (x - 7)^(-y) = 1 := by
  sorry

end circle_equation_implies_value_l2506_250659


namespace tina_pens_theorem_l2506_250610

def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def yellow_pens : ℕ := pink_pens + green_pens - 5
def pens_used_per_day : ℕ := 4

theorem tina_pens_theorem :
  let total_pens := pink_pens + green_pens + blue_pens + yellow_pens
  let days_to_use_pink := (pink_pens + pens_used_per_day - 1) / pens_used_per_day
  total_pens = 46 ∧ days_to_use_pink = 4 := by
  sorry

end tina_pens_theorem_l2506_250610


namespace quadratic_roots_transformation_l2506_250625

theorem quadratic_roots_transformation (D E F : ℝ) (α β : ℝ) (h1 : D ≠ 0) :
  (D * α^2 + E * α + F = 0) →
  (D * β^2 + E * β + F = 0) →
  ∃ (p q : ℝ), (α^2 + 1)^2 + p * (α^2 + 1) + q = 0 ∧
                (β^2 + 1)^2 + p * (β^2 + 1) + q = 0 ∧
                p = (2 * D * F - E^2 - 2 * D^2) / D^2 :=
by sorry

end quadratic_roots_transformation_l2506_250625


namespace fruit_drink_volume_l2506_250657

theorem fruit_drink_volume (orange_percent : ℝ) (watermelon_percent : ℝ) (grape_ounces : ℝ) :
  orange_percent = 0.15 →
  watermelon_percent = 0.60 →
  grape_ounces = 30 →
  ∃ total_ounces : ℝ,
    total_ounces = 120 ∧
    orange_percent * total_ounces + watermelon_percent * total_ounces + grape_ounces = total_ounces :=
by
  sorry

end fruit_drink_volume_l2506_250657


namespace fraction_product_theorem_l2506_250606

theorem fraction_product_theorem : 
  (5 / 4 : ℚ) * (8 / 16 : ℚ) * (20 / 12 : ℚ) * (32 / 64 : ℚ) * 
  (50 / 20 : ℚ) * (40 / 80 : ℚ) * (70 / 28 : ℚ) * (48 / 96 : ℚ) = 625 / 768 := by
  sorry

end fraction_product_theorem_l2506_250606


namespace evaluate_expression_l2506_250623

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end evaluate_expression_l2506_250623


namespace first_oil_price_first_oil_price_is_40_l2506_250640

/-- Given two varieties of oil mixed together, calculate the price of the first variety. -/
theorem first_oil_price 
  (second_oil_volume : ℝ) 
  (second_oil_price : ℝ) 
  (mixture_price : ℝ) 
  (first_oil_volume : ℝ) : ℝ :=
  let total_volume := first_oil_volume + second_oil_volume
  let second_oil_total_cost := second_oil_volume * second_oil_price
  let mixture_total_cost := total_volume * mixture_price
  let first_oil_total_cost := mixture_total_cost - second_oil_total_cost
  first_oil_total_cost / first_oil_volume

/-- The price of the first variety of oil is 40, given the specified conditions. -/
theorem first_oil_price_is_40 : 
  first_oil_price 240 60 52 160 = 40 := by
  sorry

end first_oil_price_first_oil_price_is_40_l2506_250640


namespace arithmetic_expression_equality_l2506_250639

theorem arithmetic_expression_equality : 76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 := by
  sorry

end arithmetic_expression_equality_l2506_250639


namespace jacobs_age_multiple_l2506_250672

/-- Proves that Jacob's age will be 3 times his son's age in five years -/
theorem jacobs_age_multiple (jacob_age son_age : ℕ) : 
  jacob_age = 40 →
  son_age = 10 →
  jacob_age - 5 = 7 * (son_age - 5) →
  (jacob_age + 5) = 3 * (son_age + 5) := by
  sorry

end jacobs_age_multiple_l2506_250672


namespace probability_theorem_l2506_250679

/-- The number of possible outcomes when rolling a single six-sided die -/
def dice_outcomes : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The probability of rolling seven standard six-sided dice and getting at least one pair
    but not a three-of-a-kind -/
def probability_at_least_one_pair_no_three_of_a_kind : ℚ :=
  6426 / 13997

/-- Theorem stating that the probability of rolling seven standard six-sided dice
    and getting at least one pair but not a three-of-a-kind is 6426/13997 -/
theorem probability_theorem :
  probability_at_least_one_pair_no_three_of_a_kind = 6426 / 13997 := by
  sorry

end probability_theorem_l2506_250679


namespace integer_roots_of_polynomial_l2506_250601

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 13

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = {-13, -1, 1, 13} := by sorry

end integer_roots_of_polynomial_l2506_250601


namespace sean_patch_profit_l2506_250686

/-- Sean's patch business profit calculation -/
theorem sean_patch_profit :
  let order_size : ℕ := 100
  let cost_per_patch : ℚ := 125 / 100
  let selling_price : ℚ := 12
  let total_cost : ℚ := order_size * cost_per_patch
  let total_revenue : ℚ := order_size * selling_price
  let net_profit : ℚ := total_revenue - total_cost
  net_profit = 1075 := by sorry

end sean_patch_profit_l2506_250686


namespace arithmetic_sequence_seventh_term_l2506_250648

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 2)
  (h_sum : a 3 + a 5 = 10) :
  a 7 = 8 := by
sorry

end arithmetic_sequence_seventh_term_l2506_250648


namespace bees_after_14_days_l2506_250603

/-- Calculates the total number of bees in a hive after a given number of days -/
def totalBeesAfterDays (initialBees : ℕ) (beesHatchedPerDay : ℕ) (beesLostPerDay : ℕ) (days : ℕ) : ℕ :=
  initialBees + (beesHatchedPerDay - beesLostPerDay) * days + 1

/-- Theorem: Given the specified conditions, the total number of bees after 14 days is 64801 -/
theorem bees_after_14_days :
  totalBeesAfterDays 20000 5000 1800 14 = 64801 := by
  sorry

#eval totalBeesAfterDays 20000 5000 1800 14

end bees_after_14_days_l2506_250603


namespace find_x_l2506_250630

theorem find_x (y z : ℝ) (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
                         (h2 : y + z = 110) : x = 106 := by
  sorry

end find_x_l2506_250630


namespace unique_cube_prime_l2506_250618

theorem unique_cube_prime (n : ℕ) : (∃ p : ℕ, Nat.Prime p ∧ 2^n + n^2 + 25 = p^3) ↔ n = 6 :=
  sorry

end unique_cube_prime_l2506_250618


namespace sum_consecutive_integers_n_plus_3_l2506_250608

theorem sum_consecutive_integers_n_plus_3 (n : ℕ) (h : n = 1) :
  (List.range (n + 3 + 1)).sum = ((n + 3) * (n + 4)) / 2 := by
  sorry

end sum_consecutive_integers_n_plus_3_l2506_250608
