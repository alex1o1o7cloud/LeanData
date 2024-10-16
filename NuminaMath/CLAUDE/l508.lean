import Mathlib

namespace NUMINAMATH_CALUDE_sweets_distribution_l508_50827

/-- The number of children initially supposed to receive sweets -/
def initial_children : ℕ := 190

/-- The number of absent children -/
def absent_children : ℕ := 70

/-- The number of extra sweets each child received due to absences -/
def extra_sweets : ℕ := 14

/-- The total number of sweets each child received -/
def total_sweets : ℕ := 38

theorem sweets_distribution :
  initial_children * (total_sweets - extra_sweets) = 
  (initial_children - absent_children) * total_sweets :=
sorry

end NUMINAMATH_CALUDE_sweets_distribution_l508_50827


namespace NUMINAMATH_CALUDE_optimal_sale_info_l508_50872

/-- Represents the selling prices and quantities of notebooks and sticky notes -/
structure SaleInfo where
  notebook_price : ℝ
  sticky_note_price : ℝ
  notebook_quantity : ℕ
  sticky_note_quantity : ℕ

/-- Calculates the total income given the sale information -/
def total_income (s : SaleInfo) : ℝ :=
  s.notebook_price * s.notebook_quantity + s.sticky_note_price * s.sticky_note_quantity

/-- Theorem stating the optimal selling prices and quantities for maximum income -/
theorem optimal_sale_info :
  ∃ (s : SaleInfo),
    -- Total number of items is 100
    s.notebook_quantity + s.sticky_note_quantity = 100 ∧
    -- 3 notebooks and 2 sticky notes sold for 65 yuan
    3 * s.notebook_price + 2 * s.sticky_note_price = 65 ∧
    -- 4 notebooks and 3 sticky notes sold for 90 yuan
    4 * s.notebook_price + 3 * s.sticky_note_price = 90 ∧
    -- Number of notebooks does not exceed 3 times the number of sticky notes
    s.notebook_quantity ≤ 3 * s.sticky_note_quantity ∧
    -- Notebook price is 15 yuan
    s.notebook_price = 15 ∧
    -- Sticky note price is 10 yuan
    s.sticky_note_price = 10 ∧
    -- Optimal quantities are 75 notebooks and 25 sticky notes
    s.notebook_quantity = 75 ∧
    s.sticky_note_quantity = 25 ∧
    -- Maximum total income is 1375 yuan
    total_income s = 1375 ∧
    -- This is the maximum income
    ∀ (t : SaleInfo),
      t.notebook_quantity + t.sticky_note_quantity = 100 →
      t.notebook_quantity ≤ 3 * t.sticky_note_quantity →
      total_income t ≤ total_income s := by
  sorry

end NUMINAMATH_CALUDE_optimal_sale_info_l508_50872


namespace NUMINAMATH_CALUDE_min_distance_sum_l508_50818

theorem min_distance_sum (x y : ℝ) (hx : x ∈ (Set.Ioo 0 1)) (hy : y ∈ (Set.Ioo 0 1)) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ 
  (Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + (y-1)^2) + 
   Real.sqrt ((x-1)^2 + y^2) + Real.sqrt ((x-1)^2 + (y-1)^2)) ≥ m ∧
  ∃ (x₀ y₀ : ℝ), x₀ ∈ (Set.Ioo 0 1) ∧ y₀ ∈ (Set.Ioo 0 1) ∧
    (Real.sqrt (x₀^2 + y₀^2) + Real.sqrt (x₀^2 + (y₀-1)^2) + 
     Real.sqrt ((x₀-1)^2 + y₀^2) + Real.sqrt ((x₀-1)^2 + (y₀-1)^2)) = m :=
by sorry


end NUMINAMATH_CALUDE_min_distance_sum_l508_50818


namespace NUMINAMATH_CALUDE_cubic_roots_of_27_l508_50855

theorem cubic_roots_of_27 :
  let z₁ : ℂ := 3
  let z₂ : ℂ := -3/2 + (3*Complex.I*Real.sqrt 3)/2
  let z₃ : ℂ := -3/2 - (3*Complex.I*Real.sqrt 3)/2
  (z₁^3 = 27 ∧ z₂^3 = 27 ∧ z₃^3 = 27) ∧
  ∀ z : ℂ, z^3 = 27 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_of_27_l508_50855


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l508_50832

/-- The simple interest calculation problem --/
theorem simple_interest_calculation (P : ℝ) : 
  (∀ (r : ℝ) (A : ℝ), 
    r = 0.04 → 
    A = 36.4 → 
    A = P + P * r) → 
  P = 35 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l508_50832


namespace NUMINAMATH_CALUDE_ab_positive_iff_hyperbola_l508_50861

-- Define the condition for a hyperbola
def is_hyperbola (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x^2 - b * y^2 = 1

-- State the theorem
theorem ab_positive_iff_hyperbola (a b : ℝ) :
  a * b > 0 ↔ is_hyperbola a b :=
sorry

end NUMINAMATH_CALUDE_ab_positive_iff_hyperbola_l508_50861


namespace NUMINAMATH_CALUDE_books_in_fiction_section_l508_50888

theorem books_in_fiction_section 
  (initial_books : ℕ) 
  (books_left : ℕ) 
  (history_books : ℕ) 
  (children_books : ℕ) 
  (wrong_place_books : ℕ) 
  (h1 : initial_books = 51) 
  (h2 : books_left = 16) 
  (h3 : history_books = 12) 
  (h4 : children_books = 8) 
  (h5 : wrong_place_books = 4) : 
  initial_books - books_left - history_books - (children_books - wrong_place_books) = 19 := by
  sorry

end NUMINAMATH_CALUDE_books_in_fiction_section_l508_50888


namespace NUMINAMATH_CALUDE_minimum_cost_is_correct_l508_50826

/-- Represents the dimensions and cost of a box --/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ
  cost : ℚ

/-- Represents the capacity of a box for different painting sizes --/
structure BoxCapacity where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the collection of paintings --/
structure PaintingCollection where
  small : ℕ
  medium : ℕ
  large : ℕ

def smallBox : Box := ⟨20, 20, 15, 4/5⟩
def mediumBox : Box := ⟨22, 22, 17, 11/10⟩
def largeBox : Box := ⟨24, 24, 20, 27/20⟩

def smallBoxCapacity : BoxCapacity := ⟨3, 2, 0⟩
def mediumBoxCapacity : BoxCapacity := ⟨5, 4, 3⟩
def largeBoxCapacity : BoxCapacity := ⟨8, 6, 5⟩

def collection : PaintingCollection := ⟨1350, 2700, 3150⟩

/-- Calculates the minimum cost to move the entire collection --/
def minimumCost (collection : PaintingCollection) (largeBox : Box) (largeBoxCapacity : BoxCapacity) : ℚ :=
  let smallBoxes := (collection.small + largeBoxCapacity.small - 1) / largeBoxCapacity.small
  let mediumBoxes := (collection.medium + largeBoxCapacity.medium - 1) / largeBoxCapacity.medium
  let largeBoxes := (collection.large + largeBoxCapacity.large - 1) / largeBoxCapacity.large
  (smallBoxes + mediumBoxes + largeBoxes) * largeBox.cost

theorem minimum_cost_is_correct :
  minimumCost collection largeBox largeBoxCapacity = 1686.15 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_is_correct_l508_50826


namespace NUMINAMATH_CALUDE_element_in_set_l508_50868

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l508_50868


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l508_50824

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
theorem distance_to_y_axis (A : ℝ × ℝ) : 
  A.1 = -3 → A.2 = 4 → |A.1| = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l508_50824


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l508_50874

/-- The charge difference between two print shops for a given number of copies. -/
def chargeDifference (priceX priceY : ℚ) (copies : ℕ) : ℚ :=
  copies * (priceY - priceX)

/-- The theorem stating the charge difference for 70 color copies between shop Y and shop X. -/
theorem print_shop_charge_difference :
  chargeDifference (1.20 : ℚ) (1.70 : ℚ) 70 = 35 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l508_50874


namespace NUMINAMATH_CALUDE_product_of_one_plus_roots_l508_50822

theorem product_of_one_plus_roots (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 10 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 51 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_roots_l508_50822


namespace NUMINAMATH_CALUDE_horner_method_v2_value_l508_50802

def f (x : ℝ) : ℝ := 4*x^4 + 3*x^3 - 6*x^2 + x - 1

def horner_v2 (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  let v₀ := a₄
  let v₁ := v₀ * x + a₃
  v₁ * x + a₂

theorem horner_method_v2_value :
  horner_v2 4 3 (-6) 1 (-1) (-1) = -5 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_value_l508_50802


namespace NUMINAMATH_CALUDE_prob_two_red_cards_l508_50848

/-- Probability of drawing two red cards in succession from a special deck -/
theorem prob_two_red_cards (total_cards : Nat) (red_cards : Nat) 
  (h1 : total_cards = 60)
  (h2 : red_cards = 36) : 
  (red_cards * (red_cards - 1)) / (total_cards * (total_cards - 1)) = 21 / 59 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_cards_l508_50848


namespace NUMINAMATH_CALUDE_quadrilateral_angle_difference_l508_50858

/-- A quadrilateral with angles in ratio 3:4:5:6 has a difference of 60° between its largest and smallest angles -/
theorem quadrilateral_angle_difference (a b c d : ℝ) : 
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k ∧ d = 6*k →  -- Angles in ratio 3:4:5:6
  (6*k) - (3*k) = 60 :=  -- Difference between largest and smallest angles
by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_difference_l508_50858


namespace NUMINAMATH_CALUDE_total_amount_is_correct_l508_50841

/-- Calculates the final price of a good after applying rebate and sales tax -/
def finalPrice (originalPrice : ℚ) (rebatePercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let reducedPrice := originalPrice * (1 - rebatePercentage / 100)
  reducedPrice * (1 + salesTaxPercentage / 100)

/-- The total amount John has to pay for all goods -/
def totalAmount : ℚ :=
  finalPrice 2500 6 10 + finalPrice 3150 8 12 + finalPrice 1000 5 7

/-- Theorem stating that the total amount John has to pay is equal to 6847.26 -/
theorem total_amount_is_correct : totalAmount = 6847.26 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_correct_l508_50841


namespace NUMINAMATH_CALUDE_quadratic_properties_l508_50886

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 + 6

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  -- The parabola opens upwards
  (∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2) ∧
  -- When x < 2, f(x) decreases as x increases
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) ∧
  -- The axis of symmetry is x = 2
  (∀ x : ℝ, f (2 - x) = f (2 + x)) ∧
  -- f(0) = 10
  f 0 = 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l508_50886


namespace NUMINAMATH_CALUDE_class_average_proof_l508_50891

theorem class_average_proof (x : ℝ) : 
  (0.25 * x + 0.5 * 65 + 0.25 * 90 = 75) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_proof_l508_50891


namespace NUMINAMATH_CALUDE_julie_hourly_rate_l508_50825

/-- Calculates the hourly rate given the following conditions:
  * Hours worked per day
  * Days worked per week
  * Monthly salary when missing one day of work
  * Number of weeks in a month
-/
def calculate_hourly_rate (hours_per_day : ℕ) (days_per_week : ℕ) 
  (monthly_salary_missing_day : ℕ) (weeks_per_month : ℕ) : ℚ :=
  let total_hours := hours_per_day * days_per_week * weeks_per_month - hours_per_day
  monthly_salary_missing_day / total_hours

theorem julie_hourly_rate :
  calculate_hourly_rate 8 6 920 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_julie_hourly_rate_l508_50825


namespace NUMINAMATH_CALUDE_quadratic_radical_always_defined_l508_50836

theorem quadratic_radical_always_defined (x : ℝ) : 0 ≤ x^2 + 2 := by sorry

#check quadratic_radical_always_defined

end NUMINAMATH_CALUDE_quadratic_radical_always_defined_l508_50836


namespace NUMINAMATH_CALUDE_seed_mixture_percentage_l508_50819

/-- Given two seed mixtures X and Y, and a final mixture containing both, 
    this theorem proves the percentage of mixture X in the final mixture. -/
theorem seed_mixture_percentage 
  (x_ryegrass : ℚ) (x_bluegrass : ℚ) (y_ryegrass : ℚ) (y_fescue : ℚ) 
  (final_ryegrass : ℚ) : 
  x_ryegrass = 40 / 100 →
  x_bluegrass = 60 / 100 →
  y_ryegrass = 25 / 100 →
  y_fescue = 75 / 100 →
  final_ryegrass = 38 / 100 →
  x_ryegrass + x_bluegrass = 1 →
  y_ryegrass + y_fescue = 1 →
  ∃ (p : ℚ), p * x_ryegrass + (1 - p) * y_ryegrass = final_ryegrass ∧ 
             p = 260 / 3 :=
by sorry

end NUMINAMATH_CALUDE_seed_mixture_percentage_l508_50819


namespace NUMINAMATH_CALUDE_farmer_truck_count_l508_50821

/-- Proves the number of trucks a farmer has, given tank capacity, tanks per truck, and total water capacity. -/
theorem farmer_truck_count (tank_capacity : ℕ) (tanks_per_truck : ℕ) (total_capacity : ℕ) 
  (h1 : tank_capacity = 150)
  (h2 : tanks_per_truck = 3)
  (h3 : total_capacity = 1350) :
  total_capacity / (tank_capacity * tanks_per_truck) = 3 :=
by sorry

end NUMINAMATH_CALUDE_farmer_truck_count_l508_50821


namespace NUMINAMATH_CALUDE_power_multiplication_l508_50803

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l508_50803


namespace NUMINAMATH_CALUDE_min_c_value_sinusoidal_l508_50896

/-- Given a sinusoidal function y = a * sin(b * x + c) where a > 0 and b > 0,
    if the function reaches its minimum at x = 0,
    then the smallest possible value of c is 3π/2. -/
theorem min_c_value_sinusoidal (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, a * Real.sin (b * x + c) ≥ a * Real.sin c) →
  c ≥ 3 * Real.pi / 2 ∧ 
  ∀ c' ≥ 3 * Real.pi / 2, c' < c → ∃ x : ℝ, a * Real.sin (b * x + c') < a * Real.sin c' :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_sinusoidal_l508_50896


namespace NUMINAMATH_CALUDE_square_rectangle_intersection_l508_50875

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def Square (q : Quadrilateral) : Prop :=
  ∃ s : ℝ, s > 0 ∧
    q.B.1 - q.A.1 = s ∧ q.B.2 - q.A.2 = 0 ∧
    q.C.1 - q.B.1 = 0 ∧ q.C.2 - q.B.2 = s ∧
    q.D.1 - q.C.1 = -s ∧ q.D.2 - q.C.2 = 0 ∧
    q.A.1 - q.D.1 = 0 ∧ q.A.2 - q.D.2 = -s

def Rectangle (q : Quadrilateral) : Prop :=
  ∃ w h : ℝ, w > 0 ∧ h > 0 ∧
    q.B.1 - q.A.1 = w ∧ q.B.2 - q.A.2 = 0 ∧
    q.C.1 - q.B.1 = 0 ∧ q.C.2 - q.B.2 = h ∧
    q.D.1 - q.C.1 = -w ∧ q.D.2 - q.C.2 = 0 ∧
    q.A.1 - q.D.1 = 0 ∧ q.A.2 - q.D.2 = -h

def Perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem square_rectangle_intersection (EFGH IJKL : Quadrilateral) (E Q : ℝ × ℝ) :
  Square EFGH ∧ 
  Rectangle IJKL ∧
  EFGH.A = E ∧
  IJKL.B.1 - IJKL.A.1 = 12 ∧
  IJKL.C.2 - IJKL.B.2 = 8 ∧
  Perpendicular (EFGH.D.1 - EFGH.A.1, EFGH.D.2 - EFGH.A.2) (IJKL.B.1 - IJKL.A.1, IJKL.B.2 - IJKL.A.2) ∧
  (Q.2 - EFGH.D.2) * (IJKL.B.1 - IJKL.A.1) = 1/3 * (IJKL.B.1 - IJKL.A.1) * (IJKL.C.2 - IJKL.B.2) →
  E.2 - Q.2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_intersection_l508_50875


namespace NUMINAMATH_CALUDE_triangle_area_with_perimeter_12_l508_50806

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area_with_perimeter_12 (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → c + a > b → 
  (a : ℝ) * b * (12 : ℝ) / 4 = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_with_perimeter_12_l508_50806


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l508_50893

theorem equidistant_point_on_y_axis :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (2, 5)
  ∃! y : ℝ, 
    ((-3 - 0)^2 + (0 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧
    y = 2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l508_50893


namespace NUMINAMATH_CALUDE_remainder_of_3_19_times_5_7_mod_100_l508_50814

theorem remainder_of_3_19_times_5_7_mod_100 : (3^19 * 5^7) % 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_19_times_5_7_mod_100_l508_50814


namespace NUMINAMATH_CALUDE_smallest_shift_for_even_function_l508_50829

/-- Given a function f(x) = sin(ωx + π/2) where ω > 0, 
    if the distance between adjacent axes of symmetry is 2π
    and shifting f(x) to the left by m units results in an even function,
    then the smallest positive value of m is π/(2ω). -/
theorem smallest_shift_for_even_function (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x + π / 2)
  (∀ x : ℝ, f (x + 2*π/ω) = f x) →  -- distance between adjacent axes of symmetry is 2π
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, f (x + m) = f (-x + m)) →  -- shifting by m results in an even function
  (∃ m : ℝ, m > 0 ∧ 
    (∀ x : ℝ, f (x + m) = f (-x + m)) ∧  -- m shift results in even function
    (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, f (x + m') = f (-x + m')) → m ≤ m') ∧  -- m is the smallest such shift
    m = π / (2 * ω)) :=  -- m equals π/(2ω)
by sorry

end NUMINAMATH_CALUDE_smallest_shift_for_even_function_l508_50829


namespace NUMINAMATH_CALUDE_fraction_not_on_time_l508_50833

/-- Represents the fraction of attendees who are male -/
def male_fraction : ℚ := 3/5

/-- Represents the fraction of male attendees who arrived on time -/
def male_on_time : ℚ := 7/8

/-- Represents the fraction of female attendees who arrived on time -/
def female_on_time : ℚ := 4/5

/-- Theorem stating that the fraction of attendees who did not arrive on time is 3/20 -/
theorem fraction_not_on_time : 
  1 - (male_fraction * male_on_time + (1 - male_fraction) * female_on_time) = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_on_time_l508_50833


namespace NUMINAMATH_CALUDE_probabilities_correct_l508_50883

/-- Represents the color of a ball -/
inductive Color
  | Black
  | White

/-- Represents a bag containing balls -/
structure Bag where
  black : ℕ
  white : ℕ

/-- Calculate the probability of drawing a ball of a specific color from a bag -/
def prob_color (b : Bag) (c : Color) : ℚ :=
  match c with
  | Color.Black => b.black / (b.black + b.white)
  | Color.White => b.white / (b.black + b.white)

/-- The contents of bag A -/
def bag_A : Bag := ⟨2, 2⟩

/-- The contents of bag B -/
def bag_B : Bag := ⟨2, 1⟩

theorem probabilities_correct :
  (prob_color bag_A Color.Black * prob_color bag_B Color.Black = 1/3) ∧
  (prob_color bag_A Color.White * prob_color bag_B Color.White = 1/6) ∧
  (prob_color bag_A Color.White * prob_color bag_B Color.White +
   prob_color bag_A Color.White * prob_color bag_B Color.Black +
   prob_color bag_A Color.Black * prob_color bag_B Color.White = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_probabilities_correct_l508_50883


namespace NUMINAMATH_CALUDE_simplify_expression_l508_50869

theorem simplify_expression : 
  (Real.sqrt (32^(1/5)) - Real.sqrt 7)^2 = 11 - 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l508_50869


namespace NUMINAMATH_CALUDE_min_value_sqrt_fraction_min_value_achieved_l508_50810

theorem min_value_sqrt_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4*a^2 + b^2)).sqrt) / (a * b) ≥ ((1 + 2^(2/3)) * (4 + 2^(2/3)) / 2^(2/3)).sqrt :=
sorry

theorem min_value_achieved (a : ℝ) (ha : a > 0) :
  let b := a * (2^(1/3))
  ((a^2 + b^2) * (4*a^2 + b^2)).sqrt / (a * b) = ((1 + 2^(2/3)) * (4 + 2^(2/3)) / 2^(2/3)).sqrt :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_fraction_min_value_achieved_l508_50810


namespace NUMINAMATH_CALUDE_stratified_sampling_is_most_appropriate_l508_50876

/-- Represents the different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | SystematicSampling
  | StratifiedSampling

/-- Represents a product category --/
structure ProductCategory where
  name : String
  count : Nat

/-- Represents a population of products --/
structure ProductPopulation where
  categories : List ProductCategory
  total : Nat

/-- Determines the most appropriate sampling method for a given product population and sample size --/
def mostAppropriateSamplingMethod (population : ProductPopulation) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- The given product population --/
def givenPopulation : ProductPopulation :=
  { categories := [
      { name := "First-class", count := 10 },
      { name := "Second-class", count := 25 },
      { name := "Defective", count := 5 }
    ],
    total := 40
  }

/-- The given sample size --/
def givenSampleSize : Nat := 8

/-- Theorem stating that Stratified Sampling is the most appropriate method for the given scenario --/
theorem stratified_sampling_is_most_appropriate :
  mostAppropriateSamplingMethod givenPopulation givenSampleSize = SamplingMethod.StratifiedSampling :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_is_most_appropriate_l508_50876


namespace NUMINAMATH_CALUDE_cat_adoptions_correct_l508_50843

/-- The number of families who adopted cats at an animal shelter event -/
def num_cat_adoptions : ℕ := 3

/-- Vet fees for dogs in dollars -/
def dog_fee : ℕ := 15

/-- Vet fees for cats in dollars -/
def cat_fee : ℕ := 13

/-- Number of families who adopted dogs -/
def num_dog_adoptions : ℕ := 8

/-- The fraction of fees donated back to the shelter -/
def donation_fraction : ℚ := 1/3

/-- The amount donated back to the shelter in dollars -/
def donation_amount : ℕ := 53

theorem cat_adoptions_correct : 
  (num_dog_adoptions * dog_fee + num_cat_adoptions * cat_fee) * donation_fraction = donation_amount :=
sorry

end NUMINAMATH_CALUDE_cat_adoptions_correct_l508_50843


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l508_50851

/-- A decagon is a polygon with 10 sides and vertices -/
def Decagon := Nat

/-- The number of vertices in a decagon -/
def num_vertices : Decagon → Nat := fun _ => 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def num_adjacent_vertices : Decagon → Nat := fun _ => 2

/-- The total number of ways to choose the second vertex -/
def total_second_vertex_choices : Decagon → Nat := fun d => num_vertices d - 1

theorem adjacent_vertices_probability (d : Decagon) :
  (num_adjacent_vertices d : ℚ) / (total_second_vertex_choices d) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l508_50851


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l508_50899

theorem smallest_number_of_eggs (total_containers : ℕ) (filled_containers : ℕ) : 
  total_containers > 10 →
  filled_containers = total_containers - 3 →
  15 * filled_containers + 14 * 3 > 150 →
  15 * filled_containers + 14 * 3 ≤ 15 * (filled_containers + 1) + 14 * 3 - 3 →
  15 * filled_containers + 14 * 3 = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l508_50899


namespace NUMINAMATH_CALUDE_exists_multiple_with_odd_digit_sum_l508_50881

/-- Sum of digits of a natural number in decimal notation -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- Theorem: For any natural number M, there exists a multiple of M with an odd sum of digits -/
theorem exists_multiple_with_odd_digit_sum (M : ℕ) : 
  ∃ k : ℕ, M ∣ k ∧ isOdd (sumOfDigits k) := by sorry

end NUMINAMATH_CALUDE_exists_multiple_with_odd_digit_sum_l508_50881


namespace NUMINAMATH_CALUDE_whiteboard_washing_time_l508_50859

/-- If four kids can wash three whiteboards in 20 minutes, 
    then one kid can wash six whiteboards in 160 minutes. -/
theorem whiteboard_washing_time 
  (wash_rate : ℝ) -- Rate at which kids wash whiteboards (whiteboards per kid per minute)
  (h : 4 * wash_rate * 20 = 3) -- Four kids can wash three whiteboards in 20 minutes
  : 1 * wash_rate * 160 = 6 := by -- One kid can wash six whiteboards in 160 minutes
sorry

end NUMINAMATH_CALUDE_whiteboard_washing_time_l508_50859


namespace NUMINAMATH_CALUDE_sales_and_profit_theorem_l508_50892

/-- Represents the monthly sales quantity as a function of selling price -/
def monthly_sales (x : ℝ) : ℝ := -30 * x + 960

/-- Represents the monthly profit as a function of selling price -/
def monthly_profit (x : ℝ) : ℝ := (x - 10) * (monthly_sales x)

theorem sales_and_profit_theorem :
  let cost_price : ℝ := 10
  let price1 : ℝ := 20
  let price2 : ℝ := 30
  let sales1 : ℝ := 360
  let sales2 : ℝ := 60
  let target_profit : ℝ := 3600
  (∀ x, monthly_sales x = -30 * x + 960) ∧
  (monthly_sales price1 = sales1) ∧
  (monthly_sales price2 = sales2) ∧
  (∃ x, monthly_profit x = target_profit) ∧
  (monthly_profit 22 = target_profit) ∧
  (monthly_profit 20 = target_profit) := by
  sorry

#check sales_and_profit_theorem

end NUMINAMATH_CALUDE_sales_and_profit_theorem_l508_50892


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l508_50820

theorem player_a_not_losing_probability 
  (prob_draw : ℝ) 
  (prob_a_win : ℝ) 
  (h1 : prob_draw = 0.4) 
  (h2 : prob_a_win = 0.4) : 
  prob_draw + prob_a_win = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l508_50820


namespace NUMINAMATH_CALUDE_relatively_prime_2n_plus_1_and_4n_squared_plus_1_l508_50847

theorem relatively_prime_2n_plus_1_and_4n_squared_plus_1 (n : ℕ+) :
  Nat.gcd (2 * n.val + 1) (4 * n.val^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_relatively_prime_2n_plus_1_and_4n_squared_plus_1_l508_50847


namespace NUMINAMATH_CALUDE_distance_to_focus_l508_50813

/-- Given a parabola y^2 = 4x and a point M(x₀, 2√3) on it, 
    the distance from M to the focus of the parabola is 4. -/
theorem distance_to_focus (x₀ : ℝ) : 
  (2 * Real.sqrt 3)^2 = 4 * x₀ →   -- Point M is on the parabola
  x₀ + 1 = 4 :=                    -- Distance to focus
by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l508_50813


namespace NUMINAMATH_CALUDE_max_ab_value_l508_50877

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (2 : ℝ) / (2 * a - 3) * (b / 2) = 1) : a * b ≤ 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l508_50877


namespace NUMINAMATH_CALUDE_batsman_average_increase_l508_50878

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  initialInnings : ℕ
  newScore : ℝ
  newAverage : ℝ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (b : Batsman) : ℝ :=
  b.newAverage - b.initialAverage

/-- Theorem: The batsman's average increases by 5 runs -/
theorem batsman_average_increase (b : Batsman) 
  (h1 : b.initialInnings = 10)
  (h2 : b.newScore = 95)
  (h3 : b.newAverage = 45)
  : averageIncrease b = 5 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l508_50878


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l508_50889

theorem adult_ticket_cost (num_students : Nat) (num_adults : Nat) (student_ticket_cost : Nat) (total_cost : Nat) :
  num_students = 12 →
  num_adults = 4 →
  student_ticket_cost = 1 →
  total_cost = 24 →
  (total_cost - num_students * student_ticket_cost) / num_adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l508_50889


namespace NUMINAMATH_CALUDE_nine_five_times_theorem_l508_50863

def is_valid_expression (expr : ℕ → Bool) : Prop :=
  ∃ (a b c d e : ℕ), 
    (a = 9 ∧ b = 9 ∧ c = 9 ∧ d = 9 ∧ e = 9) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 13 → expr n = true)

theorem nine_five_times_theorem : 
  ∃ expr : ℕ → Bool, is_valid_expression expr :=
sorry

end NUMINAMATH_CALUDE_nine_five_times_theorem_l508_50863


namespace NUMINAMATH_CALUDE_willam_farm_tax_l508_50815

/-- Farm tax calculation for Mr. Willam -/
theorem willam_farm_tax (total_tax : ℝ) (willam_percentage : ℝ) :
  let willam_tax := total_tax * (willam_percentage / 100)
  willam_tax = total_tax * (willam_percentage / 100) :=
by sorry

#check willam_farm_tax 3840 27.77777777777778

end NUMINAMATH_CALUDE_willam_farm_tax_l508_50815


namespace NUMINAMATH_CALUDE_homework_problem_l508_50894

theorem homework_problem (a b c d : ℤ) 
  (h1 : a = -1) 
  (h2 : b = -c) 
  (h3 : d = -2) : 
  4*a + (b + c) - |3*d| = -10 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l508_50894


namespace NUMINAMATH_CALUDE_opposite_vertices_distance_l508_50801

def cube_side_length : ℝ := 2

theorem opposite_vertices_distance (cube_side : ℝ) (h : cube_side = cube_side_length) :
  let diagonal := Real.sqrt (3 * cube_side ^ 2)
  diagonal = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_opposite_vertices_distance_l508_50801


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l508_50853

theorem arithmetic_expression_evaluation : 7 + 15 / 3 - 5 * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l508_50853


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l508_50830

/-- Given a parabola and a line with exactly one intersection point, prove a specific algebraic identity. -/
theorem parabola_line_intersection (m : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m) → 
  m^36 + 1155 / m^12 = 39236 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l508_50830


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l508_50871

/-- Given that when x = 3, the value of px³ + qx + 3 is 2005, 
    prove that when x = -3, the value of px³ + qx + 3 is -1999 -/
theorem algebraic_expression_value (p q : ℝ) : 
  (3^3 * p + 3 * q + 3 = 2005) → ((-3)^3 * p + (-3) * q + 3 = -1999) := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l508_50871


namespace NUMINAMATH_CALUDE_largest_mediocre_number_l508_50828

def is_mediocre (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = (100 * a + 10 * b + c +
       100 * a + 10 * c + b +
       100 * b + 10 * a + c +
       100 * b + 10 * c + a +
       100 * c + 10 * a + b +
       100 * c + 10 * b + a) / 6

theorem largest_mediocre_number :
  is_mediocre 629 ∧ ∀ n : ℕ, is_mediocre n → n ≤ 629 :=
sorry

end NUMINAMATH_CALUDE_largest_mediocre_number_l508_50828


namespace NUMINAMATH_CALUDE_age_difference_l508_50873

theorem age_difference (age1 age2 : ℕ) : 
  age1 + age2 = 27 → age1 = 13 → age2 = 14 → age2 - age1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l508_50873


namespace NUMINAMATH_CALUDE_bela_has_winning_strategy_l508_50849

/-- Represents a stick with a certain length -/
structure Stick :=
  (length : ℕ)

/-- Represents the game state -/
structure GameState :=
  (sticks : List Stick)

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- Represents a player's strategy -/
def Strategy := GameState → Stick

/-- Represents the initial game state with 99 sticks -/
def initialGameState : GameState :=
  { sticks := List.range 99 |>.map (fun n => ⟨n + 1⟩) }

/-- Represents Béla's winning strategy -/
noncomputable def belasStrategy : Strategy :=
  sorry

/-- Theorem stating that Béla has a winning strategy -/
theorem bela_has_winning_strategy :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.sticks.length = 3 →
      ¬(∃ (s1 s2 s3 : Stick), s1 ∈ game.sticks ∧ s2 ∈ game.sticks ∧ s3 ∈ game.sticks ∧ canFormTriangle s1 s2 s3) :=
by
  sorry

end NUMINAMATH_CALUDE_bela_has_winning_strategy_l508_50849


namespace NUMINAMATH_CALUDE_circle_radius_is_zero_l508_50804

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * x + 2 * y^2 + 4 * y + 10 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 0

/-- Theorem: The radius of the circle defined by the equation is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y → ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_zero_l508_50804


namespace NUMINAMATH_CALUDE_cube_root_of_64_l508_50840

theorem cube_root_of_64 (m : ℝ) : (64 : ℝ)^(1/3) = 2^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l508_50840


namespace NUMINAMATH_CALUDE_problem_line_direction_cosines_l508_50831

/-- The line defined by two equations -/
structure Line where
  eq1 : ℝ → ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ → ℝ

/-- Direction cosines of a line -/
structure DirectionCosines where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The specific line from the problem -/
def problemLine : Line where
  eq1 := fun x y z => 2*x - 3*y - 3*z - 9
  eq2 := fun x y z => x - 2*y + z + 3

/-- Compute direction cosines for a given line -/
noncomputable def computeDirectionCosines (l : Line) : DirectionCosines :=
  { α := 9 / Real.sqrt 107
  , β := 5 / Real.sqrt 107
  , γ := 1 / Real.sqrt 107 }

/-- Theorem: The direction cosines of the problem line are (9/√107, 5/√107, 1/√107) -/
theorem problem_line_direction_cosines :
  computeDirectionCosines problemLine = 
  { α := 9 / Real.sqrt 107
  , β := 5 / Real.sqrt 107
  , γ := 1 / Real.sqrt 107 } := by
  sorry

end NUMINAMATH_CALUDE_problem_line_direction_cosines_l508_50831


namespace NUMINAMATH_CALUDE_min_value_theorem_l508_50856

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (b^2 + b*c) + 1 / (c^2 + c*a) + 1 / (a^2 + a*b) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l508_50856


namespace NUMINAMATH_CALUDE_towel_shrinkage_l508_50817

theorem towel_shrinkage (L B : ℝ) (h_positive : L > 0 ∧ B > 0) : 
  let new_length := 0.8 * L
  let new_area := 0.72 * (L * B)
  ∃ (new_breadth : ℝ), new_area = new_length * new_breadth ∧ new_breadth = 0.9 * B :=
sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l508_50817


namespace NUMINAMATH_CALUDE_min_tangent_length_l508_50846

/-- The minimum length of a tangent from a point on the line x - y + 1 = 0 to the circle (x - 2)² + (y + 1)² = 1 is √7 -/
theorem min_tangent_length (x y : ℝ) : 
  let line := {(x, y) | x - y + 1 = 0}
  let circle := {(x, y) | (x - 2)^2 + (y + 1)^2 = 1}
  let tangent_length (p : ℝ × ℝ) := 
    Real.sqrt ((p.1 - 2)^2 + (p.2 + 1)^2 - 1)
  ∃ (p : ℝ × ℝ), p ∈ line ∧ 
    ∀ (q : ℝ × ℝ), q ∈ line → tangent_length p ≤ tangent_length q ∧
    tangent_length p = Real.sqrt 7 :=
by sorry


end NUMINAMATH_CALUDE_min_tangent_length_l508_50846


namespace NUMINAMATH_CALUDE_sarah_interview_combinations_l508_50852

/-- Represents the number of interview choices for each day of the week -/
structure WeekChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of interview combinations for the week -/
def totalCombinations (choices : WeekChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Represents Sarah's interview choices for the week -/
def sarahChoices : WeekChoices :=
  { monday := 1
  , tuesday := 2
  , wednesday := 5  -- 2 + 3, accounting for both Tuesday possibilities
  , thursday := 5
  , friday := 1 }  -- No interviews, but included for completeness

/-- Theorem stating that Sarah's total interview combinations is 50 -/
theorem sarah_interview_combinations :
  totalCombinations sarahChoices = 50 := by
  sorry

#eval totalCombinations sarahChoices  -- Should output 50

end NUMINAMATH_CALUDE_sarah_interview_combinations_l508_50852


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l508_50809

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a = 5 → b = 12 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 13 ∨ c = Real.sqrt 119 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l508_50809


namespace NUMINAMATH_CALUDE_power_of_negative_64_l508_50864

theorem power_of_negative_64 : (-64 : ℝ) ^ (7/6 : ℝ) = -128 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_64_l508_50864


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l508_50834

/-- Given parametric equations representing a curve, prove that it forms a hyperbola -/
theorem curve_is_hyperbola (θ : ℝ) (h_θ : ∀ n : ℤ, θ ≠ n * π / 2) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = ((Real.exp t + Real.exp (-t)) / 2) * Real.cos θ) ∧
    (∀ t, y t = ((Real.exp t - Real.exp (-t)) / 2) * Real.sin θ) →
    ∀ t, (x t)^2 / (Real.cos θ)^2 - (y t)^2 / (Real.sin θ)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l508_50834


namespace NUMINAMATH_CALUDE_symmetry_axes_symmetry_origin_l508_50890

-- Define the curve C
def C (x y : ℝ) : Prop :=
  ((x + 1)^2 + y^2) * ((x - 1)^2 + y^2) = 2

-- Theorem for symmetry with respect to axes
theorem symmetry_axes :
  (∀ x y, C x y ↔ C (-x) y) ∧ (∀ x y, C x y ↔ C x (-y)) :=
sorry

-- Theorem for symmetry with respect to origin
theorem symmetry_origin :
  ∀ x y, C x y ↔ C (-x) (-y) :=
sorry

end NUMINAMATH_CALUDE_symmetry_axes_symmetry_origin_l508_50890


namespace NUMINAMATH_CALUDE_max_customers_interviewed_l508_50880

theorem max_customers_interviewed (total : ℕ) (impulsive : ℕ) (ad_influence_ratio : ℚ) (consultant_ratio : ℚ) : 
  total ≤ 50 ∧
  impulsive = 7 ∧
  ad_influence_ratio = 3/4 ∧
  consultant_ratio = 1/3 ∧
  (∃ k : ℕ, total - impulsive = 4 * k) ∧
  (ad_influence_ratio * (total - impulsive)).isInt ∧
  (consultant_ratio * ad_influence_ratio * (total - impulsive)).isInt →
  total ≤ 47 ∧ 
  (∃ max_total : ℕ, max_total = 47 ∧ 
    max_total ≤ 50 ∧
    (∃ k : ℕ, max_total - impulsive = 4 * k) ∧
    (ad_influence_ratio * (max_total - impulsive)).isInt ∧
    (consultant_ratio * ad_influence_ratio * (max_total - impulsive)).isInt) :=
by sorry

end NUMINAMATH_CALUDE_max_customers_interviewed_l508_50880


namespace NUMINAMATH_CALUDE_expected_composite_count_l508_50882

/-- The number of elements in the set {1, 2, 3, ..., 100} -/
def setSize : ℕ := 100

/-- The number of composite numbers in the set {1, 2, 3, ..., 100} -/
def compositeCount : ℕ := 74

/-- The number of selections made -/
def selectionCount : ℕ := 5

/-- The probability of selecting a composite number -/
def compositeProbability : ℚ := compositeCount / setSize

/-- Expected number of composite numbers when selecting 5 numbers with replacement from {1, 2, 3, ..., 100} -/
theorem expected_composite_count : 
  (selectionCount : ℚ) * compositeProbability = 37 / 10 := by sorry

end NUMINAMATH_CALUDE_expected_composite_count_l508_50882


namespace NUMINAMATH_CALUDE_quadratic_root_and_m_l508_50837

/-- Given a quadratic equation x^2 + 2x + m = 0 where 2 is a root,
    prove that the other root is -4 and m = -8 -/
theorem quadratic_root_and_m (m : ℝ) : 
  (2 : ℝ)^2 + 2*2 + m = 0 → 
  (∃ (other_root : ℝ), other_root = -4 ∧ 
   other_root^2 + 2*other_root + m = 0 ∧ 
   m = -8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_and_m_l508_50837


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l508_50898

/-- Given an arithmetic sequence {aₙ} with sum Sₙ of its first n terms, 
    if S₉ = a₄ + a₅ + a₆ + 72, then a₃ + a₇ = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →  -- Definition of Sₙ for arithmetic sequence
  (∀ n k, a (n + k) - a n = k * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  S 9 = a 4 + a 5 + a 6 + 72 →  -- Given condition
  a 3 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l508_50898


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l508_50895

theorem stratified_sampling_male_athletes 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 48) 
  (h2 : total_female = 36) 
  (h3 : sample_size = 21) : 
  ℕ :=
  12

#check stratified_sampling_male_athletes

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l508_50895


namespace NUMINAMATH_CALUDE_circle_radius_proof_l508_50812

theorem circle_radius_proof (r : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h_r_pos : r > 0)
  (h_circle₁ : x₁^2 + y₁^2 = r^2)
  (h_circle₂ : x₂^2 + y₂^2 = r^2)
  (h_sum₁ : x₁ + y₁ = 3)
  (h_sum₂ : x₂ + y₂ = 3)
  (h_product : x₁ * x₂ + y₁ * y₂ = -1/2 * r^2) :
  r = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l508_50812


namespace NUMINAMATH_CALUDE_book_profit_rate_l508_50807

/-- Calculate the rate of profit given the cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at Rs 50 and sold at Rs 80 is 60% -/
theorem book_profit_rate :
  let cost_price : ℚ := 50
  let selling_price : ℚ := 80
  rate_of_profit cost_price selling_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l508_50807


namespace NUMINAMATH_CALUDE_range_of_m_l508_50839

/-- Proposition p: m + 2 < 0 -/
def p (m : ℝ) : Prop := m + 2 < 0

/-- Proposition q: the equation x^2 + mx + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≠ 0

/-- The range of real numbers for m given the conditions -/
theorem range_of_m (m : ℝ) (h1 : ¬¬p m) (h2 : ¬(p m ∧ q m)) : m < -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l508_50839


namespace NUMINAMATH_CALUDE_cubic_function_properties_l508_50862

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem cubic_function_properties (a b : ℝ) :
  (f a b 1 = 4) →
  ((3 * a * (-2)^2 + 2 * b * (-2)) = 0) →
  (a = 1 ∧ b = 3) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Ioo m (m + 1), (3 * x^2 + 6 * x) ≥ 0) ↔ (m ≤ -3 ∨ m ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l508_50862


namespace NUMINAMATH_CALUDE_power_of_two_geq_double_l508_50842

theorem power_of_two_geq_double (n : ℕ) : 2^n ≥ 2*n := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_geq_double_l508_50842


namespace NUMINAMATH_CALUDE_jane_visited_six_more_l508_50850

/-- The number of rehabilitation centers visited by each person --/
structure RehabCenters where
  lisa : ℕ
  jude : ℕ
  han : ℕ
  jane : ℕ

/-- The conditions of the problem --/
def problem_conditions (rc : RehabCenters) : Prop :=
  rc.lisa = 6 ∧
  rc.jude = rc.lisa / 2 ∧
  rc.han = 2 * rc.jude - 2 ∧
  rc.jane > 2 * rc.han ∧
  rc.lisa + rc.jude + rc.han + rc.jane = 27

/-- The theorem to be proved --/
theorem jane_visited_six_more (rc : RehabCenters) : 
  problem_conditions rc → rc.jane = 2 * rc.han + 6 := by
  sorry


end NUMINAMATH_CALUDE_jane_visited_six_more_l508_50850


namespace NUMINAMATH_CALUDE_matrix_vector_computation_l508_50866

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w u : Fin 2 → ℝ)

theorem matrix_vector_computation
  (hv : M.mulVec v = ![2, 6])
  (hw : M.mulVec w = ![3, -5])
  (hu : M.mulVec u = ![-1, 4]) :
  M.mulVec (2 • v - w + 4 • u) = ![-3, 33] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l508_50866


namespace NUMINAMATH_CALUDE_harrys_father_age_difference_l508_50857

/-- Proves that Harry's father is 24 years older than Harry given the problem conditions -/
theorem harrys_father_age_difference : 
  ∀ (harry_age father_age mother_age : ℕ),
    harry_age = 50 →
    father_age > harry_age →
    mother_age = harry_age + 22 →
    father_age = mother_age + harry_age / 25 →
    father_age - harry_age = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_harrys_father_age_difference_l508_50857


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l508_50845

theorem geometric_mean_minimum (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h_geom_mean : z^2 = x*y) : 
  (Real.log z) / (4 * Real.log x) + (Real.log z) / (Real.log y) ≥ 9/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l508_50845


namespace NUMINAMATH_CALUDE_range_of_f_l508_50867

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≤ -Real.sqrt 2 ∨ y ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l508_50867


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_linear_expression_factorization_l508_50879

-- Problem 1
theorem difference_of_squares_factorization (x : ℝ) :
  4 * x^2 - 9 = (2*x + 3) * (2*x - 3) := by sorry

-- Problem 2
theorem linear_expression_factorization (a b x y : ℝ) :
  2*a*(x - y) - 3*b*(y - x) = (x - y)*(2*a + 3*b) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_linear_expression_factorization_l508_50879


namespace NUMINAMATH_CALUDE_carter_ate_twelve_green_mms_l508_50897

/-- Represents the number of M&Ms of each color in the jar -/
structure JarContents where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- The initial state of the jar -/
def initial_jar : JarContents := { green := 20, red := 20, yellow := 0 }

/-- The state of the jar after all actions -/
def final_jar (green_eaten : ℕ) : JarContents :=
  { green := initial_jar.green - green_eaten,
    red := initial_jar.red / 2,
    yellow := 14 }

/-- The total number of M&Ms in the jar -/
def total_mms (jar : JarContents) : ℕ := jar.green + jar.red + jar.yellow

/-- The probability of picking a green M&M from the jar -/
def green_probability (jar : JarContents) : ℚ :=
  jar.green / (total_mms jar : ℚ)

theorem carter_ate_twelve_green_mms :
  ∃ (green_eaten : ℕ),
    green_eaten ≤ initial_jar.green ∧
    green_probability (final_jar green_eaten) = 1/4 ∧
    green_eaten = 12 := by sorry

end NUMINAMATH_CALUDE_carter_ate_twelve_green_mms_l508_50897


namespace NUMINAMATH_CALUDE_unique_box_filling_l508_50800

/-- Represents a rectangular parallelepiped with integer dimensions -/
structure Brick where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a brick -/
def Brick.volume (b : Brick) : ℕ := b.length * b.width * b.height

/-- The box to be filled -/
def box : Brick := ⟨10, 11, 14⟩

/-- The first type of brick -/
def brickA : Brick := ⟨2, 5, 8⟩

/-- The second type of brick -/
def brickB : Brick := ⟨2, 3, 7⟩

/-- Theorem stating that the only way to fill the box is with 14 bricks of type A and 10 of type B -/
theorem unique_box_filling :
  ∀ (x y : ℕ), 
    x * brickA.volume + y * brickB.volume = box.volume → 
    (x = 14 ∧ y = 10) := by sorry

end NUMINAMATH_CALUDE_unique_box_filling_l508_50800


namespace NUMINAMATH_CALUDE_solution_set_l508_50860

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_condition (x : ℕ) : Prop :=
  is_prime (3 * x + 1) ∧ 70 ≤ (3 * x + 1) ∧ (3 * x + 1) ≤ 110

theorem solution_set :
  {x : ℕ | satisfies_condition x} = {24, 26, 32, 34, 36} :=
sorry

end NUMINAMATH_CALUDE_solution_set_l508_50860


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l508_50885

/-- Represents the profit calculation for a lemonade stand -/
theorem lemonade_stand_profit :
  ∀ (lemon_cost sugar_cost cup_cost : ℕ) 
    (price_per_cup cups_sold : ℕ),
  lemon_cost = 10 →
  sugar_cost = 5 →
  cup_cost = 3 →
  price_per_cup = 4 →
  cups_sold = 21 →
  cups_sold * price_per_cup - (lemon_cost + sugar_cost + cup_cost) = 66 := by
sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_l508_50885


namespace NUMINAMATH_CALUDE_calculation_results_l508_50808

theorem calculation_results : 
  ((-12) - (-20) + (-8) - 15 = -15) ∧
  (-3^2 + (2/3 - 1/2 + 5/8) * (-24) = -28) ∧
  (-1^2023 + 3 * (-2)^2 - (-6) / (-1/3)^2 = 65) := by
sorry

end NUMINAMATH_CALUDE_calculation_results_l508_50808


namespace NUMINAMATH_CALUDE_cambridge_population_l508_50887

-- Define the number of people in Cambridge
variable (n : ℕ)

-- Define the total amount of water and apple juice consumed
variable (W A : ℝ)

-- Define the mayor's drink
variable (L : ℝ)

-- Each person drinks 12 ounces
axiom total_drink : W + A = 12 * n

-- The mayor's drink is 12 ounces
axiom mayor_drink : L = 12

-- The mayor drinks 1/6 of total water and 1/8 of total apple juice
axiom mayor_portions : L = (1/6) * W + (1/8) * A

-- All drinks have positive amounts of both liquids
axiom positive_amounts : W > 0 ∧ A > 0

-- Theorem: The number of people in Cambridge is 7
theorem cambridge_population : n = 7 :=
sorry

end NUMINAMATH_CALUDE_cambridge_population_l508_50887


namespace NUMINAMATH_CALUDE_minimum_cost_purchase_l508_50823

/-- Represents the unit price and quantity of an ingredient -/
structure Ingredient where
  price : ℝ
  quantity : ℝ

/-- Represents the purchase of two ingredients -/
structure Purchase where
  A : Ingredient
  B : Ingredient

def total_cost (p : Purchase) : ℝ := p.A.price * p.A.quantity + p.B.price * p.B.quantity

def total_quantity (p : Purchase) : ℝ := p.A.quantity + p.B.quantity

theorem minimum_cost_purchase :
  ∀ (p : Purchase),
    p.A.price + p.B.price = 68 →
    5 * p.A.price + 3 * p.B.price = 280 →
    total_quantity p = 36 →
    p.A.quantity ≥ 2 * p.B.quantity →
    total_cost p ≥ 1272 ∧
    (total_cost p = 1272 ↔ p.A.quantity = 24 ∧ p.B.quantity = 12) :=
by sorry

end NUMINAMATH_CALUDE_minimum_cost_purchase_l508_50823


namespace NUMINAMATH_CALUDE_correct_system_is_valid_l508_50811

/-- Represents the purchase of labor tools by a school -/
structure ToolPurchase where
  x : ℕ  -- number of type A tools
  y : ℕ  -- number of type B tools
  total_tools : x + y = 145
  total_cost : 10 * x + 12 * y = 1580

/-- The correct system of equations for the tool purchase -/
def correct_system (p : ToolPurchase) : Prop :=
  (p.x + p.y = 145) ∧ (10 * p.x + 12 * p.y = 1580)

/-- Theorem stating that the given system of equations is correct -/
theorem correct_system_is_valid (p : ToolPurchase) : correct_system p := by
  sorry

#check correct_system_is_valid

end NUMINAMATH_CALUDE_correct_system_is_valid_l508_50811


namespace NUMINAMATH_CALUDE_exists_regular_polygon_with_special_diagonals_l508_50865

/-- A regular polygon is a polygon with all sides and angles equal. -/
structure RegularPolygon where
  n : ℕ
  sides : n > 2

/-- A diagonal of a polygon is a line segment that connects two non-adjacent vertices. -/
def Diagonal (p : RegularPolygon) := Unit

/-- The length of a diagonal in a regular polygon. -/
noncomputable def diagonalLength (p : RegularPolygon) (d : Diagonal p) : ℝ := sorry

/-- Theorem: There exists a regular polygon where the length of one diagonal
    is equal to the sum of the lengths of two other diagonals. -/
theorem exists_regular_polygon_with_special_diagonals :
  ∃ (p : RegularPolygon) (d₁ d₂ d₃ : Diagonal p),
    diagonalLength p d₁ = diagonalLength p d₂ + diagonalLength p d₃ :=
sorry

end NUMINAMATH_CALUDE_exists_regular_polygon_with_special_diagonals_l508_50865


namespace NUMINAMATH_CALUDE_min_value_trig_sum_l508_50844

theorem min_value_trig_sum (θ : ℝ) : 
  1 / (2 - Real.cos θ ^ 2) + 1 / (2 - Real.sin θ ^ 2) ≥ 4 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_sum_l508_50844


namespace NUMINAMATH_CALUDE_honey_shop_problem_l508_50835

/-- The honey shop problem -/
theorem honey_shop_problem (bulk_price tax min_spend penny_paid : ℚ)
  (h1 : bulk_price = 5)
  (h2 : tax = 1)
  (h3 : min_spend = 40)
  (h4 : penny_paid = 240) :
  (penny_paid / (bulk_price + tax) - min_spend / bulk_price) = 32 := by
  sorry

end NUMINAMATH_CALUDE_honey_shop_problem_l508_50835


namespace NUMINAMATH_CALUDE_correct_number_of_small_boxes_l508_50854

-- Define the number of chocolate bars in each small box
def chocolates_per_small_box : ℕ := 26

-- Define the total number of chocolate bars in the large box
def total_chocolates : ℕ := 442

-- Define the number of small boxes in the large box
def num_small_boxes : ℕ := total_chocolates / chocolates_per_small_box

-- Theorem statement
theorem correct_number_of_small_boxes : num_small_boxes = 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_small_boxes_l508_50854


namespace NUMINAMATH_CALUDE_cos_330_degrees_l508_50805

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l508_50805


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l508_50884

theorem min_distance_circle_line : 
  ∃ (d : ℝ), d = 2 ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ - Real.sqrt 3 / 2)^2 + (y₁ - 1/2)^2 = 1) →
    (Real.sqrt 3 * x₂ + y₂ = 8) →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ - Real.sqrt 3 / 2)^2 + (y₁ - 1/2)^2 = 1) ∧
    (Real.sqrt 3 * x₂ + y₂ = 8) ∧
    d = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_circle_line_l508_50884


namespace NUMINAMATH_CALUDE_sarah_age_l508_50870

/-- Given the ages of Sarah, Mark, Billy, and Ana, prove Sarah's age -/
theorem sarah_age (sarah mark billy ana : ℕ) 
  (h1 : sarah = 3 * mark - 4)
  (h2 : mark = billy + 4)
  (h3 : billy = ana / 2)
  (h4 : ana + 3 = 15) : 
  sarah = 26 := by
  sorry

end NUMINAMATH_CALUDE_sarah_age_l508_50870


namespace NUMINAMATH_CALUDE_fourth_group_draw_l508_50816

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_items : ℕ
  num_groups : ℕ
  first_draw : ℕ
  items_per_group : ℕ

/-- Calculates the number drawn in a given group for a systematic sampling -/
def draw_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_draw + s.items_per_group * (group - 1)

/-- Theorem: In the given systematic sampling, the number drawn in the fourth group is 22 -/
theorem fourth_group_draw (s : SystematicSampling) 
  (h1 : s.total_items = 30)
  (h2 : s.num_groups = 5)
  (h3 : s.first_draw = 4)
  (h4 : s.items_per_group = 6) :
  draw_in_group s 4 = 22 := by
  sorry


end NUMINAMATH_CALUDE_fourth_group_draw_l508_50816


namespace NUMINAMATH_CALUDE_floor_a_equals_1994_minus_n_l508_50838

def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (a n)^2 / (a n + 1)

theorem floor_a_equals_1994_minus_n (n : ℕ) (h : n ≤ 998) :
  ⌊a n⌋ = 1994 - n :=
by sorry

end NUMINAMATH_CALUDE_floor_a_equals_1994_minus_n_l508_50838
