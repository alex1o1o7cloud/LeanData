import Mathlib

namespace symmetry_of_abs_f_shifted_l3698_369804

-- Define a function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the property of |f(x)| being an even function
def abs_f_is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |f x| = |f (-x)|

-- State the theorem
theorem symmetry_of_abs_f_shifted (h : abs_f_is_even f) :
  ∀ y : ℝ, |f ((1 - y) - 1)| = |f ((1 + y) - 1)| :=
by
  sorry

end symmetry_of_abs_f_shifted_l3698_369804


namespace cubic_equation_one_real_root_l3698_369890

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 :=
by sorry

end cubic_equation_one_real_root_l3698_369890


namespace hardwood_flooring_area_l3698_369867

/-- Represents the dimensions of a rectangular area -/
structure RectangularArea where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular area -/
def area (r : RectangularArea) : ℝ := r.length * r.width

/-- Represents Nancy's bathroom -/
structure Bathroom where
  centralArea : RectangularArea
  hallway : RectangularArea

/-- The actual bathroom dimensions -/
def nancysBathroom : Bathroom :=
  { centralArea := { length := 10, width := 10 }
  , hallway := { length := 6, width := 4 } }

/-- Theorem: The total area of hardwood flooring in Nancy's bathroom is 124 square feet -/
theorem hardwood_flooring_area :
  area nancysBathroom.centralArea + area nancysBathroom.hallway = 124 := by
  sorry

end hardwood_flooring_area_l3698_369867


namespace insurance_payment_count_l3698_369838

/-- Calculates the number of insurance payments per year -/
def insurance_payments_per_year (quarterly_payment : ℕ) (annual_total : ℕ) : ℕ :=
  annual_total / quarterly_payment

/-- Proves that the number of insurance payments per year is 4 -/
theorem insurance_payment_count :
  insurance_payments_per_year 378 1512 = 4 := by
  sorry

end insurance_payment_count_l3698_369838


namespace closest_to_product_l3698_369883

def product : ℝ := 0.001532 * 2134672

def options : List ℝ := [3100, 3150, 3200, 3500, 4000]

theorem closest_to_product : 
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |product - x| ≤ |product - y| ∧
  x = 3150 :=
sorry

end closest_to_product_l3698_369883


namespace min_value_problem_l3698_369868

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) : 
  (1 / x + 1 / (3 * y)) ≥ 4 := by
  sorry

end min_value_problem_l3698_369868


namespace max_value_condition_l3698_369806

theorem max_value_condition (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 4, |x^2 - 4*x + 9 - 2*m| + 2*m ≤ 9) ∧ 
  (∃ x ∈ Set.Icc 0 4, |x^2 - 4*x + 9 - 2*m| + 2*m = 9) ↔ 
  m ≤ 7/2 :=
sorry

end max_value_condition_l3698_369806


namespace coefficient_of_x_cubed_l3698_369819

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 2 * (x^3 - 2*x^2 + x) + 4 * (x^4 + 3*x^3 - x^2 + x) - 3 * (x - 5*x^3 + 2*x^5)
  ∃ (a b c d e : ℝ), expression = a*x^5 + b*x^4 + 29*x^3 + c*x^2 + d*x + e :=
by sorry

end coefficient_of_x_cubed_l3698_369819


namespace line_slope_m_values_l3698_369875

theorem line_slope_m_values (m : ℝ) : 
  (∃ a b c : ℝ, (m^2 + m - 4) * a + (m + 4) * b + (2 * m + 1) = c ∧ 
   (m^2 + m - 4) = -(m + 4) ∧ (m^2 + m - 4) ≠ 0) → 
  m = 0 ∨ m = -2 := by
sorry

end line_slope_m_values_l3698_369875


namespace median_is_twelve_l3698_369892

def group_sizes : List ℕ := [10, 10, 8]

def median (l : List ℕ) (x : ℕ) : ℚ :=
  sorry

theorem median_is_twelve (x : ℕ) : median (x :: group_sizes) x = 12 :=
  sorry

end median_is_twelve_l3698_369892


namespace unique_solution_l3698_369809

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Checks if a number satisfies the first division scheme -/
def satisfies_first_scheme (n : FourDigitNumber) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ 
  (n.value / d = 10 + (n.value / 100 % 10)) ∧
  (n.value % d = (n.value / 10 % 10) * 10 + (n.value % 10))

/-- Checks if a number satisfies the second division scheme -/
def satisfies_second_scheme (n : FourDigitNumber) : Prop :=
  ∃ (d : ℕ), d < 10 ∧
  (n.value / d = 168) ∧
  (n.value % d = 0)

/-- The main theorem stating that 1512 is the only number satisfying both schemes -/
theorem unique_solution : 
  ∃! (n : FourDigitNumber), 
    satisfies_first_scheme n ∧ 
    satisfies_second_scheme n ∧ 
    n.value = 1512 := by
  sorry

end unique_solution_l3698_369809


namespace mary_total_spent_approx_l3698_369828

/-- Calculates the total amount Mary spent at the mall --/
def total_spent (shirt_price : ℝ) (shirt_tax : ℝ) 
                (jacket_price : ℝ) (jacket_discount : ℝ) (jacket_tax : ℝ) 
                (currency_rate : ℝ)
                (scarf_price : ℝ) (hat_price : ℝ) (accessories_tax : ℝ) : ℝ :=
  let shirt_total := shirt_price * (1 + shirt_tax)
  let jacket_discounted := jacket_price * (1 - jacket_discount)
  let jacket_total := jacket_discounted * (1 + jacket_tax) * currency_rate
  let accessories_total := (scarf_price + hat_price) * (1 + accessories_tax)
  shirt_total + jacket_total + accessories_total

/-- The theorem stating that Mary's total spent is approximately $49.13 --/
theorem mary_total_spent_approx :
  ∃ ε > 0, abs (total_spent 13.04 0.07 15.34 0.20 0.085 1.28 7.90 9.13 0.065 - 49.13) < ε :=
by
  sorry

end mary_total_spent_approx_l3698_369828


namespace scooter_gain_percent_correct_l3698_369805

def scooter_gain_percent (purchase_price repair1 repair2 repair3 taxes maintenance selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair1 + repair2 + repair3 + taxes + maintenance
  let gain := selling_price - total_cost
  (gain / total_cost) * 100

theorem scooter_gain_percent_correct 
  (purchase_price repair1 repair2 repair3 taxes maintenance selling_price : ℚ) :
  scooter_gain_percent purchase_price repair1 repair2 repair3 taxes maintenance selling_price =
  ((selling_price - (purchase_price + repair1 + repair2 + repair3 + taxes + maintenance)) / 
   (purchase_price + repair1 + repair2 + repair3 + taxes + maintenance)) * 100 :=
by sorry

end scooter_gain_percent_correct_l3698_369805


namespace remainder_9_pow_2048_mod_50_l3698_369876

theorem remainder_9_pow_2048_mod_50 : 9^2048 % 50 = 21 := by
  sorry

end remainder_9_pow_2048_mod_50_l3698_369876


namespace chord_length_polar_curves_l3698_369878

/-- The length of the chord formed by the intersection of two curves in polar coordinates -/
theorem chord_length_polar_curves : 
  ∃ (ρ₁ ρ₂ : ℝ → ℝ) (θ₁ θ₂ : ℝ),
    (∀ θ, ρ₁ θ * Real.sin θ = 1) →
    (∀ θ, ρ₂ θ = 4 * Real.sin θ) →
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      x₁^2 + y₁^2 = (ρ₁ θ₁)^2 ∧
      x₂^2 + y₂^2 = (ρ₁ θ₂)^2 ∧
      x₁ = ρ₁ θ₁ * Real.cos θ₁ ∧
      y₁ = ρ₁ θ₁ * Real.sin θ₁ ∧
      x₂ = ρ₁ θ₂ * Real.cos θ₂ ∧
      y₂ = ρ₁ θ₂ * Real.sin θ₂ ∧
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 2 * Real.sqrt 3 :=
by sorry

end chord_length_polar_curves_l3698_369878


namespace inverse_direct_variation_l3698_369844

/-- Given positive real numbers x, y, and z satisfying the following conditions:
    1. x² and y vary inversely
    2. y and z vary directly
    3. y = 8 when x = 4
    4. z = 32 when x = 4
    Prove that z = 512 when x = 1 -/
theorem inverse_direct_variation (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x^2 * y = k)
  (h_direct : ∃ c : ℝ, ∀ y z, y / z = c)
  (h_y : y = 8 → x = 4)
  (h_z : z = 32 → x = 4) :
  x = 1 → z = 512 := by
  sorry

end inverse_direct_variation_l3698_369844


namespace dealership_sedan_sales_l3698_369863

/-- Represents the ratio of sports cars to sedans -/
structure CarRatio :=
  (sports : ℕ)
  (sedans : ℕ)

/-- Calculates the expected sedan sales given a car ratio and anticipated sports car sales -/
def expectedSedanSales (ratio : CarRatio) (anticipatedSportsCars : ℕ) : ℕ :=
  (anticipatedSportsCars * ratio.sedans) / ratio.sports

theorem dealership_sedan_sales :
  let ratio : CarRatio := ⟨3, 5⟩
  let anticipatedSportsCars : ℕ := 36
  expectedSedanSales ratio anticipatedSportsCars = 60 := by
  sorry

end dealership_sedan_sales_l3698_369863


namespace no_x_term_l3698_369872

theorem no_x_term (m : ℝ) : (∀ x : ℝ, (x + m) * (x + 3) = x^2 + 3*m) → m = -3 := by
  sorry

end no_x_term_l3698_369872


namespace fraction_multiplication_l3698_369891

theorem fraction_multiplication : (1/4 - 1/2 + 2/3) * (-12 : ℚ) = -8 := by
  sorry

end fraction_multiplication_l3698_369891


namespace circle_radius_from_chords_l3698_369814

/-- Given a circle with two chords AB and AC, where AB = a, AC = b, and the length of arc AC is twice
    the length of arc AB, prove that the radius R of the circle is equal to a^2 / sqrt(4a^2 - b^2). -/
theorem circle_radius_from_chords (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R = a^2 / Real.sqrt (4 * a^2 - b^2) := by
  sorry

end circle_radius_from_chords_l3698_369814


namespace mike_found_four_more_seashells_l3698_369879

/-- The number of seashells Mike initially found -/
def initial_seashells : ℝ := 6.0

/-- The total number of seashells Mike ended up with -/
def total_seashells : ℝ := 10

/-- The number of additional seashells Mike found -/
def additional_seashells : ℝ := total_seashells - initial_seashells

theorem mike_found_four_more_seashells : additional_seashells = 4.0 := by
  sorry

end mike_found_four_more_seashells_l3698_369879


namespace warehouse_length_calculation_l3698_369825

/-- Represents the dimensions and walking pattern around a rectangular warehouse. -/
structure Warehouse :=
  (width : ℝ)
  (length : ℝ)
  (circles : ℕ)
  (total_distance : ℝ)

/-- Theorem stating the length of the warehouse given specific conditions. -/
theorem warehouse_length_calculation (w : Warehouse) 
  (h1 : w.width = 400)
  (h2 : w.circles = 8)
  (h3 : w.total_distance = 16000)
  : w.length = 600 := by
  sorry

#check warehouse_length_calculation

end warehouse_length_calculation_l3698_369825


namespace karl_process_preserves_swapped_pairs_l3698_369880

/-- Represents a permutation of cards -/
def Permutation := List Nat

/-- Counts the number of swapped pairs (inversions) in a permutation -/
def countSwappedPairs (p : Permutation) : Nat :=
  sorry

/-- Karl's process of rearranging cards -/
def karlProcess (p : Permutation) : Permutation :=
  sorry

theorem karl_process_preserves_swapped_pairs (n : Nat) (initial : Permutation) :
  initial.length = n →
  initial.toFinset = Finset.range n →
  countSwappedPairs initial = countSwappedPairs (karlProcess initial) :=
sorry

end karl_process_preserves_swapped_pairs_l3698_369880


namespace max_sections_five_lines_l3698_369888

/-- The maximum number of sections a rectangle can be divided into by n line segments -/
def maxSections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => maxSections m + m + 1

/-- Theorem stating that 5 line segments can divide a rectangle into at most 16 sections -/
theorem max_sections_five_lines :
  maxSections 5 = 16 := by
  sorry

end max_sections_five_lines_l3698_369888


namespace solve_for_A_l3698_369861

theorem solve_for_A : ∀ A : ℤ, A + 10 = 15 → A = 5 := by
  sorry

end solve_for_A_l3698_369861


namespace plane_perpendicular_condition_l3698_369818

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_condition 
  (a : Line) (α β : Plane) :
  perpendicular a β ∧ parallel a α → perp α β :=
sorry

end plane_perpendicular_condition_l3698_369818


namespace star_four_three_l3698_369882

-- Define the new operation
def star (a b : ℤ) : ℤ := a^2 - a*b + b^2

-- State the theorem
theorem star_four_three : star 4 3 = 13 := by
  sorry

end star_four_three_l3698_369882


namespace aaron_position_2015_l3698_369816

/-- Represents a point on a 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Defines Aaron's walking pattern -/
def walk (n : Nat) : Point :=
  sorry

/-- The theorem to be proved -/
theorem aaron_position_2015 : walk 2015 = Point.mk 22 13 := by
  sorry

end aaron_position_2015_l3698_369816


namespace reciprocal_of_negative_one_fifth_l3698_369858

theorem reciprocal_of_negative_one_fifth :
  ((-1 : ℚ) / 5)⁻¹ = -5 := by sorry

end reciprocal_of_negative_one_fifth_l3698_369858


namespace shipping_weight_calculation_l3698_369822

/-- The maximum weight a shipping box can hold in pounds, given the initial number of plates,
    weight of each plate, and number of plates removed. -/
def max_shipping_weight (initial_plates : ℕ) (plate_weight : ℕ) (removed_plates : ℕ) : ℚ :=
  ((initial_plates - removed_plates) * plate_weight : ℚ) / 16

theorem shipping_weight_calculation :
  max_shipping_weight 38 10 6 = 20 := by
  sorry

end shipping_weight_calculation_l3698_369822


namespace candy_bar_cost_l3698_369823

theorem candy_bar_cost (marvin_sales : ℕ) (tina_sales : ℕ) (price : ℚ) : 
  marvin_sales = 35 →
  tina_sales = 3 * marvin_sales →
  tina_sales * price = marvin_sales * price + 140 →
  price = 2 := by sorry

end candy_bar_cost_l3698_369823


namespace prove_newly_added_groups_l3698_369803

/-- Represents the number of groups of students recently added to the class -/
def newly_added_groups : ℕ := 2

theorem prove_newly_added_groups :
  let tables : ℕ := 6
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_students : ℕ := 3 * bathroom_students
  let students_per_group : ℕ := 4
  let foreign_students : ℕ := 3 * 3  -- 3 each from 3 countries
  let total_students : ℕ := 47
  newly_added_groups = 
    (total_students - (tables * students_per_table + bathroom_students + canteen_students + foreign_students)) / students_per_group :=
by
  sorry

#check prove_newly_added_groups

end prove_newly_added_groups_l3698_369803


namespace tip_percentage_lower_limit_l3698_369813

theorem tip_percentage_lower_limit 
  (meal_cost : ℝ) 
  (total_paid : ℝ) 
  (tip_percentage : ℝ → Prop) : 
  meal_cost = 35.50 →
  total_paid = 40.825 →
  (∀ x, tip_percentage x → x ≥ 15 ∧ x < 25) →
  total_paid = meal_cost + (meal_cost * (15 / 100)) :=
by sorry

end tip_percentage_lower_limit_l3698_369813


namespace existence_of_triangle_l3698_369849

theorem existence_of_triangle (l : Fin 7 → ℝ) 
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    l i + l j > l k ∧ 
    l j + l k > l i ∧ 
    l k + l i > l j :=
sorry

end existence_of_triangle_l3698_369849


namespace circles_have_another_common_tangent_l3698_369893

-- Define the basic geometric objects
structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the given conditions
def Semicircle (k : Circle) (A B : Point) : Prop :=
  k.center = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) ∧
  k.radius = (((B.x - A.x)^2 + (B.y - A.y)^2)^(1/2)) / 2

def OnCircle (P : Point) (k : Circle) : Prop :=
  (P.x - k.center.x)^2 + (P.y - k.center.y)^2 = k.radius^2

def Perpendicular (C D : Point) (A B : Point) : Prop :=
  (B.x - A.x) * (C.x - D.x) + (B.y - A.y) * (C.y - D.y) = 0

def Incircle (k : Circle) (A B C : Point) : Prop :=
  -- Definition of incircle omitted for brevity
  sorry

def TouchesSegmentAndCircle (k : Circle) (C D : Point) (semicircle : Circle) : Prop :=
  -- Definition of touching segment and circle omitted for brevity
  sorry

def CommonTangent (k1 k2 k3 : Circle) (A B : Point) : Prop :=
  -- Definition of common tangent omitted for brevity
  sorry

-- Main theorem
theorem circles_have_another_common_tangent
  (k semicircle : Circle) (A B C D : Point) (k1 k2 k3 : Circle) :
  Semicircle semicircle A B →
  OnCircle C semicircle →
  C ≠ A ∧ C ≠ B →
  Perpendicular C D A B →
  Incircle k1 A B C →
  TouchesSegmentAndCircle k2 C D semicircle →
  TouchesSegmentAndCircle k3 C D semicircle →
  CommonTangent k1 k2 k3 A B →
  ∃ (E F : Point), E ≠ F ∧ CommonTangent k1 k2 k3 E F ∧ (E ≠ A ∨ F ≠ B) :=
by
  sorry

end circles_have_another_common_tangent_l3698_369893


namespace arithmetic_progression_squares_l3698_369839

/-- An arithmetic progression is represented by its first term and common difference. -/
structure ArithmeticProgression where
  a : ℤ  -- First term
  d : ℤ  -- Common difference

/-- A term in an arithmetic progression. -/
def ArithmeticProgression.term (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  ap.a + n * ap.d

/-- Predicate to check if a number is a perfect square. -/
def is_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k * k

/-- An arithmetic progression contains a square. -/
def contains_square (ap : ArithmeticProgression) : Prop :=
  ∃ n : ℕ, is_square (ap.term n)

/-- An arithmetic progression contains infinitely many squares. -/
def contains_infinite_squares (ap : ArithmeticProgression) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ is_square (ap.term n)

/-- 
If an infinite arithmetic progression contains a square number, 
then it contains infinitely many square numbers.
-/
theorem arithmetic_progression_squares 
  (ap : ArithmeticProgression) 
  (h : contains_square ap) : 
  contains_infinite_squares ap :=
sorry

end arithmetic_progression_squares_l3698_369839


namespace only_f₁_is_quadratic_l3698_369856

-- Define the four functions
def f₁ (x : ℝ) : ℝ := -3 * x^2
def f₂ (x : ℝ) : ℝ := 2 * x
def f₃ (x : ℝ) : ℝ := x + 1
def f₄ (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be quadratic
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- State the theorem
theorem only_f₁_is_quadratic :
  is_quadratic f₁ ∧ ¬is_quadratic f₂ ∧ ¬is_quadratic f₃ ∧ ¬is_quadratic f₄ :=
sorry

end only_f₁_is_quadratic_l3698_369856


namespace five_percent_of_255_l3698_369853

theorem five_percent_of_255 : 
  let percent_5 : ℝ := 0.05
  255 * percent_5 = 12.75 := by
  sorry

end five_percent_of_255_l3698_369853


namespace pure_imaginary_implies_tan_value_l3698_369871

theorem pure_imaginary_implies_tan_value (θ : ℝ) :
  (Complex.I * (Complex.cos θ - 4/5) = Complex.sin θ - 3/5) →
  Real.tan θ = -3/4 := by
  sorry

end pure_imaginary_implies_tan_value_l3698_369871


namespace only_negative_three_less_than_negative_two_l3698_369826

theorem only_negative_three_less_than_negative_two :
  let numbers : List ℚ := [-3, -1/2, 0, 2]
  ∀ x ∈ numbers, x < -2 ↔ x = -3 :=
by sorry

end only_negative_three_less_than_negative_two_l3698_369826


namespace other_solution_quadratic_l3698_369800

theorem other_solution_quadratic (x : ℚ) :
  (72 * (3/8)^2 + 37 = -95 * (3/8) + 12) →
  (72 * x^2 + 37 = -95 * x + 12) →
  (x ≠ 3/8) →
  x = 5/8 := by
sorry

end other_solution_quadratic_l3698_369800


namespace sin_2010_degrees_l3698_369836

theorem sin_2010_degrees : Real.sin (2010 * π / 180) = -1 / 2 := by
  sorry

end sin_2010_degrees_l3698_369836


namespace square_roots_theorem_l3698_369840

theorem square_roots_theorem (a : ℝ) (n : ℝ) : 
  (2 * a + 3)^2 = n ∧ (a - 18)^2 = n → n = 169 := by
  sorry

end square_roots_theorem_l3698_369840


namespace base_conversion_and_addition_l3698_369865

/-- Converts a number from base 8 to base 10 -/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10To7 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 7 -/
def addBase7 (a b : ℕ) : ℕ := sorry

theorem base_conversion_and_addition :
  addBase7 (base10To7 (base8To10 123)) 25 = 264 := by sorry

end base_conversion_and_addition_l3698_369865


namespace initial_ribbon_tape_length_l3698_369884

/-- The initial length of ribbon tape Yujin had, in meters. -/
def initial_length : ℝ := 8.9

/-- The length of ribbon tape required for one ribbon, in meters. -/
def ribbon_length : ℝ := 0.84

/-- The number of ribbons made. -/
def num_ribbons : ℕ := 10

/-- The length of remaining ribbon tape, in meters. -/
def remaining_length : ℝ := 0.5

/-- Theorem stating that the initial length of ribbon tape equals 8.9 meters. -/
theorem initial_ribbon_tape_length :
  initial_length = ribbon_length * num_ribbons + remaining_length := by
  sorry

end initial_ribbon_tape_length_l3698_369884


namespace repeating_decimal_to_fraction_l3698_369842

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 2 + 35 / 99) ∧ (x = 233 / 99) := by
  sorry

end repeating_decimal_to_fraction_l3698_369842


namespace max_value_expression_l3698_369874

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26*a*b*c) ≤ 3 := by
  sorry

end max_value_expression_l3698_369874


namespace initial_money_calculation_l3698_369866

theorem initial_money_calculation (X : ℝ) : 
  X * (1 - (0.30 + 0.25 + 0.15)) = 3500 → X = 11666.67 := by
  sorry

end initial_money_calculation_l3698_369866


namespace integer_pairs_equation_difficulty_l3698_369860

theorem integer_pairs_equation_difficulty : ¬ ∃ (count : ℕ), 
  (∀ m n : ℤ, m^2 + n^2 = m*n + 3 → count > 0) ∧ 
  (∀ k : ℕ, k ≠ count → ¬(∀ m n : ℤ, m^2 + n^2 = m*n + 3 → k > 0)) :=
sorry

end integer_pairs_equation_difficulty_l3698_369860


namespace tangent_slope_at_one_l3698_369894

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end tangent_slope_at_one_l3698_369894


namespace square_of_complex_l3698_369847

theorem square_of_complex : (3 - Complex.I) ^ 2 = 8 - 6 * Complex.I := by
  sorry

end square_of_complex_l3698_369847


namespace constant_prime_sequence_l3698_369846

-- Define the sequence of prime numbers
def isPrimeSequence (p : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 1 → Nat.Prime (p n)

-- Define the recurrence relation
def satisfiesRecurrence (p : ℕ → ℕ) (k : ℤ) : Prop :=
  ∀ n, n ≥ 1 → p (n + 2) = p (n + 1) + p n + k

-- Theorem statement
theorem constant_prime_sequence
  (p : ℕ → ℕ) (k : ℤ)
  (h_prime : isPrimeSequence p)
  (h_recurrence : satisfiesRecurrence p k) :
  ∃ c, ∀ n, n ≥ 1 → p n = c ∧ Nat.Prime c :=
by sorry

end constant_prime_sequence_l3698_369846


namespace total_money_found_l3698_369827

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 10

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 3

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 4

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 200

theorem total_money_found :
  (num_quarters : ℚ) * quarter_value +
  (num_dimes : ℚ) * dime_value +
  (num_nickels : ℚ) * nickel_value +
  (num_pennies : ℚ) * penny_value = 5 := by
  sorry

end total_money_found_l3698_369827


namespace language_school_solution_l3698_369895

/-- Represents the state of the language school at a given time --/
structure SchoolState where
  num_teachers : ℕ
  total_age : ℕ

/-- The language school problem --/
def language_school_problem (initial : SchoolState) 
  (new_teacher_age : ℕ) (left_teacher_age : ℕ) : Prop :=
  -- Initial state (2007)
  initial.num_teachers = 7 ∧
  -- State after new teacher joins (2010)
  (initial.total_age + 21 + new_teacher_age) / 8 = initial.total_age / 7 ∧
  -- State after one teacher leaves (2012)
  (initial.total_age + 37 + new_teacher_age - left_teacher_age) / 7 = initial.total_age / 7 ∧
  -- New teacher's age in 2010
  new_teacher_age = 25

theorem language_school_solution (initial : SchoolState) 
  (new_teacher_age : ℕ) (left_teacher_age : ℕ) 
  (h : language_school_problem initial new_teacher_age left_teacher_age) :
  left_teacher_age = 62 ∧ initial.total_age / 7 = 46 := by
  sorry

#check language_school_solution

end language_school_solution_l3698_369895


namespace expression_equalities_l3698_369834

theorem expression_equalities : 
  (1 / (Real.sqrt 2 - 1) + Real.sqrt 3 * (Real.sqrt 3 - Real.sqrt 6) + Real.sqrt 8 = 4) ∧
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6) := by
  sorry

end expression_equalities_l3698_369834


namespace baseball_cost_calculation_l3698_369841

/-- The amount spent on marbles in dollars -/
def marbles_cost : ℚ := 9.05

/-- The amount spent on the football in dollars -/
def football_cost : ℚ := 4.95

/-- The total amount spent on toys in dollars -/
def total_cost : ℚ := 20.52

/-- The amount spent on the baseball in dollars -/
def baseball_cost : ℚ := total_cost - (marbles_cost + football_cost)

theorem baseball_cost_calculation :
  baseball_cost = 6.52 := by sorry

end baseball_cost_calculation_l3698_369841


namespace recursive_sum_value_l3698_369812

def recursive_sum (n : ℕ) : ℚ :=
  if n = 0 then 3
  else (n + 3 : ℚ) + (1 / 3) * recursive_sum (n - 1)

theorem recursive_sum_value : 
  recursive_sum 3000 = 4504 - (1 / 4) * (1 - 1 / 3^2999) :=
by sorry

end recursive_sum_value_l3698_369812


namespace quadratic_sum_l3698_369848

/-- A quadratic function g(x) = dx^2 + ex + f passing through (1, 3) and (2, 0) with vertex at (3, -3) -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := λ x => d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (QuadraticFunction d e f 1 = 3) →
  (QuadraticFunction d e f 2 = 0) →
  (∀ x, QuadraticFunction d e f x ≥ QuadraticFunction d e f 3) →
  (QuadraticFunction d e f 3 = -3) →
  d + e + 2 * f = 19.5 := by
  sorry

end quadratic_sum_l3698_369848


namespace smallest_base_perfect_square_base_11_perfect_square_eleven_is_smallest_l3698_369808

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 5 → (∃ n : ℕ, 4 * b + 5 = n * n) → b ≥ 11 :=
by
  sorry

theorem base_11_perfect_square : 
  ∃ n : ℕ, 4 * 11 + 5 = n * n :=
by
  sorry

theorem eleven_is_smallest : 
  ∀ b : ℕ, b > 5 ∧ b < 11 → ¬(∃ n : ℕ, 4 * b + 5 = n * n) :=
by
  sorry

end smallest_base_perfect_square_base_11_perfect_square_eleven_is_smallest_l3698_369808


namespace quadratic_roots_sum_l3698_369817

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 2*m - 7 = 0) → (n^2 + 2*n - 7 = 0) → m^2 + 3*m + n = 5 := by
  sorry

end quadratic_roots_sum_l3698_369817


namespace inscribed_hexagon_area_l3698_369854

/-- A regular hexagon inscribed in a semicircle -/
structure InscribedHexagon where
  /-- The diameter of the semicircle -/
  diameter : ℝ
  /-- One side of the hexagon lies along the diameter -/
  side_on_diameter : Bool
  /-- Two opposite vertices of the hexagon are on the semicircle -/
  vertices_on_semicircle : Bool

/-- The area of an inscribed hexagon -/
def area (h : InscribedHexagon) : ℝ := sorry

/-- Theorem: The area of a regular hexagon inscribed in a semicircle of diameter 1 is 3√3/26 -/
theorem inscribed_hexagon_area :
  ∀ (h : InscribedHexagon), h.diameter = 1 → h.side_on_diameter = true → h.vertices_on_semicircle = true →
  area h = 3 * Real.sqrt 3 / 26 := by sorry

end inscribed_hexagon_area_l3698_369854


namespace greenfield_basketball_association_l3698_369815

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost for one player's home game equipment in dollars -/
def home_cost : ℕ := 2 * sock_cost + tshirt_cost

/-- The cost for one player's away game equipment in dollars -/
def away_cost : ℕ := sock_cost + tshirt_cost

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := home_cost + away_cost

/-- The total cost for equipping all players in dollars -/
def total_cost : ℕ := 3100

/-- The number of players in the Association -/
def num_players : ℕ := 72

theorem greenfield_basketball_association :
  total_cost = num_players * player_cost := by
  sorry

end greenfield_basketball_association_l3698_369815


namespace solution_properties_l3698_369862

def is_valid_solution (x y z : ℕ+) : Prop :=
  x.val + y.val + z.val = 2013

def count_solutions : ℕ := sorry

def count_solutions_x_eq_y : ℕ := sorry

def max_product_solution : ℕ+ × ℕ+ × ℕ+ := sorry

theorem solution_properties :
  (count_solutions = 2023066) ∧
  (count_solutions_x_eq_y = 1006) ∧
  (max_product_solution = (⟨671, sorry⟩, ⟨671, sorry⟩, ⟨671, sorry⟩)) :=
by sorry

end solution_properties_l3698_369862


namespace age_of_fifteenth_student_l3698_369877

theorem age_of_fifteenth_student
  (total_students : ℕ)
  (average_age : ℝ)
  (group1_count : ℕ)
  (group1_average : ℝ)
  (group2_count : ℕ)
  (group2_average : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_count = 4)
  (h4 : group1_average = 14)
  (h5 : group2_count = 10)
  (h6 : group2_average = 16)
  (h7 : group1_count + group2_count + 1 = total_students) :
  (total_students : ℝ) * average_age - 
  ((group1_count : ℝ) * group1_average + (group2_count : ℝ) * group2_average) = 9 :=
by sorry

end age_of_fifteenth_student_l3698_369877


namespace estimate_grade_a_in_population_l3698_369845

def sample_data : List ℕ := [11, 10, 6, 15, 9, 16, 13, 12, 0, 8,
                             2, 8, 10, 17, 6, 13, 7, 5, 7, 3,
                             12, 10, 7, 11, 3, 6, 8, 14, 15, 12]

def is_grade_a (m : ℕ) : Bool := m ≥ 10

def count_grade_a (data : List ℕ) : ℕ :=
  data.filter is_grade_a |>.length

def sample_size : ℕ := 30

def total_population : ℕ := 1000

theorem estimate_grade_a_in_population :
  (count_grade_a sample_data : ℚ) / sample_size * total_population = 500 := by
  sorry

end estimate_grade_a_in_population_l3698_369845


namespace alex_amount_l3698_369830

def total : ℚ := 972.45
def sam : ℚ := 325.67
def erica : ℚ := 214.29

theorem alex_amount : total - (sam + erica) = 432.49 := by
  sorry

end alex_amount_l3698_369830


namespace function_property_implies_zero_l3698_369857

open Set
open Function

theorem function_property_implies_zero (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Ioo a b, f x + f (-x) = 0) : f (a + b) = 0 := by
  sorry

end function_property_implies_zero_l3698_369857


namespace field_trip_difference_proof_l3698_369896

/-- Calculates the difference in number of people traveling by bus versus van on a field trip. -/
def field_trip_difference (num_vans : Real) (num_buses : Real) (people_per_van : Real) (people_per_bus : Real) : Real :=
  num_buses * people_per_bus - num_vans * people_per_van

/-- Proves that the difference in number of people traveling by bus versus van is 108.0 for the given conditions. -/
theorem field_trip_difference_proof :
  field_trip_difference 6.0 8.0 6.0 18.0 = 108.0 := by
  sorry

end field_trip_difference_proof_l3698_369896


namespace cherry_tart_fraction_l3698_369886

theorem cherry_tart_fraction (total : ℝ) (blueberry : ℝ) (peach : ℝ) 
  (h1 : total = 0.91)
  (h2 : blueberry = 0.75)
  (h3 : peach = 0.08)
  (h4 : ∃ cherry : ℝ, cherry + blueberry + peach = total) :
  ∃ cherry : ℝ, cherry = 0.08 ∧ cherry + blueberry + peach = total := by
sorry

end cherry_tart_fraction_l3698_369886


namespace value_of_b_l3698_369898

theorem value_of_b (p q r : ℝ) (b : ℝ) 
  (h1 : p - q = 2) 
  (h2 : p - r = 1) 
  (h3 : b = (r - q) * ((p - q)^2 + (p - q)*(p - r) + (p - r)^2)) :
  b = 7 := by
  sorry

end value_of_b_l3698_369898


namespace leonardo_sleep_fraction_l3698_369831

-- Define the number of minutes in an hour
def minutes_in_hour : ℕ := 60

-- Define Leonardo's sleep duration in minutes
def leonardo_sleep_minutes : ℕ := 12

-- Theorem to prove
theorem leonardo_sleep_fraction :
  (leonardo_sleep_minutes : ℚ) / minutes_in_hour = 1 / 5 := by
  sorry

end leonardo_sleep_fraction_l3698_369831


namespace simplify_expressions_l3698_369801

theorem simplify_expressions :
  ((-4 : ℝ)^2023 * (-0.25)^2024 = -0.25) ∧
  (23 * (-4/11 : ℝ) + (-5/11) * 23 - 23 * (2/11) = -23) := by
  sorry

end simplify_expressions_l3698_369801


namespace ivan_pension_sufficient_for_ticket_l3698_369811

theorem ivan_pension_sufficient_for_ticket : 
  (149^6 - 199^3) / (149^4 + 199^2 + 199 * 149^2) > 22000 := by
  sorry

end ivan_pension_sufficient_for_ticket_l3698_369811


namespace hundred_with_five_threes_l3698_369833

-- Define a custom type for our arithmetic expressions
inductive Expr
  | const : ℕ → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

-- Function to count the number of 3's in an expression
def countThrees : Expr → ℕ
  | Expr.const n => if n = 3 then 1 else 0
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

-- Function to evaluate an expression
def evaluate : Expr → ℚ
  | Expr.const n => n
  | Expr.add e1 e2 => evaluate e1 + evaluate e2
  | Expr.sub e1 e2 => evaluate e1 - evaluate e2
  | Expr.mul e1 e2 => evaluate e1 * evaluate e2
  | Expr.div e1 e2 => evaluate e1 / evaluate e2

-- Theorem statement
theorem hundred_with_five_threes : 
  ∃ e : Expr, countThrees e = 5 ∧ evaluate e = 100 := by
  sorry

end hundred_with_five_threes_l3698_369833


namespace intersection_equals_A_intersection_is_empty_l3698_369850

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for the first question
theorem intersection_equals_A (m : ℝ) :
  A ∩ B m = A ↔ m ≤ -2 :=
sorry

-- Theorem for the second question
theorem intersection_is_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ 0 ≤ m :=
sorry

end intersection_equals_A_intersection_is_empty_l3698_369850


namespace number_times_five_equals_hundred_l3698_369820

theorem number_times_five_equals_hundred (x : ℝ) : 5 * x = 100 → x = 20 := by
  sorry

end number_times_five_equals_hundred_l3698_369820


namespace calculation_difference_is_zero_l3698_369807

def salesTaxRate : ℝ := 0.08
def originalPrice : ℝ := 120.00
def mainDiscount : ℝ := 0.25
def additionalDiscount : ℝ := 0.10
def numberOfSweaters : ℕ := 4

def amyCalculation : ℝ :=
  numberOfSweaters * (originalPrice * (1 + salesTaxRate) * (1 - mainDiscount) * (1 - additionalDiscount))

def bobCalculation : ℝ :=
  numberOfSweaters * (originalPrice * (1 - mainDiscount) * (1 - additionalDiscount) * (1 + salesTaxRate))

theorem calculation_difference_is_zero :
  amyCalculation = bobCalculation :=
by sorry

end calculation_difference_is_zero_l3698_369807


namespace complex_number_quadrant_l3698_369887

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l3698_369887


namespace function_properties_l3698_369885

/-- A function type that represents the relationship between x and y --/
def Function := ℝ → ℝ

/-- The given values in the table --/
structure TableValues where
  y_neg5 : ℝ
  y_neg2 : ℝ
  y_2 : ℝ
  y_5 : ℝ

/-- Proposition: If y is an inverse proportion function of x, then 2m + 5n = 0 --/
def inverse_proportion_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ k : ℝ, ∀ x : ℝ, f x * x = k) →
  2 * tv.y_neg2 + 5 * tv.y_5 = 0

/-- Proposition: If y is a linear function of x, then n - m = 7 --/
def linear_function_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) →
  tv.y_5 - tv.y_neg2 = 7

/-- Proposition: If y is a quadratic function of x and the graph opens downwards, 
    then m > n is not necessarily true --/
def quadratic_function_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ a b c : ℝ, a < 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c) →
  ¬(tv.y_neg2 > tv.y_5)

/-- The main theorem that combines all three propositions --/
theorem function_properties (f : Function) (tv : TableValues) : 
  inverse_proportion_prop f tv ∧ 
  linear_function_prop f tv ∧ 
  quadratic_function_prop f tv := by sorry

end function_properties_l3698_369885


namespace circle_symmetry_l3698_369859

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-2)^2 + (y+3)^2 = 2

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧
    (x - x₀ = x₀ - 2) ∧ (y - y₀ = y₀ + 3) ∧
    symmetry_line ((x + x₀) / 2) ((y + y₀) / 2)) →
  symmetric_circle x y :=
sorry

end circle_symmetry_l3698_369859


namespace power_of_three_mod_five_l3698_369855

theorem power_of_three_mod_five : 3^2040 % 5 = 1 := by
  sorry

end power_of_three_mod_five_l3698_369855


namespace binomial_coefficient_divisibility_l3698_369802

theorem binomial_coefficient_divisibility (p k : ℕ) : 
  Prime p → 1 ≤ k → k ≤ p - 1 → p ∣ Nat.choose p k := by
  sorry

end binomial_coefficient_divisibility_l3698_369802


namespace triangle_problem_l3698_369881

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : 2 * abc.b * Real.cos abc.A = abc.c * Real.cos abc.A + abc.a * Real.cos abc.C)
  (h2 : abc.a = Real.sqrt 7)
  (h3 : abc.b + abc.c = 4) :
  abc.A = π / 3 ∧ abc.b * abc.c = 3 := by
sorry

end triangle_problem_l3698_369881


namespace joy_tape_problem_l3698_369870

/-- The initial amount of tape given field dimensions and leftover tape -/
def initial_tape (width length leftover : ℕ) : ℕ :=
  2 * (width + length) + leftover

/-- Theorem: Given a field 20 feet wide and 60 feet long, with 90 feet of tape left over after wrapping once, the initial amount of tape is 250 feet -/
theorem joy_tape_problem :
  initial_tape 20 60 90 = 250 := by
  sorry

end joy_tape_problem_l3698_369870


namespace prob_not_edge_10x10_l3698_369843

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  perimeter_squares : ℕ

/-- Calculates the probability of a randomly chosen square not touching the outer edge -/
def prob_not_edge (board : Checkerboard) : ℚ :=
  (board.total_squares - board.perimeter_squares : ℚ) / board.total_squares

/-- Theorem: The probability of a randomly chosen square not touching the outer edge
    on a 10x10 checkerboard is 16/25 -/
theorem prob_not_edge_10x10 :
  ∃ (board : Checkerboard),
    board.size = 10 ∧
    board.total_squares = 100 ∧
    board.perimeter_squares = 36 ∧
    prob_not_edge board = 16 / 25 := by
  sorry

end prob_not_edge_10x10_l3698_369843


namespace vector_equality_l3698_369829

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points in the vector space
variable (A B C M O : V)

-- Define vectors as differences between points
def vec (P Q : V) : V := Q - P

-- State the theorem
theorem vector_equality :
  (vec A B + vec M B) + (vec B O + vec B C) + vec O M = vec A C :=
by sorry

end vector_equality_l3698_369829


namespace smallest_common_divisor_l3698_369832

theorem smallest_common_divisor : ∃ (x : ℕ), 
  x - 16 = 136 ∧ 
  (∀ d : ℕ, d > 0 ∧ d ∣ 136 ∧ d ∣ 6 ∧ d ∣ 8 ∧ d ∣ 10 → d ≥ 2) ∧
  2 ∣ 136 ∧ 2 ∣ 6 ∧ 2 ∣ 8 ∧ 2 ∣ 10 :=
by
  sorry

end smallest_common_divisor_l3698_369832


namespace probability_problem_l3698_369889

structure JarContents where
  red : Nat
  white : Nat
  black : Nat

def jarA : JarContents := { red := 5, white := 2, black := 3 }
def jarB : JarContents := { red := 4, white := 3, black := 3 }

def totalBalls (jar : JarContents) : Nat :=
  jar.red + jar.white + jar.black

def P_A1 : Rat := jarA.red / totalBalls jarA
def P_A2 : Rat := jarA.white / totalBalls jarA
def P_A3 : Rat := jarA.black / totalBalls jarA

def P_B_given_A1 : Rat := (jarB.red + 1) / (totalBalls jarB + 1)
def P_B_given_A2 : Rat := jarB.red / (totalBalls jarB + 1)
def P_B_given_A3 : Rat := jarB.red / (totalBalls jarB + 1)

theorem probability_problem :
  (P_B_given_A1 = 5 / 11) ∧
  (P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 9 / 22) ∧
  (P_A1 + P_A2 + P_A3 = 1) :=
by sorry

#check probability_problem

end probability_problem_l3698_369889


namespace raisins_cranberries_fraction_l3698_369837

/-- Represents the quantities (in pounds) of each ingredient in the mixture -/
structure Quantities where
  raisins : ℕ
  almonds : ℕ
  cashews : ℕ
  walnuts : ℕ
  dried_apricots : ℕ
  dried_cranberries : ℕ

/-- Represents the prices (in dollars per pound) of each ingredient -/
structure Prices where
  raisins : ℕ
  almonds : ℕ
  cashews : ℕ
  walnuts : ℕ
  dried_apricots : ℕ
  dried_cranberries : ℕ

/-- Calculates the total cost of the mixture -/
def total_cost (q : Quantities) (p : Prices) : ℕ :=
  q.raisins * p.raisins + q.almonds * p.almonds + q.cashews * p.cashews +
  q.walnuts * p.walnuts + q.dried_apricots * p.dried_apricots + q.dried_cranberries * p.dried_cranberries

/-- Calculates the cost of raisins and dried cranberries combined -/
def raisins_cranberries_cost (q : Quantities) (p : Prices) : ℕ :=
  q.raisins * p.raisins + q.dried_cranberries * p.dried_cranberries

/-- Theorem stating that the fraction of the total cost that is the cost of raisins and dried cranberries is 19/107 -/
theorem raisins_cranberries_fraction (q : Quantities) (p : Prices)
  (h_quantities : q = { raisins := 5, almonds := 4, cashews := 3, walnuts := 2, dried_apricots := 4, dried_cranberries := 3 })
  (h_prices : p = { raisins := 2, almonds := 6, cashews := 8, walnuts := 10, dried_apricots := 5, dried_cranberries := 3 }) :
  (raisins_cranberries_cost q p : ℚ) / (total_cost q p) = 19 / 107 := by
  sorry

end raisins_cranberries_fraction_l3698_369837


namespace incircle_tangent_bisects_altitude_median_l3698_369897

/-- Triangle with incircle -/
structure TriangleWithIncircle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Positivity of sides
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  -- Triangle inequality
  hab : a + b > c
  hbc : b + c > a
  hca : c + a > b
  -- Existence of incircle (implied by above conditions)

/-- Point on a line segment -/
def PointOnSegment (A B T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (1 - t) • A + t • B

/-- Midpoint of a line segment -/
def Midpoint (A B M : ℝ × ℝ) : Prop :=
  M = (A + B) / 2

/-- Foot of altitude from a point to a line -/
def AltitudeFoot (C H : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  H ∈ l ∧ (∀ P ∈ l, ‖C - H‖ ≤ ‖C - P‖)

/-- Tangent point of incircle -/
def TangentPoint (T : ℝ × ℝ) (circle : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop :=
  T ∈ circle ∧ T ∈ l ∧ (∀ P ∈ circle ∩ l, P = T)

theorem incircle_tangent_bisects_altitude_median 
  (triangle : TriangleWithIncircle) 
  (A B C T H M : ℝ × ℝ) 
  (l : Set (ℝ × ℝ)) 
  (circle : Set (ℝ × ℝ)) :
  (PointOnSegment A B T ∧ 
   Midpoint A B M ∧ 
   AltitudeFoot C H l ∧
   TangentPoint T circle l) →
  (T = (H + M) / 2 ↔ triangle.c = (triangle.a + triangle.b) / 2) :=
sorry

end incircle_tangent_bisects_altitude_median_l3698_369897


namespace fibonacci_closed_form_l3698_369869

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_closed_form (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) (h3 : a > b) :
  ∀ n : ℕ, fibonacci n = (a^(n+1) - b^(n+1)) / Real.sqrt 5 := by
  sorry

end fibonacci_closed_form_l3698_369869


namespace sum_first_ten_enhanced_nice_l3698_369810

def is_prime (n : ℕ) : Prop := sorry

def proper_divisors (n : ℕ) : Set ℕ := sorry

def product_of_set (s : Set ℕ) : ℕ := sorry

def prime_factors (n : ℕ) : List ℕ := sorry

def is_enhanced_nice (n : ℕ) : Prop :=
  (n > 1) ∧
  ((product_of_set (proper_divisors n) = n) ∨
   (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p * q) ∨
   (∃ p : ℕ, is_prime p ∧ n = p^3))

def first_ten_enhanced_nice_under_100 : List ℕ :=
  [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

theorem sum_first_ten_enhanced_nice :
  (List.sum first_ten_enhanced_nice_under_100 = 182) ∧
  (∀ n ∈ first_ten_enhanced_nice_under_100, is_enhanced_nice n) ∧
  (∀ n < 100, is_enhanced_nice n → n ∈ first_ten_enhanced_nice_under_100) :=
sorry

end sum_first_ten_enhanced_nice_l3698_369810


namespace order_of_abc_l3698_369873

theorem order_of_abc : ∀ (a b c : ℝ), 
  a = Real.exp 0.25 → 
  b = 1 → 
  c = -4 * Real.log 0.75 → 
  b < c ∧ c < a := by sorry

end order_of_abc_l3698_369873


namespace tangent_line_to_ln_curve_l3698_369824

theorem tangent_line_to_ln_curve (b : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1/2 * x + b = Real.log x) ∧ 
  (∀ y : ℝ, y > 0 → 1/2 * y + b ≥ Real.log y)) → 
  b = Real.log 2 - 1 := by
sorry

end tangent_line_to_ln_curve_l3698_369824


namespace train_length_calculation_l3698_369852

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time taken for the train to pass the jogger. -/
def train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed - jogger_speed) * passing_time - initial_distance

/-- Theorem stating that under the given conditions, the length of the train is 120 meters. -/
theorem train_length_calculation :
  let jogger_speed : ℝ := 9 * (1000 / 3600)  -- 9 kmph in m/s
  let train_speed : ℝ := 45 * (1000 / 3600)  -- 45 kmph in m/s
  let initial_distance : ℝ := 240  -- meters
  let passing_time : ℝ := 36  -- seconds
  train_length jogger_speed train_speed initial_distance passing_time = 120 := by
  sorry


end train_length_calculation_l3698_369852


namespace omar_coffee_cup_size_l3698_369864

/-- Represents the size of Omar's coffee cup in ounces -/
def coffee_cup_size : ℝ := 6

theorem omar_coffee_cup_size :
  ∀ (remaining_after_work : ℝ) (remaining_after_office : ℝ),
  remaining_after_work = coffee_cup_size - (1/4 * coffee_cup_size + 1/2 * coffee_cup_size) →
  remaining_after_office = remaining_after_work - 1 →
  remaining_after_office = 2 →
  coffee_cup_size = 6 := by
sorry

end omar_coffee_cup_size_l3698_369864


namespace max_value_of_f_l3698_369851

-- Define the function f
def f (x : ℝ) : ℝ := x * (6 - 2*x)^2

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 3 ∧ 
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f c) ∧
  f c = 16 :=
sorry

end max_value_of_f_l3698_369851


namespace part_one_part_two_combined_theorem_l3698_369821

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a m : ℝ) :
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) →
  a = 2 ∧ m = 3 := by sorry

-- Part II
theorem part_two (t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) := by sorry

-- Combined theorem
theorem combined_theorem (a m t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) →
  (a = 2 ∧ m = 3) ∧
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) := by sorry

end part_one_part_two_combined_theorem_l3698_369821


namespace hockey_players_count_l3698_369899

theorem hockey_players_count (total_players cricket_players football_players softball_players : ℕ) 
  (h1 : total_players = 77)
  (h2 : cricket_players = 22)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  total_players - (cricket_players + football_players + softball_players) = 15 := by
sorry

end hockey_players_count_l3698_369899


namespace original_equals_scientific_l3698_369835

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def original_number : ℕ := 12910000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation :=
  { coefficient := 1.291
    exponent := 7
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end original_equals_scientific_l3698_369835
