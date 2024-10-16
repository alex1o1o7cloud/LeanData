import Mathlib

namespace NUMINAMATH_CALUDE_negative_64_to_two_thirds_power_l542_54205

theorem negative_64_to_two_thirds_power (x : ℝ) : x = (-64)^(2/3) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_two_thirds_power_l542_54205


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l542_54209

theorem chord_length_concentric_circles (R r : ℝ) (h : R^2 - r^2 = 15) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ c^2 / 4 + r^2 = R^2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l542_54209


namespace NUMINAMATH_CALUDE_no_valid_operation_l542_54247

-- Define the set of basic arithmetic operations
inductive BasicOperation
  | Add
  | Subtract
  | Multiply
  | Divide

-- Define a function to apply a basic operation
def applyOperation (op : BasicOperation) (a b : ℤ) : ℤ :=
  match op with
  | BasicOperation.Add => a + b
  | BasicOperation.Subtract => a - b
  | BasicOperation.Multiply => a * b
  | BasicOperation.Divide => a / b

-- Theorem statement
theorem no_valid_operation :
  ¬ ∃ (op : BasicOperation), (applyOperation op 8 2) + 5 - (3 - 2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_no_valid_operation_l542_54247


namespace NUMINAMATH_CALUDE_existence_of_six_numbers_l542_54223

theorem existence_of_six_numbers : ∃ (a b c d e f : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  (a + b + c + d + e + f : ℚ) / ((1 : ℚ)/a + 1/b + 1/c + 1/d + 1/e + 1/f) = 2012 :=
sorry

end NUMINAMATH_CALUDE_existence_of_six_numbers_l542_54223


namespace NUMINAMATH_CALUDE_olivias_score_l542_54238

theorem olivias_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) :
  n = 20 →
  avg_without = 85 →
  avg_with = 86 →
  (n * avg_without + x) / (n + 1) = avg_with →
  x = 106 :=
by sorry

end NUMINAMATH_CALUDE_olivias_score_l542_54238


namespace NUMINAMATH_CALUDE_fraction_of_x_l542_54265

theorem fraction_of_x (x y : ℝ) (k : ℝ) 
  (h1 : 5 * x = 3 * y) 
  (h2 : x * y ≠ 0) 
  (h3 : k * x / (1/6 * y) = 0.7200000000000001) : 
  k = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_x_l542_54265


namespace NUMINAMATH_CALUDE_smallest_surface_area_is_cube_l542_54290

-- Define a rectangular parallelepiped
structure Parallelepiped where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z

-- Define the volume of a parallelepiped
def volume (p : Parallelepiped) : ℝ := p.x * p.y * p.z

-- Define the surface area of a parallelepiped
def surfaceArea (p : Parallelepiped) : ℝ := 2 * (p.x * p.y + p.x * p.z + p.y * p.z)

-- State the theorem
theorem smallest_surface_area_is_cube (V : ℝ) (hV : 0 < V) :
  ∃ (p : Parallelepiped), volume p = V ∧
    ∀ (q : Parallelepiped), volume q = V → surfaceArea p ≤ surfaceArea q ∧
      (surfaceArea p = surfaceArea q → p.x = p.y ∧ p.y = p.z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_surface_area_is_cube_l542_54290


namespace NUMINAMATH_CALUDE_sin_870_degrees_l542_54271

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l542_54271


namespace NUMINAMATH_CALUDE_balcony_orchestra_difference_is_40_l542_54278

/-- Represents the ticket sales for a theater performance --/
structure TheaterSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ

/-- Calculates the difference between balcony and orchestra ticket sales --/
def balcony_orchestra_difference (sales : TheaterSales) : ℕ :=
  sales.total_tickets - 2 * (sales.total_revenue - sales.balcony_price * sales.total_tickets) / (sales.orchestra_price - sales.balcony_price)

/-- Theorem stating the difference between balcony and orchestra ticket sales --/
theorem balcony_orchestra_difference_is_40 (sales : TheaterSales) 
  (h1 : sales.orchestra_price = 12)
  (h2 : sales.balcony_price = 8)
  (h3 : sales.total_tickets = 340)
  (h4 : sales.total_revenue = 3320) :
  balcony_orchestra_difference sales = 40 := by
  sorry

#eval balcony_orchestra_difference ⟨12, 8, 340, 3320⟩

end NUMINAMATH_CALUDE_balcony_orchestra_difference_is_40_l542_54278


namespace NUMINAMATH_CALUDE_expression_evaluation_l542_54272

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  ((a + 3) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 10) / (c + 7)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l542_54272


namespace NUMINAMATH_CALUDE_expression_equals_one_l542_54263

theorem expression_equals_one (x : ℝ) 
  (h1 : x^3 + 2*x + 1 ≠ 0) 
  (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ((((x+2)^2 * (x^2-x+2)^2) / (x^3+2*x+1)^2)^3 * 
   (((x-2)^2 * (x^2+x+2)^2) / (x^3-2*x-1)^2)^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l542_54263


namespace NUMINAMATH_CALUDE_prob_sum_three_l542_54253

/-- Represents a ball with a number label -/
inductive Ball : Type
| one : Ball
| two : Ball

/-- Represents the result of two draws -/
structure TwoDraws where
  first : Ball
  second : Ball

/-- The set of all possible outcomes from two draws -/
def allOutcomes : Finset TwoDraws :=
  sorry

/-- The set of favorable outcomes (sum of drawn numbers is 3) -/
def favorableOutcomes : Finset TwoDraws :=
  sorry

/-- The probability of an event is the number of favorable outcomes
    divided by the total number of outcomes -/
def probability (event : Finset TwoDraws) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

/-- The main theorem: the probability of drawing two balls with sum 3 is 1/2 -/
theorem prob_sum_three : probability favorableOutcomes = 1/2 :=
  sorry

end NUMINAMATH_CALUDE_prob_sum_three_l542_54253


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l542_54281

/-- Given two vectors in 2D space satisfying certain conditions, prove that the magnitude of one vector is 3. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  (∃ θ : ℝ, θ = Real.pi / 3 ∧ a.fst * b.fst + a.snd * b.snd = Real.cos θ * ‖a‖ * ‖b‖) →  -- angle between a and b is 60°
  ‖a‖ = 1 →  -- |a| = 1
  ‖2 • a - b‖ = Real.sqrt 7 →  -- |2a - b| = √7
  ‖b‖ = 3 := by
sorry


end NUMINAMATH_CALUDE_vector_magnitude_problem_l542_54281


namespace NUMINAMATH_CALUDE_least_value_of_x_l542_54214

theorem least_value_of_x (x p q : ℕ) : 
  x > 0 →
  Nat.Prime p →
  Nat.Prime q →
  p < q →
  x / (12 * p * q) = 2 →
  2 * p - q = 3 →
  ∀ y, y > 0 ∧ 
       ∃ p' q', Nat.Prime p' ∧ Nat.Prime q' ∧ p' < q' ∧
                y / (12 * p' * q') = 2 ∧
                2 * p' - q' = 3 →
       x ≤ y →
  x = 840 := by
sorry


end NUMINAMATH_CALUDE_least_value_of_x_l542_54214


namespace NUMINAMATH_CALUDE_sine_function_omega_values_l542_54261

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is symmetric about a point (a, b) if f(a + x) = f(a - x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

/-- A function f is monotonic on an interval [a, b] if it is either increasing or decreasing on that interval -/
def IsMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem sine_function_omega_values 
  (ω φ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : f = fun x ↦ Real.sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : 0 ≤ φ ∧ φ ≤ π)
  (h4 : IsEven f)
  (h5 : IsSymmetricAbout f (3 * π / 4) 0)
  (h6 : IsMonotonicOn f 0 (π / 2)) :
  ω = 2/3 ∨ ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_omega_values_l542_54261


namespace NUMINAMATH_CALUDE_weeks_to_save_shirt_l542_54203

/-- Calculates the minimum number of whole weeks needed to save for a shirt -/
def min_weeks_to_save (shirt_cost : ℚ) (initial_savings : ℚ) (weekly_savings : ℚ) : ℕ :=
  Nat.ceil ((shirt_cost - initial_savings) / weekly_savings)

/-- Theorem stating that 34 weeks are needed to save for the shirt under given conditions -/
theorem weeks_to_save_shirt : min_weeks_to_save 15 5 0.3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_save_shirt_l542_54203


namespace NUMINAMATH_CALUDE_sin_double_angle_l542_54206

theorem sin_double_angle (α : ℝ) (h : Real.sin (α - π/4) = 3/5) : 
  Real.sin (2 * α) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_l542_54206


namespace NUMINAMATH_CALUDE_max_even_differences_l542_54225

/-- A permutation of numbers from 1 to 25 -/
def Arrangement := Fin 25 → Fin 25

/-- The sequence 1, 2, 3, ..., 25 -/
def OriginalSequence : Fin 25 → ℕ := fun i => i.val + 1

/-- The difference function, always subtracting the smaller from the larger -/
def Difference (arr : Arrangement) (i : Fin 25) : ℕ :=
  max (OriginalSequence i) (arr i).val + 1 - min (OriginalSequence i) (arr i).val + 1

/-- Predicate to check if a number is even -/
def IsEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem max_even_differences :
  ∃ (arr : Arrangement), ∀ (i : Fin 25), IsEven (Difference arr i) :=
sorry

end NUMINAMATH_CALUDE_max_even_differences_l542_54225


namespace NUMINAMATH_CALUDE_fourth_column_is_quadratic_l542_54280

/-- A quadruple of real numbers is quadratic if it satisfies the quadratic condition. -/
def is_quadratic (y : Fin 4 → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ n : Fin 4, y n = a * (n.val + 1)^2 + b * (n.val + 1) + c

/-- A 4×4 grid of real numbers. -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- All rows of the grid are quadratic. -/
def all_rows_quadratic (g : Grid) : Prop :=
  ∀ i : Fin 4, is_quadratic (λ j => g i j)

/-- The first three columns of the grid are quadratic. -/
def first_three_columns_quadratic (g : Grid) : Prop :=
  ∀ j : Fin 3, is_quadratic (λ i => g i j)

/-- The fourth column of the grid is quadratic. -/
def fourth_column_quadratic (g : Grid) : Prop :=
  is_quadratic (λ i => g i 3)

/-- 
If all rows and the first three columns of a 4×4 grid are quadratic,
then the fourth column is also quadratic.
-/
theorem fourth_column_is_quadratic (g : Grid)
  (h_rows : all_rows_quadratic g)
  (h_cols : first_three_columns_quadratic g) :
  fourth_column_quadratic g :=
sorry

end NUMINAMATH_CALUDE_fourth_column_is_quadratic_l542_54280


namespace NUMINAMATH_CALUDE_dining_bill_problem_l542_54266

theorem dining_bill_problem (bill : ℝ) (tip_percentage : ℝ) (share : ℝ) : 
  bill = 139 → 
  tip_percentage = 0.1 → 
  share = 30.58 → 
  (bill + bill * tip_percentage) / share = 5 := by
sorry

end NUMINAMATH_CALUDE_dining_bill_problem_l542_54266


namespace NUMINAMATH_CALUDE_kishore_savings_percentage_l542_54229

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 6100
def savings : ℕ := 2400

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_kishore_savings_percentage_l542_54229


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l542_54222

/-- Farmer's field ploughing problem -/
theorem farmer_ploughing_problem 
  (initial_daily_area : ℝ) 
  (productivity_increase : ℝ) 
  (days_ahead : ℕ) 
  (total_field_area : ℝ) 
  (h1 : initial_daily_area = 120)
  (h2 : productivity_increase = 0.25)
  (h3 : days_ahead = 2)
  (h4 : total_field_area = 1440) :
  ∃ (planned_days : ℕ) (actual_days : ℕ),
    planned_days = 10 ∧ 
    actual_days = planned_days - days_ahead ∧
    actual_days * initial_daily_area + 
      (planned_days - actual_days) * (initial_daily_area * (1 + productivity_increase)) = 
    total_field_area :=
by sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l542_54222


namespace NUMINAMATH_CALUDE_bread_pieces_theorem_l542_54282

/-- The number of pieces a slice of bread becomes when torn in half twice -/
def pieces_per_slice : ℕ := 4

/-- The number of slices of bread used -/
def num_slices : ℕ := 2

/-- The total number of bread pieces after tearing -/
def total_pieces : ℕ := num_slices * pieces_per_slice

theorem bread_pieces_theorem : total_pieces = 8 := by
  sorry

end NUMINAMATH_CALUDE_bread_pieces_theorem_l542_54282


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l542_54287

def pirate_share (n : ℕ) (k : ℕ) (remaining : ℚ) : ℚ :=
  (k : ℚ) / (n : ℚ) * remaining

def remaining_coins (n : ℕ) (k : ℕ) (initial : ℚ) : ℚ :=
  if k = 0 then initial
  else
    (1 - (k : ℚ) / (n : ℚ)) * remaining_coins n (k - 1) initial

def is_valid_distribution (n : ℕ) (initial : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → ∃ (m : ℕ), pirate_share n k (remaining_coins n (k - 1) initial) = m

theorem pirate_treasure_division (n : ℕ) (h : n = 15) :
  ∃ (initial : ℕ),
    (∀ smaller : ℕ, smaller < initial → ¬is_valid_distribution n smaller) ∧
    is_valid_distribution n initial ∧
    pirate_share n n (remaining_coins n (n - 1) initial) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_division_l542_54287


namespace NUMINAMATH_CALUDE_consecutive_sum_transformation_l542_54230

theorem consecutive_sum_transformation (S : ℤ) : 
  ∃ (a : ℤ), 
    (a + (a + 1) = S) → 
    (3 * (a + 5) + 3 * ((a + 1) + 5) = 3 * S + 30) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_transformation_l542_54230


namespace NUMINAMATH_CALUDE_pizza_toppings_l542_54257

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 14)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  (pepperoni_slices + mushroom_slices - total_slices : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l542_54257


namespace NUMINAMATH_CALUDE_mMobileCheaperByEleven_l542_54245

-- Define the cost structure for T-Mobile
def tMobileBaseCost : ℕ := 50
def tMobileAdditionalLineCost : ℕ := 16

-- Define the cost structure for M-Mobile
def mMobileBaseCost : ℕ := 45
def mMobileAdditionalLineCost : ℕ := 14

-- Define the number of lines needed
def totalLines : ℕ := 5

-- Define the function to calculate the total cost for a given plan
def calculateTotalCost (baseCost additionalLineCost : ℕ) : ℕ :=
  baseCost + (totalLines - 2) * additionalLineCost

-- Theorem statement
theorem mMobileCheaperByEleven :
  calculateTotalCost tMobileBaseCost tMobileAdditionalLineCost -
  calculateTotalCost mMobileBaseCost mMobileAdditionalLineCost = 11 := by
  sorry

end NUMINAMATH_CALUDE_mMobileCheaperByEleven_l542_54245


namespace NUMINAMATH_CALUDE_unique_records_count_l542_54227

/-- The number of records in either Samantha's or Lily's collection, but not both -/
def unique_records (samantha_total : ℕ) (shared : ℕ) (lily_unique : ℕ) : ℕ :=
  (samantha_total - shared) + lily_unique

/-- Proof that the number of unique records is 18 -/
theorem unique_records_count :
  unique_records 24 15 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_records_count_l542_54227


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l542_54297

/-- The number of sides of a regular polygon where the difference between 
    the number of diagonals and the number of sides is 7. -/
def polygon_sides : ℕ := 7

/-- The number of diagonals in a polygon with n sides. -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_polygon_sides :
  ∃ (n : ℕ), n > 0 ∧ num_diagonals n - n = 7 → n = polygon_sides :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l542_54297


namespace NUMINAMATH_CALUDE_line_through_point_representation_l542_54291

/-- A line in a 2D plane --/
structure Line where
  slope : Option ℝ
  yIntercept : ℝ

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point --/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  match l.slope with
  | some k => p.y = k * p.x + l.yIntercept
  | none => p.x = l.yIntercept

/-- The statement to be proven false --/
theorem line_through_point_representation (b : ℝ) :
  ∃ (k : ℝ), ∀ (l : Line), l.passesThrough ⟨0, b⟩ → 
  ∃ (k' : ℝ), l.slope = some k' ∧ l.yIntercept = b :=
sorry

end NUMINAMATH_CALUDE_line_through_point_representation_l542_54291


namespace NUMINAMATH_CALUDE_constant_term_value_l542_54251

/-- The constant term in the expansion of (3x + 2/x)^8 -/
def constant_term : ℕ :=
  let binomial_coeff := (8 : ℕ).choose 4
  let x_power_term := 3^4 * 2^4
  binomial_coeff * x_power_term

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_value : constant_term = 90720 := by
  sorry

#eval constant_term -- This will evaluate the constant term

end NUMINAMATH_CALUDE_constant_term_value_l542_54251


namespace NUMINAMATH_CALUDE_pyramid_paint_theorem_l542_54284

/-- Represents a pyramid-like structure with a given number of floors -/
structure PyramidStructure where
  floors : Nat

/-- Calculates the number of painted faces on one side of the structure -/
def sideFaces (p : PyramidStructure) : Nat :=
  (p.floors * (p.floors + 1)) / 2

/-- Calculates the total number of red-painted faces -/
def redFaces (p : PyramidStructure) : Nat :=
  4 * sideFaces p

/-- Calculates the total number of blue-painted faces -/
def blueFaces (p : PyramidStructure) : Nat :=
  sideFaces p

/-- Calculates the total number of painted faces -/
def totalPaintedFaces (p : PyramidStructure) : Nat :=
  redFaces p + blueFaces p

/-- Theorem stating the ratio of red to blue painted faces and the total number of painted faces -/
theorem pyramid_paint_theorem (p : PyramidStructure) (h : p.floors = 25) :
  redFaces p / blueFaces p = 4 ∧ totalPaintedFaces p = 1625 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_paint_theorem_l542_54284


namespace NUMINAMATH_CALUDE_all_digits_satisfy_inequality_l542_54254

theorem all_digits_satisfy_inequality :
  ∀ A : ℕ, A ≤ 9 → 27 * 10 * A + 2708 - 1203 > 1022 := by
  sorry

end NUMINAMATH_CALUDE_all_digits_satisfy_inequality_l542_54254


namespace NUMINAMATH_CALUDE_fruit_basket_count_l542_54292

/-- Represents the number of apples available -/
def num_apples : ℕ := 6

/-- Represents the number of oranges available -/
def num_oranges : ℕ := 8

/-- Represents the minimum number of apples required in each basket -/
def min_apples : ℕ := 2

/-- Calculates the number of possible fruit baskets -/
def num_fruit_baskets : ℕ :=
  (num_apples - min_apples + 1) * (num_oranges + 1)

/-- Theorem stating the number of possible fruit baskets -/
theorem fruit_basket_count :
  num_fruit_baskets = 45 := by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l542_54292


namespace NUMINAMATH_CALUDE_souvenir_spending_l542_54270

/-- Given the total spending on souvenirs and the difference between
    key chains & bracelets and t-shirts, proves the amount spent on
    key chains and bracelets. -/
theorem souvenir_spending
  (total : ℚ)
  (difference : ℚ)
  (h1 : total = 548)
  (h2 : difference = 146) :
  let tshirts := (total - difference) / 2
  let keychains_bracelets := tshirts + difference
  keychains_bracelets = 347 := by
sorry

end NUMINAMATH_CALUDE_souvenir_spending_l542_54270


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l542_54219

/-- The number of diagonals from a single vertex in a decagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: In a decagon, the number of diagonals from a single vertex is 7 -/
theorem decagon_diagonals_from_vertex : 
  diagonals_from_vertex 10 = 7 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l542_54219


namespace NUMINAMATH_CALUDE_complement_of_union_l542_54224

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {1, 3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l542_54224


namespace NUMINAMATH_CALUDE_original_balance_l542_54246

/-- Proves that if a balance incurs a 2% finance charge and results in a total of $153, then the original balance was $150. -/
theorem original_balance (total : ℝ) (finance_charge_rate : ℝ) (h1 : finance_charge_rate = 0.02) (h2 : total = 153) :
  ∃ (original : ℝ), original * (1 + finance_charge_rate) = total ∧ original = 150 :=
by sorry

end NUMINAMATH_CALUDE_original_balance_l542_54246


namespace NUMINAMATH_CALUDE_slope_angle_range_l542_54237

/-- The range of slope angles for a line passing through (2,1) and (1,m²) where m ∈ ℝ -/
theorem slope_angle_range :
  ∀ m : ℝ, ∃ θ : ℝ, 
    (θ ∈ Set.Icc 0 (π/2) ∪ Set.Ioo (π/2) π) ∧ 
    (θ = Real.arctan ((m^2 - 1) / (2 - 1)) ∨ θ = Real.arctan ((m^2 - 1) / (2 - 1)) + π) :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l542_54237


namespace NUMINAMATH_CALUDE_divisible_by_six_and_inductive_step_l542_54264

theorem divisible_by_six_and_inductive_step (n : ℕ) :
  6 ∣ (n * (n + 1) * (2 * n + 1)) ∧
  (∀ k : ℕ, (k + 1) * ((k + 1) + 1) * (2 * (k + 1) + 1) = k * (k + 1) * (2 * k + 1) + 6 * (k + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_six_and_inductive_step_l542_54264


namespace NUMINAMATH_CALUDE_range_of_a_l542_54298

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ |x^2 - 2*x|}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a ≤ 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l542_54298


namespace NUMINAMATH_CALUDE_edmund_computer_savings_l542_54258

/-- Represents the savings problem for Edmund's computer purchase. -/
def computer_savings_problem (computer_cost starting_balance monthly_gift : ℚ)
  (part_time_daily_wage part_time_days_per_week : ℚ)
  (extra_chore_wage chores_per_day regular_chores_per_week : ℚ)
  (car_wash_wage car_washes_per_week : ℚ)
  (lawn_mowing_wage lawns_per_week : ℚ) : Prop :=
  let weekly_earnings := 
    part_time_daily_wage * part_time_days_per_week +
    extra_chore_wage * (chores_per_day * 7 - regular_chores_per_week) +
    car_wash_wage * car_washes_per_week +
    lawn_mowing_wage * lawns_per_week
  let weekly_savings := weekly_earnings + monthly_gift / 4
  let days_to_save := 
    (↑(Nat.ceil ((computer_cost - starting_balance) / weekly_savings)) * 7 : ℚ)
  days_to_save = 49

/-- Theorem stating that Edmund will save enough for the computer in 49 days. -/
theorem edmund_computer_savings : 
  computer_savings_problem 750 200 50 10 3 2 4 12 3 2 5 1 := by
  sorry


end NUMINAMATH_CALUDE_edmund_computer_savings_l542_54258


namespace NUMINAMATH_CALUDE_orthocenters_collinear_l542_54285

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the concept of an acute-angled triangle
def IsAcuteAngled (t : Triangle) : Prop := sorry

-- Define altitude, internal and external angle bisectors
def Altitude (t : Triangle) (v : Point) : Set Point := sorry
def InternalBisector (t : Triangle) (v : Point) : Set Point := sorry
def ExternalBisector (t : Triangle) (v : Point) : Set Point := sorry

-- Define intersection points
def IntersectionPoints (t : Triangle) : Point × Point × Point × Point × Point × Point := sorry

-- Define orthocenter
def Orthocenter (t : Triangle) : Point := sorry

-- Main theorem
theorem orthocenters_collinear 
  (t : Triangle) 
  (h_acute : IsAcuteAngled t) 
  (p q r s u v : Point) 
  (h_intersections : IntersectionPoints t = (p, q, r, s, u, v)) :
  ∃ (l : Set Point), 
    Orthocenter ⟨p, q, t.A⟩ ∈ l ∧ 
    Orthocenter ⟨r, s, t.B⟩ ∈ l ∧ 
    Orthocenter ⟨u, v, t.C⟩ ∈ l ∧ 
    t.A ∈ l ∧ 
    ∀ (x y : Point), x ∈ l → y ∈ l → ∃ (k : ℝ), y.x - x.x = k * (y.y - x.y) := by
  sorry

end NUMINAMATH_CALUDE_orthocenters_collinear_l542_54285


namespace NUMINAMATH_CALUDE_room_length_calculation_l542_54286

/-- Given a rectangular room with the following properties:
  * The width is 4 meters
  * The cost of paving is 800 Rs per square meter
  * The total cost of paving is 17600 Rs
  Then the length of the room is 5.5 meters -/
theorem room_length_calculation (width : ℝ) (paving_cost_per_sqm : ℝ) (total_paving_cost : ℝ) :
  width = 4 →
  paving_cost_per_sqm = 800 →
  total_paving_cost = 17600 →
  (total_paving_cost / (width * paving_cost_per_sqm)) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l542_54286


namespace NUMINAMATH_CALUDE_no_three_squares_sum_2015_l542_54207

theorem no_three_squares_sum_2015 : ¬ ∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_three_squares_sum_2015_l542_54207


namespace NUMINAMATH_CALUDE_inequality_solution_set_l542_54262

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l542_54262


namespace NUMINAMATH_CALUDE_gecko_eating_pattern_l542_54279

/-- Represents the gecko's eating pattern over three days -/
structure GeckoEating where
  total_crickets : ℕ
  third_day_crickets : ℕ
  second_day_difference : ℕ

/-- Calculates the percentage of crickets eaten on the first day -/
def first_day_percentage (g : GeckoEating) : ℚ :=
  let first_two_days := g.total_crickets - g.third_day_crickets
  let x := (2 * first_two_days + g.second_day_difference) / (2 * g.total_crickets)
  x * 100

/-- Theorem stating that under the given conditions, the gecko eats 30% of crickets on the first day -/
theorem gecko_eating_pattern :
  let g : GeckoEating := {
    total_crickets := 70,
    third_day_crickets := 34,
    second_day_difference := 6
  }
  first_day_percentage g = 30 := by sorry

end NUMINAMATH_CALUDE_gecko_eating_pattern_l542_54279


namespace NUMINAMATH_CALUDE_negative_number_with_abs_two_l542_54296

theorem negative_number_with_abs_two (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_number_with_abs_two_l542_54296


namespace NUMINAMATH_CALUDE_shaded_area_circles_l542_54210

theorem shaded_area_circles (R : ℝ) (h : R = 10) : 
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let shaded_area := large_circle_area - 2 * small_circle_area
  shaded_area = 50 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l542_54210


namespace NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l542_54236

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def first_four_composites : List ℕ := [4, 6, 8, 9]

theorem units_digit_of_first_four_composites_product :
  (first_four_composites.prod % 10 = 8) ∧
  (∀ n ∈ first_four_composites, is_composite n) ∧
  (∀ m, is_composite m → m ≥ 4 → ∃ n ∈ first_four_composites, n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l542_54236


namespace NUMINAMATH_CALUDE_quadratic_function_range_l542_54235

/-- Given a quadratic function y = -x^2 + 2ax + a + 1, if y > a + 1 for all x in (-1, a),
    then -1 < a ≤ -1/2 -/
theorem quadratic_function_range (a : ℝ) :
  (∀ x, -1 < x ∧ x < a → -x^2 + 2*a*x + a + 1 > a + 1) →
  -1 < a ∧ a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l542_54235


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l542_54221

theorem perfect_square_binomial : ∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + 100 = (a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l542_54221


namespace NUMINAMATH_CALUDE_different_course_choices_l542_54250

theorem different_course_choices (n : ℕ) (k : ℕ) : n = 4 → k = 2 →
  (Nat.choose n k)^2 - (Nat.choose n k) = 30 := by
  sorry

end NUMINAMATH_CALUDE_different_course_choices_l542_54250


namespace NUMINAMATH_CALUDE_line_points_k_value_l542_54256

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2*n + 5) →                   -- First point (m, n) satisfies the line equation
  (m + 5 = 2*(n + k) + 5) →         -- Second point (m + 5, n + k) satisfies the line equation
  k = 5/2 := by                     -- Conclusion: k = 2.5
sorry


end NUMINAMATH_CALUDE_line_points_k_value_l542_54256


namespace NUMINAMATH_CALUDE_distance_between_points_l542_54249

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-2, 4)
  let p2 : ℝ × ℝ := (3, -8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l542_54249


namespace NUMINAMATH_CALUDE_remainder_sum_factorials_60_l542_54252

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_sum_factorials_60 (h : ∀ k ≥ 5, 15 ∣ factorial k) :
  sum_factorials 60 % 15 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_factorials_60_l542_54252


namespace NUMINAMATH_CALUDE_order_of_a_l542_54212

theorem order_of_a (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_a_l542_54212


namespace NUMINAMATH_CALUDE_ball_travel_distance_l542_54232

/-- The distance traveled by the center of a ball rolling along a track of semicircular arcs -/
theorem ball_travel_distance (ball_diameter : ℝ) (R₁ R₂ R₃ : ℝ) : 
  ball_diameter = 6 → R₁ = 120 → R₂ = 70 → R₃ = 90 → 
  (R₁ - ball_diameter / 2) * π + (R₂ - ball_diameter / 2) * π + (R₃ - ball_diameter / 2) * π = 271 * π :=
by sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l542_54232


namespace NUMINAMATH_CALUDE_triangle_problem_l542_54231

noncomputable section

def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 - 1

theorem triangle_problem (A B C a b c : ℝ) :
  c = Real.sqrt 3 →
  f C = 0 →
  Real.sin B = 2 * Real.sin A →
  0 < C →
  C < π →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  C = π/3 ∧ a = 1 ∧ b = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l542_54231


namespace NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l542_54299

/-- Given a quadratic equation ax² + bx + c = 0, returns the coefficient of the linear term (b) -/
def linearCoefficient (a b c : ℚ) : ℚ := b

theorem linear_coefficient_of_example_quadratic :
  linearCoefficient 2 3 (-4) = 3 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l542_54299


namespace NUMINAMATH_CALUDE_printer_problem_l542_54248

/-- Calculates the time needed to print a given number of pages with breaks -/
def print_time (pages_per_minute : ℕ) (total_pages : ℕ) (pages_before_break : ℕ) (break_duration : ℕ) : ℕ :=
  let full_segments := total_pages / pages_before_break
  let remaining_pages := total_pages % pages_before_break
  let printing_time := (full_segments * pages_before_break + remaining_pages) / pages_per_minute
  let break_time := full_segments * break_duration
  printing_time + break_time

theorem printer_problem :
  print_time 25 350 150 5 = 24 := by
sorry

end NUMINAMATH_CALUDE_printer_problem_l542_54248


namespace NUMINAMATH_CALUDE_tom_bought_six_oranges_l542_54200

/-- Represents the number of oranges Tom bought -/
def num_oranges : ℕ := 6

/-- Represents the number of apples Tom bought -/
def num_apples : ℕ := 7 - num_oranges

/-- The cost of an orange in cents -/
def orange_cost : ℕ := 90

/-- The cost of an apple in cents -/
def apple_cost : ℕ := 60

/-- The total number of fruits bought -/
def total_fruits : ℕ := 7

/-- The total cost in cents -/
def total_cost : ℕ := orange_cost * num_oranges + apple_cost * num_apples

theorem tom_bought_six_oranges :
  num_oranges + num_apples = total_fruits ∧
  total_cost % 100 = 0 ∧
  num_oranges = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_bought_six_oranges_l542_54200


namespace NUMINAMATH_CALUDE_circle_area_difference_l542_54273

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l542_54273


namespace NUMINAMATH_CALUDE_sphere_plane_intersection_l542_54288

/-- A sphere intersecting a plane creates a circular intersection. -/
theorem sphere_plane_intersection
  (r : ℝ) -- radius of the sphere
  (h : ℝ) -- depth of the intersection
  (w : ℝ) -- radius of the circular intersection
  (hr : r = 16.25)
  (hh : h = 10)
  (hw : w = 15) :
  r^2 = h * (2 * r - h) + w^2 :=
sorry

end NUMINAMATH_CALUDE_sphere_plane_intersection_l542_54288


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l542_54215

theorem sufficient_but_not_necessary :
  (∃ p q : Prop, (p ∨ q = False) → (¬p = True)) ∧
  (∃ p q : Prop, (¬p = True) ∧ ¬(p ∨ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l542_54215


namespace NUMINAMATH_CALUDE_quadratic_roots_and_m_values_l542_54267

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (m - 2) * x + m - 3

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m - 2)^2 - 4 * (m - 3)

-- Define the condition for the roots
def rootCondition (m x₁ x₂ : ℝ) : Prop := 2 * x₁ + x₂ = m + 1

theorem quadratic_roots_and_m_values :
  (∀ m : ℝ, discriminant m ≥ 0) ∧
  (∀ m x₁ x₂ : ℝ, 
    quadratic m x₁ = 0 → 
    quadratic m x₂ = 0 → 
    rootCondition m x₁ x₂ → 
    (m = 0 ∨ m = 4/3)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_m_values_l542_54267


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l542_54201

theorem factorization_of_quadratic (a : ℚ) : 2 * a^2 - 4 * a = 2 * a * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l542_54201


namespace NUMINAMATH_CALUDE_three_monotonic_intervals_l542_54241

/-- A cubic function with a linear term -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

/-- Theorem stating the condition for f to have exactly three monotonic intervals -/
theorem three_monotonic_intervals (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_three_monotonic_intervals_l542_54241


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l542_54274

theorem rahul_deepak_age_ratio :
  ∀ (rahul_age deepak_age : ℕ),
    deepak_age = 27 →
    rahul_age + 6 = 42 →
    (rahul_age : ℚ) / deepak_age = 4 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l542_54274


namespace NUMINAMATH_CALUDE_inequality_proof_l542_54269

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l542_54269


namespace NUMINAMATH_CALUDE_emily_beads_count_l542_54244

/-- The number of beads per necklace -/
def beads_per_necklace : ℕ := 5

/-- The number of necklaces Emily made -/
def necklaces_made : ℕ := 4

/-- The total number of beads Emily used -/
def total_beads : ℕ := beads_per_necklace * necklaces_made

theorem emily_beads_count : total_beads = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l542_54244


namespace NUMINAMATH_CALUDE_cos_sum_eleventh_pi_l542_54289

open Complex

theorem cos_sum_eleventh_pi : 
  cos (π / 11) + cos (3 * π / 11) + cos (7 * π / 11) + cos (9 * π / 11) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_eleventh_pi_l542_54289


namespace NUMINAMATH_CALUDE_workers_wage_increase_l542_54294

theorem workers_wage_increase (original_wage new_wage : ℝ) : 
  (new_wage = original_wage * 1.5) → (new_wage = 51) → (original_wage = 34) := by
  sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l542_54294


namespace NUMINAMATH_CALUDE_currency_conversion_l542_54259

/-- Conversion rates and constants --/
def paise_per_rupee : ℚ := 100
def usd_per_inr : ℚ := 12 / 1000
def eur_per_inr : ℚ := 10 / 1000
def gbp_per_inr : ℚ := 9 / 1000

/-- The value 'a' in paise --/
def a_paise : ℚ := 15000

/-- Theorem stating the correct values of 'a' in different currencies --/
theorem currency_conversion (a : ℚ) 
  (h1 : a * (1/2) / 100 = 75) : 
  a / paise_per_rupee = 150 ∧ 
  a / paise_per_rupee * usd_per_inr = 9/5 ∧ 
  a / paise_per_rupee * eur_per_inr = 3/2 ∧ 
  a / paise_per_rupee * gbp_per_inr = 27/20 := by
  sorry

#check currency_conversion

end NUMINAMATH_CALUDE_currency_conversion_l542_54259


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l542_54233

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) : 
  z.im = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l542_54233


namespace NUMINAMATH_CALUDE_probability_two_red_marbles_l542_54239

/-- The probability of drawing two red marbles without replacement from a jar containing
    2 red marbles, 3 green marbles, and 10 white marbles is 1/105. -/
theorem probability_two_red_marbles :
  let red_marbles : ℕ := 2
  let green_marbles : ℕ := 3
  let white_marbles : ℕ := 10
  let total_marbles : ℕ := red_marbles + green_marbles + white_marbles
  (red_marbles : ℚ) / total_marbles * (red_marbles - 1) / (total_marbles - 1) = 1 / 105 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_marbles_l542_54239


namespace NUMINAMATH_CALUDE_max_a_for_three_integer_solutions_l542_54293

theorem max_a_for_three_integer_solutions : 
  ∃ (a : ℝ), 
    (∀ x : ℤ, (-1/3 : ℝ) * (x : ℝ) > 2/3 - (x : ℝ) ∧ 
               (1/2 : ℝ) * (x : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) →
    (∃! (x₁ x₂ x₃ : ℤ), 
      ((-1/3 : ℝ) * (x₁ : ℝ) > 2/3 - (x₁ : ℝ) ∧ 
       (1/2 : ℝ) * (x₁ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      ((-1/3 : ℝ) * (x₂ : ℝ) > 2/3 - (x₂ : ℝ) ∧ 
       (1/2 : ℝ) * (x₂ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      ((-1/3 : ℝ) * (x₃ : ℝ) > 2/3 - (x₃ : ℝ) ∧ 
       (1/2 : ℝ) * (x₃ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
    a = 5 ∧ ∀ b > a, 
      ¬(∃! (x₁ x₂ x₃ : ℤ), 
        ((-1/3 : ℝ) * (x₁ : ℝ) > 2/3 - (x₁ : ℝ) ∧ 
         (1/2 : ℝ) * (x₁ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        ((-1/3 : ℝ) * (x₂ : ℝ) > 2/3 - (x₂ : ℝ) ∧ 
         (1/2 : ℝ) * (x₂ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        ((-1/3 : ℝ) * (x₃ : ℝ) > 2/3 - (x₃ : ℝ) ∧ 
         (1/2 : ℝ) * (x₃ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_three_integer_solutions_l542_54293


namespace NUMINAMATH_CALUDE_orange_count_in_second_group_l542_54208

def apple_cost : ℚ := 21/100

theorem orange_count_in_second_group 
  (first_group : 6 * apple_cost + 3 * orange_cost = 177/100)
  (second_group : 2 * apple_cost + x * orange_cost = 127/100)
  (orange_cost : ℚ) (x : ℚ) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_in_second_group_l542_54208


namespace NUMINAMATH_CALUDE_intersection_single_point_l542_54275

def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem intersection_single_point (r : ℝ) (hr : r > 0) 
  (h_intersection : ∃! p, p ∈ A ∩ B r) : r = 3 ∨ r = 7 :=
sorry

end NUMINAMATH_CALUDE_intersection_single_point_l542_54275


namespace NUMINAMATH_CALUDE_friday_temperature_l542_54242

theorem friday_temperature
  (temp_mon : ℝ)
  (temp_tue : ℝ)
  (temp_wed : ℝ)
  (temp_thu : ℝ)
  (temp_fri : ℝ)
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 40)
  (h3 : temp_mon = 42) :
  temp_fri = 10 := by
sorry

end NUMINAMATH_CALUDE_friday_temperature_l542_54242


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l542_54211

theorem mark_and_carolyn_money_sum : (5 : ℚ) / 8 + (7 : ℚ) / 20 = 0.975 := by
  sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l542_54211


namespace NUMINAMATH_CALUDE_correct_calculation_l542_54240

theorem correct_calculation (x : ℤ) (h : x - 6 = 51) : 6 * x = 342 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l542_54240


namespace NUMINAMATH_CALUDE_vector_dot_product_properties_l542_54283

/-- Given vectors a and b in R², prove properties about their dot product. -/
theorem vector_dot_product_properties (α β : ℝ) (k : ℝ) 
  (h_k_pos : k > 0)
  (a : ℝ × ℝ := (Real.cos α, Real.sin α))
  (b : ℝ × ℝ := (Real.cos β, Real.sin β))
  (h_norm : ‖k • a + b‖ = Real.sqrt 3 * ‖a - k • b‖) :
  let dot := a.1 * b.1 + a.2 * b.2
  ∃ θ : ℝ,
    (dot = Real.cos (α - β)) ∧ 
    (dot = (k^2 + 1) / (4 * k)) ∧
    (0 ≤ θ ∧ θ ≤ π) ∧
    (dot ≥ 1/2) ∧
    (dot = 1/2 ↔ θ = π/3) :=
sorry

end NUMINAMATH_CALUDE_vector_dot_product_properties_l542_54283


namespace NUMINAMATH_CALUDE_f_range_and_triangle_property_l542_54218

noncomputable def f (x : Real) : Real :=
  2 * Real.sqrt 3 * Real.sin x * Real.cos x - 3 * Real.sin x ^ 2 - Real.cos x ^ 2 + 3

theorem f_range_and_triangle_property :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc 0 3) ∧
  (∀ (a b c : Real) (A B C : Real),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    A > 0 ∧ A < Real.pi ∧
    B > 0 ∧ B < Real.pi ∧
    C > 0 ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    b / a = Real.sqrt 3 ∧
    Real.sin (2 * A + C) / Real.sin A = 2 + 2 * Real.cos (A + C) →
    f B = 2) := by
  sorry

end NUMINAMATH_CALUDE_f_range_and_triangle_property_l542_54218


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l542_54226

/-- Given a car's speed over two hours, prove its speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (total_time : ℝ)
  (h1 : speed_first_hour = 50)
  (h2 : average_speed = 55)
  (h3 : total_time = 2)
  : ∃ (speed_second_hour : ℝ), speed_second_hour = 60 :=
by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l542_54226


namespace NUMINAMATH_CALUDE_football_team_handedness_ratio_l542_54276

/-- Given a football team with the following properties:
  - There are 70 players in total
  - 52 players are throwers
  - All throwers are right-handed
  - There are 64 right-handed players in total

  Prove that the ratio of left-handed players to non-throwers is 1:3 -/
theorem football_team_handedness_ratio 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (right_handed : ℕ) 
  (h1 : total_players = 70) 
  (h2 : throwers = 52) 
  (h3 : right_handed = 64) : 
  (total_players - right_handed) / (total_players - throwers) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_football_team_handedness_ratio_l542_54276


namespace NUMINAMATH_CALUDE_nail_polish_count_l542_54295

theorem nail_polish_count (num_girls : ℕ) (nails_per_girl : ℕ) : 
  num_girls = 5 → nails_per_girl = 20 → num_girls * nails_per_girl = 100 := by
  sorry

end NUMINAMATH_CALUDE_nail_polish_count_l542_54295


namespace NUMINAMATH_CALUDE_bridge_support_cans_l542_54216

/-- The weight of a full can of soda in ounces -/
def full_can_weight : ℕ := 12 + 2

/-- The weight of an empty can in ounces -/
def empty_can_weight : ℕ := 2

/-- The total weight the bridge must support in ounces -/
def total_bridge_weight : ℕ := 88

/-- The number of additional empty cans -/
def additional_empty_cans : ℕ := 2

/-- The number of full cans of soda the bridge needs to support -/
def num_full_cans : ℕ := (total_bridge_weight - additional_empty_cans * empty_can_weight) / full_can_weight

theorem bridge_support_cans : num_full_cans = 6 := by
  sorry

end NUMINAMATH_CALUDE_bridge_support_cans_l542_54216


namespace NUMINAMATH_CALUDE_cloth_cost_price_l542_54213

/-- Given a trader sells cloth with the following conditions:
  * Sells 45 meters of cloth
  * Total selling price is 4500 Rs
  * Profit per meter is 14 Rs
  Prove that the cost price of one meter of cloth is 86 Rs -/
theorem cloth_cost_price 
  (total_meters : ℕ) 
  (selling_price : ℕ) 
  (profit_per_meter : ℕ) 
  (h1 : total_meters = 45)
  (h2 : selling_price = 4500)
  (h3 : profit_per_meter = 14) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 86 := by
sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l542_54213


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l542_54217

theorem quadratic_unique_solution (a : ℚ) :
  (∃! x : ℚ, 2 * a * x^2 + 15 * x + 9 = 0) →
  (a = 25/8 ∧ ∃! x : ℚ, x = -12/5 ∧ 2 * a * x^2 + 15 * x + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l542_54217


namespace NUMINAMATH_CALUDE_max_expressions_greater_than_one_two_expressions_can_be_greater_than_one_max_expressions_greater_than_one_is_two_l542_54268

theorem max_expressions_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  (∃ (x y z : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧ z ∈ Set.range p ∧
    x > 1 ∧ y > 1 ∧ z > 1) → False :=
by sorry

theorem two_expressions_can_be_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ∃ (x y : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧
    x > 1 ∧ y > 1 :=
by sorry

theorem max_expressions_greater_than_one_is_two (a b c : ℝ) (h : a * b * c = 1) :
  (∃ (x y : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧
    x > 1 ∧ y > 1) ∧
  (∃ (x y z : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧ z ∈ Set.range p ∧
    x > 1 ∧ y > 1 ∧ z > 1) → False :=
by sorry

end NUMINAMATH_CALUDE_max_expressions_greater_than_one_two_expressions_can_be_greater_than_one_max_expressions_greater_than_one_is_two_l542_54268


namespace NUMINAMATH_CALUDE_calculation_proof_l542_54234

theorem calculation_proof (a b : ℝ) (h1 : a = 7) (h2 : b = 3) : 
  ((a^3 + b^3) / (a^2 - a*b + b^2) = 10) ∧ ((a^2 + b^2) / (a + b) = 5.8) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l542_54234


namespace NUMINAMATH_CALUDE_marble_jar_problem_l542_54220

theorem marble_jar_problem :
  ∀ (total_marbles : ℕ) (blue1 green1 blue2 green2 : ℕ),
    -- Jar 1 ratio condition
    7 * green1 = 2 * blue1 →
    -- Jar 2 ratio condition
    8 * green2 = blue2 →
    -- Equal total marbles in each jar
    blue1 + green1 = blue2 + green2 →
    -- Total green marbles
    green1 + green2 = 135 →
    -- Difference in blue marbles
    blue2 - blue1 = 45 :=
by sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l542_54220


namespace NUMINAMATH_CALUDE_danielas_age_l542_54204

/-- Given the ages and relationships of several people, prove Daniela's age --/
theorem danielas_age (clara_age : ℕ) (daniela_age evelina_age fidel_age caitlin_age : ℕ) :
  clara_age = 60 →
  daniela_age = evelina_age - 8 →
  evelina_age = clara_age / 3 →
  fidel_age = 2 * caitlin_age →
  fidel_age = evelina_age - 6 →
  daniela_age = 12 := by
sorry


end NUMINAMATH_CALUDE_danielas_age_l542_54204


namespace NUMINAMATH_CALUDE_inequality_solution_and_abc_inequality_l542_54255

theorem inequality_solution_and_abc_inequality :
  let solution_set := {x : ℝ | -1/2 < x ∧ x < 7/2}
  let p : ℝ := -3
  let q : ℝ := -7/4
  (∀ x, x ∈ solution_set ↔ |2*x - 3| < 4) →
  (∀ x, x ∈ solution_set ↔ x^2 + p*x + q < 0) →
  ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c →
    a + b + c = 2*p - 4*q →
    Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_abc_inequality_l542_54255


namespace NUMINAMATH_CALUDE_mean_of_car_counts_l542_54202

theorem mean_of_car_counts : 
  let counts : List ℕ := [30, 14, 14, 21, 25]
  (counts.sum / counts.length : ℚ) = 20.8 := by sorry

end NUMINAMATH_CALUDE_mean_of_car_counts_l542_54202


namespace NUMINAMATH_CALUDE_iris_jacket_purchase_l542_54243

theorem iris_jacket_purchase (jacket_price shorts_price pants_price : ℕ)
  (shorts_quantity pants_quantity : ℕ) (total_spent : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  pants_price = 12 →
  shorts_quantity = 2 →
  pants_quantity = 4 →
  total_spent = 90 →
  ∃ (jacket_quantity : ℕ), 
    jacket_quantity * jacket_price + 
    shorts_quantity * shorts_price + 
    pants_quantity * pants_price = total_spent ∧
    jacket_quantity = 3 :=
by sorry

end NUMINAMATH_CALUDE_iris_jacket_purchase_l542_54243


namespace NUMINAMATH_CALUDE_quadratic_transformation_l542_54228

/-- Transformation of a quadratic equation under a linear substitution -/
theorem quadratic_transformation (A B C D E F α β γ β' γ' : ℝ) :
  let Δ := A * C - B^2
  let x := λ x' y' : ℝ => α * x' + β * y' + γ
  let y := λ x' y' : ℝ => x' + β' * y' + γ'
  let original_eq := λ x y : ℝ => A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F
  ∃ a b : ℝ, 
    (Δ > 0 → ∀ x' y' : ℝ, original_eq (x x' y') (y x' y') = 0 ↔ x'^2 / a^2 + y'^2 / b^2 = 1) ∧
    (Δ < 0 → ∀ x' y' : ℝ, original_eq (x x' y') (y x' y') = 0 ↔ x'^2 / a^2 - y'^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l542_54228


namespace NUMINAMATH_CALUDE_inequality_system_solution_l542_54277

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - a ≥ b ∧ 2*x - a - 1 < 2*b) ↔ (3 ≤ x ∧ x < 5)) →
  a = -3 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l542_54277


namespace NUMINAMATH_CALUDE_snowman_volume_l542_54260

theorem snowman_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4 / 3) * π * r^3
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8 + sphere_volume 10 = (7168 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_snowman_volume_l542_54260
