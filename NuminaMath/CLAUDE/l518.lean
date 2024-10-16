import Mathlib

namespace NUMINAMATH_CALUDE_van_distance_proof_l518_51887

theorem van_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 32 →
  (initial_time * 3 / 2) * new_speed = 288 := by
  sorry

end NUMINAMATH_CALUDE_van_distance_proof_l518_51887


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l518_51889

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if there are at least six consecutive nonprime numbers before n, false otherwise -/
def hasSixConsecutiveNonprimes (n : ℕ) : Prop := sorry

theorem smallest_prime_after_six_nonprimes : 
  ∃ (k : ℕ), 
    isPrime k ∧ 
    hasSixConsecutiveNonprimes k ∧ 
    (∀ (m : ℕ), m < k → ¬(isPrime m ∧ hasSixConsecutiveNonprimes m)) ∧
    k = 97 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l518_51889


namespace NUMINAMATH_CALUDE_triangles_forming_square_even_l518_51899

theorem triangles_forming_square_even (n : ℕ) (a : ℕ) : 
  (n * 6 = a * a) → Even n := by sorry

end NUMINAMATH_CALUDE_triangles_forming_square_even_l518_51899


namespace NUMINAMATH_CALUDE_bisection_method_root_existence_l518_51855

theorem bisection_method_root_existence
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_cont : ContinuousOn f (Set.Icc a b))
  (h_sign : f a * f b < 0)
  (h_a_neg : f a < 0)
  (h_b_pos : f b > 0)
  (h_mid_pos : f ((a + b) / 2) > 0) :
  ∃ x ∈ Set.Ioo a ((a + b) / 2), f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_root_existence_l518_51855


namespace NUMINAMATH_CALUDE_jason_final_cards_l518_51868

def pokemon_card_transactions (initial_cards : ℕ) 
  (benny_trade_out benny_trade_in : ℕ) 
  (sean_trade_out sean_trade_in : ℕ) 
  (given_to_brother : ℕ) : ℕ :=
  initial_cards - benny_trade_out + benny_trade_in - sean_trade_out + sean_trade_in - given_to_brother

theorem jason_final_cards : 
  pokemon_card_transactions 5 2 3 3 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jason_final_cards_l518_51868


namespace NUMINAMATH_CALUDE_locus_is_conic_locus_degenerate_line_locus_circle_l518_51843

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in the first quadrant -/
structure Square where
  a : ℝ
  A : Point
  B : Point

/-- Defines the locus of a point P relative to the square -/
def locus (s : Square) (P : Point) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ θ : ℝ, 
    x = P.x * Real.sin θ + (s.a - P.x) * Real.cos θ ∧
    y = (s.a - P.y) * Real.sin θ + P.y * Real.cos θ}

theorem locus_is_conic (s : Square) (P : Point) 
  (h1 : s.A.y = 0 ∧ s.B.x = 0)  -- A is on x-axis, B is on y-axis
  (h2 : 0 ≤ P.x ∧ P.x ≤ 2*s.a ∧ 0 ≤ P.y ∧ P.y ≤ 2*s.a)  -- P is inside or on the square
  : ∃ (A B C D E F : ℝ), 
    A * P.x^2 + B * P.x * P.y + C * P.y^2 + D * P.x + E * P.y + F = 0 :=
sorry

theorem locus_degenerate_line (s : Square) (P : Point)
  (h : P.y = P.x)  -- P is on the diagonal
  : ∃ (m b : ℝ), ∀ (x y : ℝ), (x, y) ∈ locus s P → y = m * x + b :=
sorry

theorem locus_circle (s : Square) (P : Point)
  (h : P.x = s.a ∧ P.y = 0)  -- P is at midpoint of AB
  : ∃ (c : Point) (r : ℝ), ∀ (x y : ℝ), 
    (x, y) ∈ locus s P → (x - c.x)^2 + (y - c.y)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_conic_locus_degenerate_line_locus_circle_l518_51843


namespace NUMINAMATH_CALUDE_sum_of_specific_values_l518_51806

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem sum_of_specific_values (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 3)
  (h_f_1 : f 1 = 2014) :
  f 2013 + f 2014 + f 2015 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_values_l518_51806


namespace NUMINAMATH_CALUDE_initial_fund_calculation_l518_51826

theorem initial_fund_calculation (initial_per_employee final_per_employee undistributed : ℕ) : 
  initial_per_employee = 50 →
  final_per_employee = 45 →
  undistributed = 95 →
  (initial_per_employee - final_per_employee) * (undistributed / (initial_per_employee - final_per_employee)) = 950 := by
  sorry

end NUMINAMATH_CALUDE_initial_fund_calculation_l518_51826


namespace NUMINAMATH_CALUDE_modified_ohara_triple_49_64_l518_51865

/-- Definition of a Modified O'Hara triple -/
def isModifiedOHaraTriple (a b x : ℕ+) : Prop :=
  (a.val : ℝ).sqrt + (b.val : ℝ).sqrt = (x.val : ℝ)^2

/-- Theorem: If (49, 64, x) is a Modified O'Hara triple, then x = √113 -/
theorem modified_ohara_triple_49_64 (x : ℕ+) :
  isModifiedOHaraTriple 49 64 x → x.val = Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_modified_ohara_triple_49_64_l518_51865


namespace NUMINAMATH_CALUDE_vincent_book_cost_l518_51898

/-- Calculates the total cost of Vincent's books --/
def total_cost (animal_books train_books history_books cooking_books : ℕ) 
  (animal_price outer_space_price train_price history_price cooking_price : ℕ) : ℕ :=
  animal_books * animal_price + 
  1 * outer_space_price + 
  train_books * train_price + 
  history_books * history_price + 
  cooking_books * cooking_price

/-- Theorem stating that Vincent's total book cost is $356 --/
theorem vincent_book_cost : 
  total_cost 10 3 5 2 16 20 14 18 22 = 356 := by
  sorry


end NUMINAMATH_CALUDE_vincent_book_cost_l518_51898


namespace NUMINAMATH_CALUDE_solution_difference_l518_51841

theorem solution_difference (a b : ℝ) (ha : a ≠ 0) 
  (h : a^2 - b*a - 4*a = 0) : a - b = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l518_51841


namespace NUMINAMATH_CALUDE_base4_equals_base2_l518_51862

-- Define a function to convert a number from base 4 to base 10
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

-- Define a function to convert a number from base 2 to base 10
def base2ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (2 ^ i)) 0

-- Theorem statement
theorem base4_equals_base2 :
  base4ToDecimal [0, 1, 0, 1] = base2ToDecimal [0, 0, 1, 0, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base4_equals_base2_l518_51862


namespace NUMINAMATH_CALUDE_fraction2012_is_16_45_l518_51867

/-- Represents a fraction in the sequence -/
structure Fraction :=
  (numerator : Nat)
  (denominator : Nat)
  (h1 : numerator ≤ denominator / 2)
  (h2 : numerator > 0)
  (h3 : denominator > 0)

/-- The sequence of fractions not exceeding 1/2 -/
def fractionSequence : Nat → Fraction := sorry

/-- The 2012th fraction in the sequence -/
def fraction2012 : Fraction := fractionSequence 2012

/-- Theorem stating that the 2012th fraction is 16/45 -/
theorem fraction2012_is_16_45 :
  fraction2012.numerator = 16 ∧ fraction2012.denominator = 45 := by sorry

end NUMINAMATH_CALUDE_fraction2012_is_16_45_l518_51867


namespace NUMINAMATH_CALUDE_polynomial_product_l518_51828

theorem polynomial_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^2 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_l518_51828


namespace NUMINAMATH_CALUDE_bulls_and_heat_games_l518_51871

/-- Given that the Chicago Bulls won 70 games and the Miami Heat won 5 more games than the Bulls,
    prove that the total number of games won by both teams together is 145. -/
theorem bulls_and_heat_games (bulls_games : ℕ) (heat_games : ℕ) : 
  bulls_games = 70 → 
  heat_games = bulls_games + 5 → 
  bulls_games + heat_games = 145 := by
sorry

end NUMINAMATH_CALUDE_bulls_and_heat_games_l518_51871


namespace NUMINAMATH_CALUDE_complex_square_root_of_18i_l518_51827

theorem complex_square_root_of_18i :
  ∀ (z : ℂ), (∃ (x y : ℝ), z = x + y * I ∧ x > 0 ∧ z^2 = 18 * I) → z = 3 + 3 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_of_18i_l518_51827


namespace NUMINAMATH_CALUDE_cross_flag_center_area_ratio_l518_51832

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  centerArea : ℝ
  crossSymmetric : Bool
  crossUniformWidth : Bool
  crossAreaRatio : crossArea = 0.49 * side * side

/-- Theorem: If the cross occupies 49% of the flag's area, then the center square occupies 25.14% of the flag's area -/
theorem cross_flag_center_area_ratio (flag : CrossFlag) :
  flag.crossSymmetric ∧ flag.crossUniformWidth →
  flag.centerArea / (flag.side * flag.side) = 0.2514 := by
  sorry

end NUMINAMATH_CALUDE_cross_flag_center_area_ratio_l518_51832


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l518_51895

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (30 / 100) * (80 / 100) * y = (24 / 100) * y := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l518_51895


namespace NUMINAMATH_CALUDE_multiple_of_p_capital_l518_51883

theorem multiple_of_p_capital (P Q R : ℚ) (total_profit : ℚ) 
  (h1 : ∃ x : ℚ, x * P = 6 * Q)
  (h2 : ∃ x : ℚ, x * P = 10 * R)
  (h3 : total_profit = 4650)
  (h4 : R * total_profit / (P + Q + R) = 900) :
  ∃ x : ℚ, x * P = 10 * R ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_multiple_of_p_capital_l518_51883


namespace NUMINAMATH_CALUDE_fruit_punch_ratio_l518_51891

theorem fruit_punch_ratio (orange_punch apple_juice cherry_punch total_punch : ℝ) : 
  orange_punch = 4.5 →
  apple_juice = cherry_punch - 1.5 →
  total_punch = orange_punch + cherry_punch + apple_juice →
  total_punch = 21 →
  cherry_punch / orange_punch = 2 := by
sorry

end NUMINAMATH_CALUDE_fruit_punch_ratio_l518_51891


namespace NUMINAMATH_CALUDE_min_value_A_l518_51840

theorem min_value_A (x y z w : ℝ) :
  ∃ (A : ℝ), A = (1 + Real.sqrt 2) / 2 ∧
  (∀ (B : ℝ), (x*y + 2*y*z + z*w ≤ B*(x^2 + y^2 + z^2 + w^2)) → A ≤ B) ∧
  (x*y + 2*y*z + z*w ≤ A*(x^2 + y^2 + z^2 + w^2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_A_l518_51840


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l518_51803

/-- The speed of a boat in still water, given its downstream speed and the current speed -/
theorem boat_speed_in_still_water
  (downstream_speed : ℝ) -- Speed of the boat downstream
  (current_speed : ℝ)    -- Speed of the current
  (h1 : downstream_speed = 36) -- Given downstream speed
  (h2 : current_speed = 6)     -- Given current speed
  : downstream_speed - current_speed = 30 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l518_51803


namespace NUMINAMATH_CALUDE_condo_rented_units_l518_51800

/-- Represents the number of units of each bedroom type in a condominium -/
structure CondoUnits where
  one_bedroom : ℕ
  two_bedroom : ℕ
  three_bedroom : ℕ

/-- Represents the number of rented units of each bedroom type in a condominium -/
structure RentedUnits where
  one_bedroom : ℕ
  two_bedroom : ℕ
  three_bedroom : ℕ

def total_units (c : CondoUnits) : ℕ :=
  c.one_bedroom + c.two_bedroom + c.three_bedroom

def total_rented (r : RentedUnits) : ℕ :=
  r.one_bedroom + r.two_bedroom + r.three_bedroom

theorem condo_rented_units 
  (c : CondoUnits)
  (r : RentedUnits)
  (h1 : total_units c = 1200)
  (h2 : total_rented r = 700)
  (h3 : r.one_bedroom * 3 = r.two_bedroom * 2)
  (h4 : r.one_bedroom * 2 = r.three_bedroom)
  (h5 : r.two_bedroom * 2 = c.two_bedroom)
  : c.two_bedroom - r.two_bedroom = 231 := by
  sorry

end NUMINAMATH_CALUDE_condo_rented_units_l518_51800


namespace NUMINAMATH_CALUDE_linear_function_proof_l518_51845

/-- A linear function passing through (-2, 0) with the form y = ax + 1 -/
def linear_function (x : ℝ) : ℝ → ℝ := λ a ↦ a * x + 1

theorem linear_function_proof :
  ∃ a : ℝ, (∀ x : ℝ, linear_function x a = (1/2) * x + 1) ∧ linear_function (-2) a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l518_51845


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l518_51804

/-- Represents an oblique parallelepiped with given properties -/
structure ObliqueParallelepiped where
  lateral_edge_projection : ℝ
  height : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculates the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : ObliqueParallelepiped) : ℝ := sorry

/-- Calculates the volume of the parallelepiped -/
def volume (p : ObliqueParallelepiped) : ℝ := sorry

/-- Theorem stating the lateral surface area and volume of the given parallelepiped -/
theorem parallelepiped_properties :
  let p : ObliqueParallelepiped := {
    lateral_edge_projection := 5,
    height := 12,
    rhombus_area := 24,
    rhombus_diagonal := 8
  }
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l518_51804


namespace NUMINAMATH_CALUDE_yevgeniy_age_unique_l518_51858

def birth_year (y : ℕ) := 1900 + y

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Define the condition from the problem
def condition (y : ℕ) : Prop :=
  y ≥ 0 ∧ y < 100 ∧ (2011 - birth_year y = sum_of_digits (birth_year y))

-- The theorem to prove
theorem yevgeniy_age_unique :
  ∃! y : ℕ, condition y ∧ (2014 - birth_year y = 23) :=
sorry

end NUMINAMATH_CALUDE_yevgeniy_age_unique_l518_51858


namespace NUMINAMATH_CALUDE_smallest_s_is_six_l518_51860

-- Define the triangle sides
def a : ℝ := 7.5
def b : ℝ := 13

-- Define the property of s being the smallest whole number that forms a valid triangle
def is_smallest_valid_s (s : ℕ) : Prop :=
  (s : ℝ) + a > b ∧ 
  (s : ℝ) + b > a ∧ 
  a + b > (s : ℝ) ∧
  ∀ t : ℕ, t < s → ¬((t : ℝ) + a > b ∧ (t : ℝ) + b > a ∧ a + b > (t : ℝ))

-- Theorem statement
theorem smallest_s_is_six : is_smallest_valid_s 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_s_is_six_l518_51860


namespace NUMINAMATH_CALUDE_equation_system_solution_l518_51850

theorem equation_system_solution :
  ∃ (x₁ x₂ : ℝ),
    (∀ x y : ℝ, 5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1 →
      x = x₁ ∨ x = x₂) ∧
    x₁ = (-21 + Real.sqrt 641) / 50 ∧
    x₂ = (-21 - Real.sqrt 641) / 50 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l518_51850


namespace NUMINAMATH_CALUDE_calculate_savings_l518_51821

/-- Given a person's income and expenditure ratio, and their income, calculate their savings. -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) 
    (h1 : income_ratio = 8)
    (h2 : expenditure_ratio = 7)
    (h3 : income = 40000) :
  income - (expenditure_ratio * income / income_ratio) = 5000 := by
  sorry

#check calculate_savings

end NUMINAMATH_CALUDE_calculate_savings_l518_51821


namespace NUMINAMATH_CALUDE_hyperbola_condition_l518_51876

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k + 1) + y^2 / (k - 5) = 1 ∧ (k + 1) * (k - 5) < 0

-- Define the condition
def condition (k : ℝ) : Prop := 0 ≤ k ∧ k < 3

-- Theorem statement
theorem hyperbola_condition :
  (∀ k, condition k → is_hyperbola k) ∧
  (∃ k, is_hyperbola k ∧ ¬condition k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l518_51876


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l518_51807

/-- The number of different types of ice cream cones available. -/
def num_cone_types : ℕ := 2

/-- The number of different ice cream flavors available. -/
def num_flavors : ℕ := 4

/-- The total number of different ways to order ice cream. -/
def total_combinations : ℕ := num_cone_types * num_flavors

/-- Theorem stating that the total number of different ways to order ice cream is 8. -/
theorem ice_cream_combinations : total_combinations = 8 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l518_51807


namespace NUMINAMATH_CALUDE_toms_apple_purchase_l518_51884

/-- The problem of determining how many kg of apples Tom purchased -/
theorem toms_apple_purchase (apple_price mango_price total_paid : ℕ) 
  (mango_quantity : ℕ) (h1 : apple_price = 70) (h2 : mango_price = 75) 
  (h3 : mango_quantity = 9) (h4 : total_paid = 1235) :
  ∃ (apple_quantity : ℕ), 
    apple_quantity * apple_price + mango_quantity * mango_price = total_paid ∧ 
    apple_quantity = 8 := by
  sorry

end NUMINAMATH_CALUDE_toms_apple_purchase_l518_51884


namespace NUMINAMATH_CALUDE_egyptian_341_correct_l518_51894

/-- Represents an Egyptian numeral symbol -/
inductive EgyptianSymbol
  | hundreds
  | tens
  | ones

/-- Converts an Egyptian symbol to its numeric value -/
def symbolValue (s : EgyptianSymbol) : ℕ :=
  match s with
  | EgyptianSymbol.hundreds => 100
  | EgyptianSymbol.tens => 10
  | EgyptianSymbol.ones => 1

/-- Represents a list of Egyptian symbols -/
def EgyptianNumber := List EgyptianSymbol

/-- Converts an Egyptian number to its decimal value -/
def egyptianToDecimal (en : EgyptianNumber) : ℕ :=
  en.foldl (fun acc s => acc + symbolValue s) 0

/-- The Egyptian representation of 234 -/
def egyptian234 : EgyptianNumber :=
  [EgyptianSymbol.hundreds, EgyptianSymbol.tens, EgyptianSymbol.tens,
   EgyptianSymbol.ones, EgyptianSymbol.ones, EgyptianSymbol.ones, EgyptianSymbol.ones]

/-- The Egyptian representation of 123 -/
def egyptian123 : EgyptianNumber :=
  [EgyptianSymbol.tens, EgyptianSymbol.tens, EgyptianSymbol.tens,
   EgyptianSymbol.ones, EgyptianSymbol.ones, EgyptianSymbol.ones]

/-- The proposed Egyptian representation of 341 -/
def egyptian341 : EgyptianNumber :=
  [EgyptianSymbol.hundreds, EgyptianSymbol.hundreds, EgyptianSymbol.hundreds,
   EgyptianSymbol.tens, EgyptianSymbol.tens, EgyptianSymbol.tens, EgyptianSymbol.tens,
   EgyptianSymbol.ones]

theorem egyptian_341_correct :
  egyptianToDecimal egyptian234 = 234 ∧
  egyptianToDecimal egyptian123 = 123 →
  egyptianToDecimal egyptian341 = 341 :=
by sorry

end NUMINAMATH_CALUDE_egyptian_341_correct_l518_51894


namespace NUMINAMATH_CALUDE_parking_cost_excess_hours_l518_51811

theorem parking_cost_excess_hours (base_cost : ℝ) (avg_cost : ℝ) (excess_cost : ℝ) : 
  base_cost = 10 →
  avg_cost = 2.4722222222222223 →
  (base_cost + 7 * excess_cost) / 9 = avg_cost →
  excess_cost = 1.75 := by
sorry

end NUMINAMATH_CALUDE_parking_cost_excess_hours_l518_51811


namespace NUMINAMATH_CALUDE_max_candies_one_student_l518_51873

/-- Given a class of students, proves the maximum number of candies one student could have taken -/
theorem max_candies_one_student 
  (n : ℕ) -- number of students
  (mean : ℕ) -- mean number of candies per student
  (min_candies : ℕ) -- minimum number of candies per student
  (h1 : n = 25) -- there are 25 students
  (h2 : mean = 6) -- the mean number of candies is 6
  (h3 : min_candies = 2) -- each student takes at least 2 candies
  : ∃ (max_candies : ℕ), max_candies = 102 ∧ 
    max_candies = n * mean - (n - 1) * min_candies :=
by sorry

end NUMINAMATH_CALUDE_max_candies_one_student_l518_51873


namespace NUMINAMATH_CALUDE_donation_growth_rate_l518_51824

theorem donation_growth_rate 
  (initial_donation : ℝ) 
  (third_day_donation : ℝ) 
  (h1 : initial_donation = 10000)
  (h2 : third_day_donation = 12100) :
  ∃ (rate : ℝ), 
    initial_donation * (1 + rate)^2 = third_day_donation ∧ 
    rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_donation_growth_rate_l518_51824


namespace NUMINAMATH_CALUDE_working_hours_growth_equation_l518_51877

theorem working_hours_growth_equation 
  (initial_hours : ℝ) 
  (final_hours : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_hours = 40) 
  (h2 : final_hours = 48.4) :
  initial_hours * (1 + growth_rate)^2 = final_hours := by
sorry

end NUMINAMATH_CALUDE_working_hours_growth_equation_l518_51877


namespace NUMINAMATH_CALUDE_tree_planting_around_lake_l518_51835

theorem tree_planting_around_lake (circumference : ℕ) (willow_interval : ℕ) : 
  circumference = 1200 → willow_interval = 10 → 
  (circumference / willow_interval + circumference / willow_interval = 240) := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_around_lake_l518_51835


namespace NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l518_51849

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), y = 2 * x ∧ x + y = 12 ∧ x = 4 ∧ y = 8 := by sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 3 * x + 5 * y = 21 ∧ 2 * x - 5 * y = -11 ∧ x = 2 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l518_51849


namespace NUMINAMATH_CALUDE_shirts_sold_l518_51810

/-- Proves that the number of shirts sold is 4 given the conditions of the problem -/
theorem shirts_sold (total_money : ℕ) (num_dresses : ℕ) (price_dress : ℕ) (price_shirt : ℕ) :
  total_money = 69 →
  num_dresses = 7 →
  price_dress = 7 →
  price_shirt = 5 →
  (total_money - num_dresses * price_dress) / price_shirt = 4 := by
  sorry

end NUMINAMATH_CALUDE_shirts_sold_l518_51810


namespace NUMINAMATH_CALUDE_no_two_solutions_l518_51897

/-- The equation has either zero or infinitely many solutions for any real parameter a -/
theorem no_two_solutions (a : ℝ) : ¬∃! (p q : ℝ × ℝ), 
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  (x₁^2 + y₁^2 + 2*x₁ = |x₁ - a| - 1) ∧ 
  (x₂^2 + y₂^2 + 2*x₂ = |x₂ - a| - 1) ∧
  p ≠ q :=
by
  sorry

end NUMINAMATH_CALUDE_no_two_solutions_l518_51897


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l518_51861

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- One side of the triangle -/
  a : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Radius of the circumscribed circle -/
  R : ℝ
  /-- The side length is positive -/
  a_pos : 0 < a
  /-- The inscribed radius is positive -/
  r_pos : 0 < r
  /-- The circumscribed radius is positive -/
  R_pos : 0 < R

/-- Theorem: The perimeter of the special triangle is 24 -/
theorem special_triangle_perimeter (t : SpecialTriangle)
    (h1 : t.a = 6)
    (h2 : t.r = 2)
    (h3 : t.R = 5) :
    ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ t.a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_perimeter_l518_51861


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l518_51813

/-- A parabola with equation y^2 = 6x -/
structure Parabola where
  equation : ∀ x y, y^2 = 6*x

/-- A point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 6*x

/-- Two lines intersecting the parabola -/
structure IntersectingLines (C : Parabola) (P : PointOnParabola C) where
  A : PointOnParabola C
  B : PointOnParabola C
  slope_AB : (B.y - A.y) / (B.x - A.x) = 2
  sum_reciprocal_slopes : 
    ((P.y - A.y) / (P.x - A.x))⁻¹ + ((P.y - B.y) / (P.x - B.x))⁻¹ = 3

/-- The theorem to be proved -/
theorem parabola_intersection_theorem 
  (C : Parabola) 
  (P : PointOnParabola C) 
  (L : IntersectingLines C P) : 
  P.y = 15/2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l518_51813


namespace NUMINAMATH_CALUDE_eccentricity_relation_l518_51851

-- Define the eccentricities and point coordinates
variable (e₁ e₂ : ℝ)
variable (O F₁ F₂ P : ℝ × ℝ)

-- Define the conditions
def is_standard_ellipse_hyperbola : Prop :=
  0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1

def foci_on_x_axis : Prop :=
  ∃ c : ℝ, F₁ = (c, 0) ∧ F₂ = (-c, 0)

def O_is_origin : Prop :=
  O = (0, 0)

def P_on_both_curves : Prop :=
  ∃ (x y : ℝ), P = (x, y)

def distance_condition : Prop :=
  2 * ‖P - O‖ = ‖F₁ - F₂‖

-- State the theorem
theorem eccentricity_relation
  (h₁ : is_standard_ellipse_hyperbola e₁ e₂)
  (h₂ : foci_on_x_axis F₁ F₂)
  (h₃ : O_is_origin O)
  (h₄ : P_on_both_curves P)
  (h₅ : distance_condition O F₁ F₂ P) :
  (e₁ * e₂) / Real.sqrt (e₁^2 + e₂^2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_relation_l518_51851


namespace NUMINAMATH_CALUDE_sales_volume_linear_profit_quadratic_max_profit_profit_3000_min_inventory_l518_51854

/-- Represents the daily sales model for a specialty store -/
structure SalesModel where
  x : ℝ  -- Selling price per item in yuan
  y : ℝ  -- Daily sales volume in items
  W : ℝ  -- Daily total profit in yuan
  h1 : 16 ≤ x ∧ x ≤ 48  -- Price constraints
  h2 : y = -10 * x + 560  -- Relationship between y and x
  h3 : W = (x - 16) * y  -- Definition of total profit

/-- The daily sales volume is a linear function of the selling price -/
theorem sales_volume_linear (model : SalesModel) :
  ∃ a b : ℝ, model.y = a * model.x + b :=
sorry

/-- The daily total profit is a quadratic function of the selling price -/
theorem profit_quadratic (model : SalesModel) :
  ∃ a b c : ℝ, model.W = a * model.x^2 + b * model.x + c :=
sorry

/-- The maximum daily profit occurs when the selling price is 36 yuan and equals 4000 yuan -/
theorem max_profit (model : SalesModel) :
  (∀ x : ℝ, 16 ≤ x ∧ x ≤ 48 → model.W ≤ 4000) ∧
  (∃ model' : SalesModel, model'.x = 36 ∧ model'.W = 4000) :=
sorry

/-- There exists a selling price that ensures a daily profit of 3000 yuan while minimizing inventory -/
theorem profit_3000_min_inventory (model : SalesModel) :
  ∃ x : ℝ, 16 ≤ x ∧ x ≤ 48 ∧
  (∃ model' : SalesModel, model'.x = x ∧ model'.W = 3000) ∧
  (∀ x' : ℝ, 16 ≤ x' ∧ x' ≤ 48 →
    (∃ model'' : SalesModel, model''.x = x' ∧ model''.W = 3000) →
    x ≤ x') :=
sorry

end NUMINAMATH_CALUDE_sales_volume_linear_profit_quadratic_max_profit_profit_3000_min_inventory_l518_51854


namespace NUMINAMATH_CALUDE_triangle_centroid_property_l518_51881

variable (A B C G : ℝ × ℝ)

def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def distance_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem triangle_centroid_property (h_centroid : is_centroid G A B C)
  (h_condition : distance_squared G A + 2 * distance_squared G B + 3 * distance_squared G C = 123) :
  distance_squared A B + distance_squared A C + distance_squared B C = 246 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_property_l518_51881


namespace NUMINAMATH_CALUDE_perimeter_of_shaded_region_l518_51830

/-- The perimeter of the shaded region formed by three touching circles -/
theorem perimeter_of_shaded_region (circle_circumference : ℝ) :
  circle_circumference = 36 →
  (3 : ℝ) * (circle_circumference / 6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_shaded_region_l518_51830


namespace NUMINAMATH_CALUDE_average_monthly_production_theorem_l518_51885

def initial_production : ℝ := 1000

def monthly_increases : List ℝ := [0.05, 0.07, 0.10, 0.04, 0.08, 0.05, 0.07, 0.06, 0.12, 0.10, 0.08]

def calculate_monthly_production (prev : ℝ) (increase : ℝ) : ℝ :=
  prev * (1 + increase)

def calculate_yearly_production (initial : ℝ) (increases : List ℝ) : ℝ :=
  initial + (increases.scanl calculate_monthly_production initial).sum

theorem average_monthly_production_theorem :
  let yearly_production := calculate_yearly_production initial_production monthly_increases
  let average_production := yearly_production / 12
  ∃ ε > 0, |average_production - 1445.084204| < ε :=
sorry

end NUMINAMATH_CALUDE_average_monthly_production_theorem_l518_51885


namespace NUMINAMATH_CALUDE_function_divisibility_l518_51839

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem function_divisibility 
  (f : ℤ → ℕ+) 
  (h : ∀ (m n : ℤ), is_divisible (f (m - n)) (f m - f n)) :
  ∀ (m n : ℤ), f m ≤ f n → is_divisible (f m) (f n) :=
sorry

end NUMINAMATH_CALUDE_function_divisibility_l518_51839


namespace NUMINAMATH_CALUDE_fraction_scaling_l518_51888

theorem fraction_scaling (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (6 * x + 6 * y) / ((6 * x) * (6 * y)) = (1 / 6) * ((x + y) / (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_scaling_l518_51888


namespace NUMINAMATH_CALUDE_weight_lifting_duration_l518_51818

-- Define the total practice time in minutes
def total_practice_time : ℕ := 120

-- Define the time spent on running and weight lifting combined
def run_lift_time : ℕ := total_practice_time / 2

-- Define the relationship between running and weight lifting time
def weight_lifting_time (x : ℕ) : Prop := 
  x + 2 * x = run_lift_time

-- Theorem statement
theorem weight_lifting_duration : 
  ∃ x : ℕ, weight_lifting_time x ∧ x = 20 := by sorry

end NUMINAMATH_CALUDE_weight_lifting_duration_l518_51818


namespace NUMINAMATH_CALUDE_max_k_value_l518_51878

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 8*x + 15 = 0 ∧ 
   y = k*x - 2 ∧ 
   ∃ cx cy : ℝ, cy = k*cx - 2 ∧ 
   (cx - x)^2 + (cy - y)^2 ≤ 1) → 
  k ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l518_51878


namespace NUMINAMATH_CALUDE_remainder_three_power_45_plus_4_mod_5_l518_51863

theorem remainder_three_power_45_plus_4_mod_5 : (3^45 + 4) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_45_plus_4_mod_5_l518_51863


namespace NUMINAMATH_CALUDE_batsman_average_is_35_l518_51836

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  total_runs : ℕ
  last_inning_runs : ℕ
  average_increase : ℕ

/-- Calculates the new average of a batsman after their latest inning -/
def new_average (b : Batsman) : ℚ :=
  (b.total_runs + b.last_inning_runs) / b.innings

/-- Theorem stating that under given conditions, the batsman's new average is 35 -/
theorem batsman_average_is_35 (b : Batsman) 
    (h1 : b.innings = 17)
    (h2 : b.last_inning_runs = 83)
    (h3 : b.average_increase = 3)
    (h4 : new_average b = (new_average b - b.average_increase) + b.average_increase) :
    new_average b = 35 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_is_35_l518_51836


namespace NUMINAMATH_CALUDE_pyramid_edges_count_l518_51805

/-- A prism is a polyhedron with two congruent bases and rectangular faces connecting corresponding edges of the bases. -/
structure Prism where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  sum_property : vertices + faces + edges = 50
  euler_formula : vertices - edges + faces = 2

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a point (apex). -/
structure Pyramid where
  base_edges : ℕ

/-- Given a prism, construct a pyramid with the same base shape. -/
def pyramid_from_prism (p : Prism) : Pyramid :=
  { base_edges := (p.edges / 3) }

theorem pyramid_edges_count (p : Prism) : 
  (pyramid_from_prism p).base_edges * 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edges_count_l518_51805


namespace NUMINAMATH_CALUDE_solve_equation_l518_51893

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (b : ℝ) : Prop :=
  (2 - i) * (4 * i) = 4 - b * i

-- State the theorem
theorem solve_equation : ∃ b : ℝ, equation b ∧ b = -8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l518_51893


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l518_51802

/-- The number of available colors for the triangles -/
def num_colors : ℕ := 7

/-- A large triangle is made up of this many smaller triangles -/
def triangles_per_large : ℕ := 4

/-- The number of corner triangles in a large triangle -/
def num_corners : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  let corner_same := num_colors -- All corners same color
  let corner_two_same := num_colors * (num_colors - 1) -- Two corners same, one different
  let corner_all_diff := choose num_colors num_corners -- All corners different
  let total_corner_combinations := corner_same + corner_two_same + corner_all_diff
  total_corner_combinations * num_colors -- Multiply by center triangle color choices

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 588 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l518_51802


namespace NUMINAMATH_CALUDE_test_scores_theorem_l518_51816

-- Define the total number of tests
def total_tests : ℕ := 13

-- Define the number of tests with scores exceeding 90
def high_score_tests : ℕ := 4

-- Define the number of tests taken by A and B
def A_tests : ℕ := 6
def B_tests : ℕ := 7

-- Define the number of excellent scores for A and B
def A_excellent : ℕ := 3
def B_excellent : ℕ := 4

-- Define the number of tests selected from A and B
def A_selected : ℕ := 4
def B_selected : ℕ := 3

-- Define the probability of selecting a test with score > 90
def prob_high_score : ℚ := high_score_tests / total_tests

-- Define the expected value of X (excellent scores when selecting 4 out of A's 6 tests)
def E_X : ℚ := 2

-- Define the expected value of Y (excellent scores when selecting 3 out of B's 7 tests)
def E_Y : ℚ := 12 / 7

theorem test_scores_theorem :
  (prob_high_score = 4 / 13) ∧
  (E_X = 2) ∧
  (E_Y = 12 / 7) ∧
  (E_X > E_Y) := by
  sorry

end NUMINAMATH_CALUDE_test_scores_theorem_l518_51816


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l518_51870

/-- Represents the price reduction equation for a medicine that undergoes two
    successive price reductions of the same percentage. -/
theorem medicine_price_reduction (x : ℝ) : 
  (58 : ℝ) * (1 - x)^2 = 43 ↔ 
  (∃ (initial_price final_price : ℝ),
    initial_price = 58 ∧
    final_price = 43 ∧
    final_price = initial_price * (1 - x)^2) :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l518_51870


namespace NUMINAMATH_CALUDE_round_trip_percentage_l518_51834

/-- The percentage of passengers with round-trip tickets, given the conditions -/
theorem round_trip_percentage (total_passengers : ℝ) 
  (h1 : 0 < total_passengers)
  (h2 : (0.2 : ℝ) * total_passengers = 
        (0.8 : ℝ) * (round_trip_passengers : ℝ)) : 
  round_trip_passengers / total_passengers = 0.25 := by
  sorry

#check round_trip_percentage

end NUMINAMATH_CALUDE_round_trip_percentage_l518_51834


namespace NUMINAMATH_CALUDE_crane_among_chickens_is_random_l518_51801

-- Define the type for events
inductive Event
| CoveringSky
| FumingOrifices
| StridingMeteor
| CraneAmongChickens

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  ∃ (outcome : Bool), (outcome = true ∨ outcome = false)

-- State the theorem
theorem crane_among_chickens_is_random :
  isRandomEvent Event.CraneAmongChickens :=
sorry

end NUMINAMATH_CALUDE_crane_among_chickens_is_random_l518_51801


namespace NUMINAMATH_CALUDE_fourth_root_of_sum_of_powers_of_two_l518_51846

theorem fourth_root_of_sum_of_powers_of_two :
  (2^3 + 2^4 + 2^5 + 2^6 : ℝ)^(1/4) = 2^(3/4) * 15^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_sum_of_powers_of_two_l518_51846


namespace NUMINAMATH_CALUDE_five_circles_intersection_l518_51825

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane --/
def Point := ℝ × ℝ

/-- Check if a point lies on a circle --/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- The main theorem --/
theorem five_circles_intersection (circles : Fin 5 → Circle) 
  (h : ∀ (s : Finset (Fin 5)) (hs : s.card = 4), ∃ p : Point, ∀ i ∈ s, point_on_circle p (circles i)) :
  ∃ p : Point, ∀ i : Fin 5, point_on_circle p (circles i) := by
  sorry


end NUMINAMATH_CALUDE_five_circles_intersection_l518_51825


namespace NUMINAMATH_CALUDE_average_of_abc_l518_51819

theorem average_of_abc (A B C : ℝ) 
  (eq1 : 1001 * C - 2002 * A = 4004)
  (eq2 : 1001 * B + 3003 * A = 5005) : 
  (A + B + C) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_l518_51819


namespace NUMINAMATH_CALUDE_negative_three_is_rational_l518_51869

theorem negative_three_is_rational : ℚ :=
  sorry

end NUMINAMATH_CALUDE_negative_three_is_rational_l518_51869


namespace NUMINAMATH_CALUDE_function_property_l518_51817

def is_solution_set (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S ↔ (x > 0 ∧ f x + f (x - 8) ≤ 2)

theorem function_property (f : ℝ → ℝ) (h1 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
  (h2 : ∀ x y, x > 0 → y > 0 → x < y → f x < f y) (h3 : f 3 = 1) :
  is_solution_set f (Set.Ioo 8 9) :=
sorry

end NUMINAMATH_CALUDE_function_property_l518_51817


namespace NUMINAMATH_CALUDE_intersecting_rectangles_area_l518_51809

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.length

/-- Represents the overlap between two rectangles -/
structure Overlap where
  width : ℕ
  length : ℕ

/-- Calculates the area of overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.length

theorem intersecting_rectangles_area (r1 r2 r3 : Rectangle) 
  (o12 o13 o23 o123 : Overlap) : 
  r1.width = 4 → r1.length = 12 →
  r2.width = 5 → r2.length = 10 →
  r3.width = 3 → r3.length = 6 →
  o12.width = 4 → o12.length = 5 →
  o13.width = 3 → o13.length = 4 →
  o23.width = 3 → o23.length = 3 →
  o123.width = 3 → o123.length = 3 →
  area r1 + area r2 + area r3 - (overlapArea o12 + overlapArea o13 + overlapArea o23) + overlapArea o123 = 84 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_rectangles_area_l518_51809


namespace NUMINAMATH_CALUDE_area_ratio_ABJ_ADE_l518_51890

/-- Represents a regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- Represents a triangle within the regular octagon -/
structure OctagonTriangle where
  vertices : Fin 3 → Point

/-- The area of a triangle -/
def area (t : OctagonTriangle) : ℝ := sorry

/-- The regular octagon ABCDEFGH -/
def octagon : RegularOctagon := sorry

/-- Triangle ABJ formed by two smaller equilateral triangles -/
def triangle_ABJ : OctagonTriangle := sorry

/-- Triangle ADE formed by connecting every third vertex of the octagon -/
def triangle_ADE : OctagonTriangle := sorry

theorem area_ratio_ABJ_ADE :
  area triangle_ABJ / area triangle_ADE = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_ABJ_ADE_l518_51890


namespace NUMINAMATH_CALUDE_tile_arrangements_l518_51838

def num_red_tiles : ℕ := 1
def num_blue_tiles : ℕ := 2
def num_green_tiles : ℕ := 2
def num_yellow_tiles : ℕ := 4

def total_tiles : ℕ := num_red_tiles + num_blue_tiles + num_green_tiles + num_yellow_tiles

theorem tile_arrangements :
  (total_tiles.factorial) / (num_red_tiles.factorial * num_blue_tiles.factorial * num_green_tiles.factorial * num_yellow_tiles.factorial) = 3780 :=
by sorry

end NUMINAMATH_CALUDE_tile_arrangements_l518_51838


namespace NUMINAMATH_CALUDE_prism_with_ten_diagonals_has_five_sides_l518_51859

/-- A right prism with n sides and d diagonals. -/
structure RightPrism where
  n : ℕ
  d : ℕ

/-- The number of diagonals in a right n-sided prism is 2n. -/
axiom diagonals_count (p : RightPrism) : p.d = 2 * p.n

/-- For a right prism with 10 diagonals, the number of sides is 5. -/
theorem prism_with_ten_diagonals_has_five_sides (p : RightPrism) (h : p.d = 10) : p.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_ten_diagonals_has_five_sides_l518_51859


namespace NUMINAMATH_CALUDE_bus_cost_proof_l518_51892

-- Define the cost of a bus ride
def bus_cost : ℝ := 3.75

-- Define the cost of a train ride
def train_cost : ℝ := bus_cost + 2.35

-- Theorem stating the conditions and the result to be proved
theorem bus_cost_proof :
  (train_cost = bus_cost + 2.35) ∧
  (train_cost + bus_cost = 9.85) →
  bus_cost = 3.75 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_cost_proof_l518_51892


namespace NUMINAMATH_CALUDE_f_properties_l518_51882

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem f_properties :
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y) ∧
  (∀ x : ℝ, x ≠ 0 → f x = f (-x)) ∧
  (∀ x : ℝ, x > 0 → deriv f x > 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l518_51882


namespace NUMINAMATH_CALUDE_triangle_rectangle_area_coefficient_l518_51886

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the rectangle PQRS
structure Rectangle :=
  (ω : ℝ)
  (α β : ℝ)

-- Define the area function for the rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  rect.α * rect.ω - rect.β * rect.ω^2

-- State the theorem
theorem triangle_rectangle_area_coefficient
  (triangle : Triangle)
  (rect : Rectangle)
  (h1 : triangle.a = 13)
  (h2 : triangle.b = 26)
  (h3 : triangle.c = 15)
  (h4 : rectangleArea rect = 0 → rect.ω = 26)
  (h5 : rectangleArea rect = (triangle.a * triangle.b) / 4 → rect.ω = 13) :
  rect.β = 105 / 338 := by
sorry

end NUMINAMATH_CALUDE_triangle_rectangle_area_coefficient_l518_51886


namespace NUMINAMATH_CALUDE_nancy_files_problem_l518_51879

theorem nancy_files_problem (deleted_files : ℕ) (files_per_folder : ℕ) (final_folders : ℕ) :
  deleted_files = 31 →
  files_per_folder = 6 →
  final_folders = 2 →
  deleted_files + (files_per_folder * final_folders) = 43 :=
by sorry

end NUMINAMATH_CALUDE_nancy_files_problem_l518_51879


namespace NUMINAMATH_CALUDE_smallest_number_l518_51822

theorem smallest_number (a b c d e : ℝ) (ha : a = 1.4) (hb : b = 1.2) (hc : c = 2.0) (hd : d = 1.5) (he : e = 2.1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d ∧ b ≤ e := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l518_51822


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l518_51874

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, 
    f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y + 4 * y^2 :=
by
  -- The unique function is f(x) = x^2
  use fun x => x^2
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l518_51874


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l518_51812

theorem regular_polygon_interior_angle_sum :
  ∀ n : ℕ,
  n > 2 →
  (360 : ℝ) / n = 20 →
  (n - 2 : ℝ) * 180 = 2880 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l518_51812


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l518_51814

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l518_51814


namespace NUMINAMATH_CALUDE_star_equation_roots_l518_51864

-- Define the "★" operation
def star (a b : ℝ) : ℝ := a^2 - b^2

-- Theorem statement
theorem star_equation_roots :
  let x₁ : ℝ := 4
  let x₂ : ℝ := -4
  (star (star 2 3) x₁ = 9) ∧ (star (star 2 3) x₂ = 9) ∧
  (∀ x : ℝ, star (star 2 3) x = 9 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_star_equation_roots_l518_51864


namespace NUMINAMATH_CALUDE_equation_solution_l518_51842

theorem equation_solution : ∃ x : ℚ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l518_51842


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l518_51875

/-- The number formed by concatenating all 2-digit integers from 10 to 99 -/
def N : ℕ := sorry

/-- The sum of all digits in N -/
def digitSum : ℕ := 855

theorem highest_power_of_three_dividing_N :
  ∃ (k : ℕ), 3^k ∣ N ∧ ¬(3^(k+1) ∣ N) ∧ k = 4 := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l518_51875


namespace NUMINAMATH_CALUDE_dog_groom_time_l518_51866

/-- The time (in hours) it takes to groom a cat -/
def cat_groom_time : ℝ := 0.5

/-- The total time (in hours) it takes to groom 5 dogs and 3 cats -/
def total_groom_time : ℝ := 14

/-- The number of dogs groomed -/
def num_dogs : ℕ := 5

/-- The number of cats groomed -/
def num_cats : ℕ := 3

/-- Theorem stating that the time to groom a dog is 2.5 hours -/
theorem dog_groom_time : 
  ∃ (dog_time : ℝ), 
    dog_time * num_dogs + cat_groom_time * num_cats = total_groom_time ∧ 
    dog_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_dog_groom_time_l518_51866


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l518_51844

/-- Calculates the average speed of a round trip given the speed of the outbound journey and the fact that the return journey takes twice as long. -/
theorem round_trip_average_speed (outbound_speed : ℝ) :
  outbound_speed = 51 →
  (2 * outbound_speed) / 3 = 34 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l518_51844


namespace NUMINAMATH_CALUDE_prime_equation_solution_l518_51880

theorem prime_equation_solution (p : ℕ) (hp : Prime p) :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l518_51880


namespace NUMINAMATH_CALUDE_flagpole_height_l518_51896

/-- Given a flagpole and a building under similar shadow-casting conditions,
    prove that the flagpole's height is 18 meters. -/
theorem flagpole_height 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 28)
  (h_building_shadow : building_shadow = 70)
  (h_similar_conditions : True)  -- This represents the similar conditions
  : ∃ (flagpole_height : ℝ), flagpole_height = 18 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l518_51896


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l518_51815

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h1 : (x + y) / 2 = 20) 
  (h2 : Real.sqrt (x * y) = Real.sqrt 132) : 
  x^2 + y^2 = 1336 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l518_51815


namespace NUMINAMATH_CALUDE_A_intersect_B_l518_51808

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | x > 2}

theorem A_intersect_B : A ∩ B = {3, 4} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l518_51808


namespace NUMINAMATH_CALUDE_max_value_zero_l518_51872

theorem max_value_zero (a : Real) (h1 : 0 ≤ a) (h2 : a ≤ 1) :
  (∀ x : Real, x ≤ Real.sqrt (a * 1 * 0) + Real.sqrt ((1 - a) * (1 - 1) * (1 - 0))) →
  (Real.sqrt (a * 1 * 0) + Real.sqrt ((1 - a) * (1 - 1) * (1 - 0))) = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_zero_l518_51872


namespace NUMINAMATH_CALUDE_third_sample_is_51_l518_51837

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalItems : Nat
  numGroups : Nat
  firstSample : Nat

/-- Calculates the sample for a given group in a systematic sampling -/
def getSample (s : SystematicSampling) (group : Nat) : Nat :=
  s.firstSample + (group - 1) * (s.totalItems / s.numGroups)

/-- Theorem: In a systematic sampling of 400 items into 20 groups, 
    if the first sample is 11, then the third sample will be 51 -/
theorem third_sample_is_51 (s : SystematicSampling) 
  (h1 : s.totalItems = 400) 
  (h2 : s.numGroups = 20) 
  (h3 : s.firstSample = 11) : 
  getSample s 3 = 51 := by
  sorry

/-- Example setup for the given problem -/
def exampleSampling : SystematicSampling := {
  totalItems := 400
  numGroups := 20
  firstSample := 11
}

#eval getSample exampleSampling 3

end NUMINAMATH_CALUDE_third_sample_is_51_l518_51837


namespace NUMINAMATH_CALUDE_square_difference_l518_51831

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l518_51831


namespace NUMINAMATH_CALUDE_consecutive_zeros_count_is_3719_l518_51847

/-- Sequence of numbers with no two consecutive zeros -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => a (n+1) + a n

/-- The number of 12-digit positive integers with digits 0 or 1 
    that have at least two consecutive 0's -/
def consecutive_zeros_count : ℕ := 2^12 - a 12

theorem consecutive_zeros_count_is_3719 : consecutive_zeros_count = 3719 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_zeros_count_is_3719_l518_51847


namespace NUMINAMATH_CALUDE_z_equals_3s_l518_51852

theorem z_equals_3s (z s : ℝ) (hz : z ≠ 0) (heq : z = Real.sqrt (6 * z * s - 9 * s^2)) : z = 3 * s := by
  sorry

end NUMINAMATH_CALUDE_z_equals_3s_l518_51852


namespace NUMINAMATH_CALUDE_evaluate_expression_l518_51823

theorem evaluate_expression (c : ℕ) (h : c = 3) : (c^c - c*(c-1)^c)^c = 27 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l518_51823


namespace NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l518_51848

/-- Given a point A(x,y) in a Cartesian coordinate system, 
    its coordinates with respect to the y-axis are (-x,y) -/
theorem coordinates_wrt_y_axis (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  let A_wrt_y_axis : ℝ × ℝ := (-x, y)
  A_wrt_y_axis = (- (A.1), A.2) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l518_51848


namespace NUMINAMATH_CALUDE_A_not_necessary_for_B_A_not_sufficient_for_B_A_neither_necessary_nor_sufficient_for_B_l518_51829

-- Define condition A
def condition_A (x y : ℝ) : Prop := x ≠ 1 ∧ y ≠ 2

-- Define condition B
def condition_B (x y : ℝ) : Prop := x + y ≠ 3

-- Theorem stating that A is not necessary for B
theorem A_not_necessary_for_B : ¬∀ x y : ℝ, condition_B x y → condition_A x y := by
  sorry

-- Theorem stating that A is not sufficient for B
theorem A_not_sufficient_for_B : ¬∀ x y : ℝ, condition_A x y → condition_B x y := by
  sorry

-- Main theorem combining the above results
theorem A_neither_necessary_nor_sufficient_for_B :
  (¬∀ x y : ℝ, condition_B x y → condition_A x y) ∧
  (¬∀ x y : ℝ, condition_A x y → condition_B x y) := by
  sorry

end NUMINAMATH_CALUDE_A_not_necessary_for_B_A_not_sufficient_for_B_A_neither_necessary_nor_sufficient_for_B_l518_51829


namespace NUMINAMATH_CALUDE_problem_solution_l518_51857

def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * x

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≥ 3 ↔ x ≥ 0) ∧
  (∀ x : ℝ, (f a x ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l518_51857


namespace NUMINAMATH_CALUDE_unique_fixed_point_l518_51833

noncomputable def F (a b c : ℝ) (x y z : ℝ) : ℝ × ℝ × ℝ :=
  ((Real.sqrt (c^2 + z^2) - z + Real.sqrt (c^2 + y^2) - y) / 2,
   (Real.sqrt (b^2 + z^2) - z + Real.sqrt (b^2 + x^2) - x) / 2,
   (Real.sqrt (a^2 + x^2) - x + Real.sqrt (a^2 + y^2) - y) / 2)

theorem unique_fixed_point (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! p : ℝ × ℝ × ℝ, F a b c p.1 p.2.1 p.2.2 = p ∧ p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_fixed_point_l518_51833


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l518_51856

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 1)^2 - 2) →
  QuadraticFunction a b c 3 = 7 →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l518_51856


namespace NUMINAMATH_CALUDE_expression_factorization_l518_51820

theorem expression_factorization (b : ℝ) : 
  (8 * b^3 + 45 * b^2 - 10) - (-12 * b^3 + 5 * b^2 - 10) = 20 * b^2 * (b + 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l518_51820


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l518_51853

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m - 1) * Complex.I + (m^2 - 1) : ℂ).re = 0 ∧ ((m - 1) * Complex.I + (m^2 - 1) : ℂ).im ≠ 0) → 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l518_51853
