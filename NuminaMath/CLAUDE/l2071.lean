import Mathlib

namespace NUMINAMATH_CALUDE_g_max_value_l2071_207101

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) := 4 * x - x^4

/-- The maximum value of g(x) on the interval [0, 2] is 3 -/
theorem g_max_value : ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ ∀ x ∈ Set.Icc 0 2, g x ≤ g c ∧ g c = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_max_value_l2071_207101


namespace NUMINAMATH_CALUDE_divisibility_of_power_difference_l2071_207166

theorem divisibility_of_power_difference (a b : ℕ) (h : a + b = 61) :
  (61 : ℤ) ∣ (a^100 - b^100) :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_power_difference_l2071_207166


namespace NUMINAMATH_CALUDE_exists_same_color_unit_apart_l2071_207179

/-- A coloring of the plane using three colors -/
def Coloring := ℝ × ℝ → Fin 3

/-- Two points are one unit apart -/
def one_unit_apart (p q : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1

/-- Main theorem: In any three-coloring of the plane, there exist two points of the same color that are exactly one unit apart -/
theorem exists_same_color_unit_apart (c : Coloring) : 
  ∃ (p q : ℝ × ℝ), c p = c q ∧ one_unit_apart p q := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_unit_apart_l2071_207179


namespace NUMINAMATH_CALUDE_third_number_proof_l2071_207130

theorem third_number_proof (A B C : ℕ+) : 
  A = 24 → B = 36 → Nat.gcd A (Nat.gcd B C) = 32 → Nat.lcm A (Nat.lcm B C) = 1248 → C = 32 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l2071_207130


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l2071_207190

theorem complex_root_modulus_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (n + 2) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l2071_207190


namespace NUMINAMATH_CALUDE_five_fourths_of_fifteen_fourths_l2071_207153

theorem five_fourths_of_fifteen_fourths (x : ℚ) : 
  x = 15 / 4 → (5 / 4 : ℚ) * x = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_fifteen_fourths_l2071_207153


namespace NUMINAMATH_CALUDE_inscribed_circle_ratio_l2071_207158

/-- A circle inscribed in a semicircle -/
structure InscribedCircle where
  R : ℝ  -- Radius of the semicircle
  r : ℝ  -- Radius of the inscribed circle
  O : ℝ × ℝ  -- Center of the semicircle
  A : ℝ × ℝ  -- One end of the semicircle's diameter
  P : ℝ × ℝ  -- Center of the inscribed circle
  h₁ : R > 0  -- Radius of semicircle is positive
  h₂ : r > 0  -- Radius of inscribed circle is positive
  h₃ : A = (O.1 - R, O.2)  -- A is R units to the left of O
  h₄ : dist P O = dist P A  -- P is equidistant from O and A

/-- The ratio of radii in an inscribed circle is 3:8 -/
theorem inscribed_circle_ratio (c : InscribedCircle) : c.r / c.R = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_ratio_l2071_207158


namespace NUMINAMATH_CALUDE_midpoint_line_slope_zero_l2071_207157

/-- The slope of the line containing the midpoints of the segments [(1, 1), (3, 4)] and [(4, 1), (7, 4)] is 0. -/
theorem midpoint_line_slope_zero : 
  let midpoint1 := ((1 + 3) / 2, (1 + 4) / 2)
  let midpoint2 := ((4 + 7) / 2, (1 + 4) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_line_slope_zero_l2071_207157


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2071_207115

/-- The equation of a hyperbola with foci at (-3, 0) and (3, 0), and |MA| - |MB| = 4 -/
theorem hyperbola_equation (x y : ℝ) :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (3, 0)
  let M : ℝ × ℝ := (x, y)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist M A - dist M B = 4) → (x > 0) →
  (x^2 / 4 - y^2 / 5 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2071_207115


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l2071_207120

/-- For any positive real number a not equal to 1, 
    the function f(x) = a^x + 1 passes through the point (0, 2) -/
theorem exponential_function_fixed_point 
  (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l2071_207120


namespace NUMINAMATH_CALUDE_jason_music_store_expenses_l2071_207105

theorem jason_music_store_expenses :
  let flute : ℝ := 142.46
  let music_tool : ℝ := 8.89
  let song_book : ℝ := 7.00
  let flute_case : ℝ := 35.25
  let music_stand : ℝ := 12.15
  let cleaning_kit : ℝ := 14.99
  let sheet_protectors : ℝ := 3.29
  flute + music_tool + song_book + flute_case + music_stand + cleaning_kit + sheet_protectors = 224.03 := by
  sorry

end NUMINAMATH_CALUDE_jason_music_store_expenses_l2071_207105


namespace NUMINAMATH_CALUDE_evaluate_expression_l2071_207138

theorem evaluate_expression : (8^6 : ℝ) / (4 * 8^3) = 128 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2071_207138


namespace NUMINAMATH_CALUDE_min_value_6x_5y_l2071_207102

theorem min_value_6x_5y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (2 * x + y) + 3 / (x + y) = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / (2 * x' + y') + 3 / (x' + y') = 2 →
    6 * x + 5 * y ≤ 6 * x' + 5 * y') ∧
  6 * x + 5 * y = (13 + 4 * Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_6x_5y_l2071_207102


namespace NUMINAMATH_CALUDE_company_capital_expenditure_l2071_207114

theorem company_capital_expenditure (C : ℚ) (C_pos : C > 0) : 
  let raw_material_cost : ℚ := (1 / 4) * C
  let remaining_after_raw : ℚ := C - raw_material_cost
  let machinery_cost : ℚ := (1 / 10) * remaining_after_raw
  let total_expenditure : ℚ := raw_material_cost + machinery_cost
  (C - total_expenditure) / C = 27 / 40 := by sorry

end NUMINAMATH_CALUDE_company_capital_expenditure_l2071_207114


namespace NUMINAMATH_CALUDE_blackboard_numbers_theorem_l2071_207128

theorem blackboard_numbers_theorem (n : ℕ) (h_n : n > 3) 
  (numbers : Fin n → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j) 
  (h_bound : ∀ i, numbers i < Nat.factorial (n - 1)) :
  ∃ (i j k l : Fin n), i ≠ k ∧ j ≠ l ∧ numbers i > numbers j ∧ numbers k > numbers l ∧
    (numbers i / numbers j : ℕ) = (numbers k / numbers l : ℕ) :=
sorry

end NUMINAMATH_CALUDE_blackboard_numbers_theorem_l2071_207128


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l2071_207168

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) 
  (h_diff : m ≠ n) 
  (h_perp : perpendicular m α) 
  (h_contained : contained_in n α) : 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l2071_207168


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2071_207100

/-- Proves that the common difference of an arithmetic sequence is 5,
    given the specified conditions. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ) -- First term
  (a_n : ℕ) -- Last term
  (S : ℕ) -- Sum of all terms
  (h_a : a = 5)
  (h_a_n : a_n = 50)
  (h_S : S = 275)
  : ∃ (n : ℕ) (d : ℕ), n > 1 ∧ d = 5 ∧ 
    a_n = a + (n - 1) * d ∧
    S = n * (a + a_n) / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2071_207100


namespace NUMINAMATH_CALUDE_equation_solution_l2071_207159

theorem equation_solution : 
  ∃ y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2071_207159


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2071_207150

/-- The number of ways to select books from odd and even positions -/
def select_books (total : Nat) : Nat :=
  (total / 2) * (total / 2)

/-- Theorem stating the total number of ways to select the books -/
theorem book_selection_theorem :
  let biology_books := 12
  let chemistry_books := 8
  (select_books biology_books) * (select_books chemistry_books) = 576 := by
  sorry

#eval select_books 12 * select_books 8

end NUMINAMATH_CALUDE_book_selection_theorem_l2071_207150


namespace NUMINAMATH_CALUDE_two_number_problem_l2071_207183

theorem two_number_problem (x y : ℚ) 
  (sum_eq : x + y = 40)
  (double_subtract : 2 * y - 4 * x = 12) :
  |y - x| = 52 / 3 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l2071_207183


namespace NUMINAMATH_CALUDE_factorization_equality_l2071_207111

theorem factorization_equality (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l2071_207111


namespace NUMINAMATH_CALUDE_not_all_zero_iff_one_nonzero_l2071_207169

theorem not_all_zero_iff_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_iff_one_nonzero_l2071_207169


namespace NUMINAMATH_CALUDE_fifth_power_sum_l2071_207103

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l2071_207103


namespace NUMINAMATH_CALUDE_wendy_albums_l2071_207127

/-- Given a total number of pictures, the number of pictures in the first album,
    and the number of pictures per album in the remaining albums,
    calculate the number of albums created for the remaining pictures. -/
def calculate_remaining_albums (total_pictures : ℕ) (first_album_pictures : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (total_pictures - first_album_pictures) / pictures_per_album

theorem wendy_albums :
  calculate_remaining_albums 79 44 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_wendy_albums_l2071_207127


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2071_207146

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2071_207146


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l2071_207122

theorem cos_two_theta_value (θ : Real) 
  (h : Real.exp (Real.log 2 * (-5/2 + 2 * Real.cos θ)) + 1 = Real.exp (Real.log 2 * (3/4 + Real.cos θ))) : 
  Real.cos (2 * θ) = 17/8 := by
sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l2071_207122


namespace NUMINAMATH_CALUDE_stratified_by_stage_is_most_reasonable_l2071_207109

-- Define the possible sampling methods
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

-- Define the characteristics of the population
structure PopulationCharacteristics where
  significantDifferenceByStage : Bool
  significantDifferenceByGender : Bool

-- Define the function to determine the most reasonable sampling method
def mostReasonableSamplingMethod (pop : PopulationCharacteristics) : SamplingMethod :=
  sorry

-- Theorem statement
theorem stratified_by_stage_is_most_reasonable 
  (pop : PopulationCharacteristics) 
  (h1 : pop.significantDifferenceByStage = true) 
  (h2 : pop.significantDifferenceByGender = false) :
  mostReasonableSamplingMethod pop = SamplingMethod.StratifiedByEducationalStage :=
sorry

end NUMINAMATH_CALUDE_stratified_by_stage_is_most_reasonable_l2071_207109


namespace NUMINAMATH_CALUDE_positive_product_of_positive_factors_l2071_207185

theorem positive_product_of_positive_factors (a b : ℝ) : a > 0 → b > 0 → a * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_product_of_positive_factors_l2071_207185


namespace NUMINAMATH_CALUDE_undefined_expression_expression_undefined_iff_x_eq_12_l2071_207172

theorem undefined_expression (x : ℝ) : 
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) := by sorry

theorem expression_undefined_iff_x_eq_12 :
  ∀ x : ℝ, (∃ y : ℝ, (3*x^3 - 5*x + 2) / (x^2 - 24*x + 144) = y) ↔ (x ≠ 12) := by sorry

end NUMINAMATH_CALUDE_undefined_expression_expression_undefined_iff_x_eq_12_l2071_207172


namespace NUMINAMATH_CALUDE_fourth_root_784_times_cube_root_512_l2071_207188

theorem fourth_root_784_times_cube_root_512 : 
  (784 : ℝ) ^ (1/4) * (512 : ℝ) ^ (1/3) = 16 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_784_times_cube_root_512_l2071_207188


namespace NUMINAMATH_CALUDE_carrie_first_day_miles_l2071_207134

/-- Represents the four-day trip driven by Carrie -/
structure CarrieTrip where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  chargeDistance : ℕ
  chargeCount : ℕ

/-- The conditions of Carrie's trip -/
def tripConditions (trip : CarrieTrip) : Prop :=
  trip.day2 = trip.day1 + 124 ∧
  trip.day3 = 159 ∧
  trip.day4 = 189 ∧
  trip.chargeDistance = 106 ∧
  trip.chargeCount = 7 ∧
  trip.day1 + trip.day2 + trip.day3 + trip.day4 = trip.chargeDistance * trip.chargeCount

/-- Theorem stating that Carrie drove 135 miles on the first day -/
theorem carrie_first_day_miles :
  ∀ (trip : CarrieTrip), tripConditions trip → trip.day1 = 135 :=
by sorry

end NUMINAMATH_CALUDE_carrie_first_day_miles_l2071_207134


namespace NUMINAMATH_CALUDE_unique_solution_l2071_207106

/-- A natural number n is a valid solution if both n^n + 1 and (2n)^(2n) + 1 are prime numbers. -/
def is_valid_solution (n : ℕ) : Prop :=
  Nat.Prime (n^n + 1) ∧ Nat.Prime ((2*n)^(2*n) + 1)

/-- Theorem stating that 2 is the only natural number satisfying the conditions. -/
theorem unique_solution :
  ∀ n : ℕ, is_valid_solution n ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2071_207106


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l2071_207112

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The day of the week given the number of days since Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Monday"
  | 1 => "Tuesday"
  | 2 => "Wednesday"
  | 3 => "Thursday"
  | 4 => "Friday"
  | 5 => "Saturday"
  | _ => "Sunday"

theorem secret_spread_theorem :
  ∃ n : ℕ, secret_spread n ≥ 2186 ∧
           ∀ m : ℕ, m < n → secret_spread m < 2186 ∧
           day_of_week n = "Sunday" :=
by
  sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l2071_207112


namespace NUMINAMATH_CALUDE_negative_division_example_l2071_207177

theorem negative_division_example : (-150) / (-25) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_example_l2071_207177


namespace NUMINAMATH_CALUDE_one_fifth_of_five_times_nine_l2071_207123

theorem one_fifth_of_five_times_nine : (1 / 5 : ℚ) * (5 * 9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_five_times_nine_l2071_207123


namespace NUMINAMATH_CALUDE_max_profit_morel_purchase_l2071_207148

/-- Represents the purchase and profit calculation for Morel mushrooms. -/
structure MorelPurchase where
  freshPrice : ℝ  -- Purchase price of fresh Morel mushrooms (RMB/kg)
  driedPrice : ℝ  -- Purchase price of dried Morel mushrooms (RMB/kg)
  freshRetail : ℝ  -- Retail price of fresh Morel mushrooms (RMB/kg)
  driedRetail : ℝ  -- Retail price of dried Morel mushrooms (RMB/kg)
  totalQuantity : ℝ  -- Total quantity to purchase (kg)

/-- Calculates the profit for a given purchase plan. -/
def calculateProfit (p : MorelPurchase) (freshQuant : ℝ) : ℝ :=
  let driedQuant := p.totalQuantity - freshQuant
  (p.freshRetail - p.freshPrice) * freshQuant + (p.driedRetail - p.driedPrice) * driedQuant

/-- Theorem stating that the maximum profit is achieved with the specified quantities. -/
theorem max_profit_morel_purchase (p : MorelPurchase)
    (h1 : p.freshPrice = 80)
    (h2 : p.driedPrice = 240)
    (h3 : p.freshRetail = 100)
    (h4 : p.driedRetail = 280)
    (h5 : p.totalQuantity = 1500) :
    ∃ (maxProfit : ℝ) (optimalFresh : ℝ),
      maxProfit = 37500 ∧
      optimalFresh = 1125 ∧
      ∀ (freshQuant : ℝ), 0 ≤ freshQuant ∧ freshQuant ≤ p.totalQuantity ∧
        3 * (p.totalQuantity - freshQuant) ≤ freshQuant →
        calculateProfit p freshQuant ≤ maxProfit := by
  sorry


end NUMINAMATH_CALUDE_max_profit_morel_purchase_l2071_207148


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2071_207194

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 5)) = Real.sqrt 10 → y = 41 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2071_207194


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2071_207187

theorem quadratic_root_difference : ∃ (x₁ x₂ : ℝ),
  (5 + 3 * Real.sqrt 2) * x₁^2 - (1 - Real.sqrt 2) * x₁ - 1 = 0 ∧
  (5 + 3 * Real.sqrt 2) * x₂^2 - (1 - Real.sqrt 2) * x₂ - 1 = 0 ∧
  x₁ ≠ x₂ ∧
  max x₁ x₂ - min x₁ x₂ = 2 * Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2071_207187


namespace NUMINAMATH_CALUDE_distinct_remainders_mod_14_l2071_207104

theorem distinct_remainders_mod_14 : ∃ (a b c d e : ℕ),
  1 ≤ a ∧ a ≤ 13 ∧
  1 ≤ b ∧ b ≤ 13 ∧
  1 ≤ c ∧ c ≤ 13 ∧
  1 ≤ d ∧ d ≤ 13 ∧
  1 ≤ e ∧ e ≤ 13 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a * b) % 14 ≠ (a * c) % 14 ∧
  (a * b) % 14 ≠ (a * d) % 14 ∧
  (a * b) % 14 ≠ (a * e) % 14 ∧
  (a * b) % 14 ≠ (b * c) % 14 ∧
  (a * b) % 14 ≠ (b * d) % 14 ∧
  (a * b) % 14 ≠ (b * e) % 14 ∧
  (a * b) % 14 ≠ (c * d) % 14 ∧
  (a * b) % 14 ≠ (c * e) % 14 ∧
  (a * b) % 14 ≠ (d * e) % 14 ∧
  (a * c) % 14 ≠ (a * d) % 14 ∧
  (a * c) % 14 ≠ (a * e) % 14 ∧
  (a * c) % 14 ≠ (b * c) % 14 ∧
  (a * c) % 14 ≠ (b * d) % 14 ∧
  (a * c) % 14 ≠ (b * e) % 14 ∧
  (a * c) % 14 ≠ (c * d) % 14 ∧
  (a * c) % 14 ≠ (c * e) % 14 ∧
  (a * c) % 14 ≠ (d * e) % 14 ∧
  (a * d) % 14 ≠ (a * e) % 14 ∧
  (a * d) % 14 ≠ (b * c) % 14 ∧
  (a * d) % 14 ≠ (b * d) % 14 ∧
  (a * d) % 14 ≠ (b * e) % 14 ∧
  (a * d) % 14 ≠ (c * d) % 14 ∧
  (a * d) % 14 ≠ (c * e) % 14 ∧
  (a * d) % 14 ≠ (d * e) % 14 ∧
  (a * e) % 14 ≠ (b * c) % 14 ∧
  (a * e) % 14 ≠ (b * d) % 14 ∧
  (a * e) % 14 ≠ (b * e) % 14 ∧
  (a * e) % 14 ≠ (c * d) % 14 ∧
  (a * e) % 14 ≠ (c * e) % 14 ∧
  (a * e) % 14 ≠ (d * e) % 14 ∧
  (b * c) % 14 ≠ (b * d) % 14 ∧
  (b * c) % 14 ≠ (b * e) % 14 ∧
  (b * c) % 14 ≠ (c * d) % 14 ∧
  (b * c) % 14 ≠ (c * e) % 14 ∧
  (b * c) % 14 ≠ (d * e) % 14 ∧
  (b * d) % 14 ≠ (b * e) % 14 ∧
  (b * d) % 14 ≠ (c * d) % 14 ∧
  (b * d) % 14 ≠ (c * e) % 14 ∧
  (b * d) % 14 ≠ (d * e) % 14 ∧
  (b * e) % 14 ≠ (c * d) % 14 ∧
  (b * e) % 14 ≠ (c * e) % 14 ∧
  (b * e) % 14 ≠ (d * e) % 14 ∧
  (c * d) % 14 ≠ (c * e) % 14 ∧
  (c * d) % 14 ≠ (d * e) % 14 ∧
  (c * e) % 14 ≠ (d * e) % 14 :=
by sorry

end NUMINAMATH_CALUDE_distinct_remainders_mod_14_l2071_207104


namespace NUMINAMATH_CALUDE_steven_owes_jeremy_l2071_207113

/-- The amount Steven pays Jeremy for cleaning rooms -/
theorem steven_owes_jeremy (base_rate : ℚ) (rooms_cleaned : ℚ) (bonus_threshold : ℚ) (bonus_rate : ℚ) : 
  base_rate = 13/3 →
  rooms_cleaned = 5/2 →
  bonus_threshold = 2 →
  bonus_rate = 1/2 →
  (if rooms_cleaned > bonus_threshold 
   then base_rate * rooms_cleaned + bonus_rate * rooms_cleaned
   else base_rate * rooms_cleaned) = 145/12 :=
by sorry

end NUMINAMATH_CALUDE_steven_owes_jeremy_l2071_207113


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l2071_207117

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℕ
  physics : ℕ
  chemistry : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.math + s.physics = 70 ∧
  s.chemistry > s.physics ∧
  (s.math + s.chemistry) / 2 = 45

/-- The theorem to be proved -/
theorem chemistry_physics_difference (s : Scores) 
  (h : satisfiesConditions s) : s.chemistry - s.physics = 20 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_physics_difference_l2071_207117


namespace NUMINAMATH_CALUDE_hundred_with_fewer_threes_l2071_207125

-- Define a datatype for arithmetic expressions
inductive Expr
  | Num : ℕ → Expr
  | Add : Expr → Expr → Expr
  | Sub : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr
  | Div : Expr → Expr → Expr

-- Function to count the number of 3's in an expression
def countThrees : Expr → ℕ
  | Expr.Num n => if n = 3 then 1 else 0
  | Expr.Add e1 e2 => countThrees e1 + countThrees e2
  | Expr.Sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.Mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.Div e1 e2 => countThrees e1 + countThrees e2

-- Function to evaluate an expression
def evaluate : Expr → ℚ
  | Expr.Num n => n
  | Expr.Add e1 e2 => evaluate e1 + evaluate e2
  | Expr.Sub e1 e2 => evaluate e1 - evaluate e2
  | Expr.Mul e1 e2 => evaluate e1 * evaluate e2
  | Expr.Div e1 e2 => evaluate e1 / evaluate e2

-- Theorem statement
theorem hundred_with_fewer_threes : 
  ∃ e : Expr, evaluate e = 100 ∧ countThrees e < 10 :=
sorry

end NUMINAMATH_CALUDE_hundred_with_fewer_threes_l2071_207125


namespace NUMINAMATH_CALUDE_min_k_value_l2071_207136

theorem min_k_value (x y k : ℝ) : 
  (x - y + 5 ≥ 0) → 
  (x ≤ 3) → 
  (x + y + k ≥ 0) → 
  (∃ z : ℝ, z = 2*x + 4*y ∧ z ≥ -6 ∧ ∀ w : ℝ, w = 2*x + 4*y → w ≥ z) →
  k ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l2071_207136


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l2071_207147

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The equation of one asymptote -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ
  /-- Condition that the first asymptote has equation y = 2x + 1 -/
  asymptote1_eq : ∀ x, asymptote1 x = 2 * x + 1
  /-- Condition that the foci have x-coordinate 4 -/
  foci_x_eq : foci_x = 4

/-- The theorem stating the equation of the other asymptote -/
theorem other_asymptote_equation (h : Hyperbola) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = -2 * x + 17) ∧ 
  (∀ x y, y = f x ↔ y = h.asymptote1 x ∨ y = -2 * x + 17) :=
sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l2071_207147


namespace NUMINAMATH_CALUDE_set_7_24_25_is_pythagorean_triple_l2071_207144

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

/-- The set (7, 24, 25) is a Pythagorean triple -/
theorem set_7_24_25_is_pythagorean_triple : is_pythagorean_triple 7 24 25 := by
  sorry

end NUMINAMATH_CALUDE_set_7_24_25_is_pythagorean_triple_l2071_207144


namespace NUMINAMATH_CALUDE_trig_identity_l2071_207131

/-- Proves that sin 69° cos 9° - sin 21° cos 81° = √3/2 -/
theorem trig_identity : Real.sin (69 * π / 180) * Real.cos (9 * π / 180) - 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2071_207131


namespace NUMINAMATH_CALUDE_ratio_equations_solution_l2071_207161

theorem ratio_equations_solution (x y z a : ℤ) : 
  (∃ k : ℤ, x = k ∧ y = 4*k ∧ z = 5*k) →
  y = 9*a^2 - 2*a - 8 →
  z = 10*a + 2 →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equations_solution_l2071_207161


namespace NUMINAMATH_CALUDE_train_carriages_count_l2071_207152

/-- Calculates the number of carriages in a train given specific conditions -/
theorem train_carriages_count (carriage_length engine_length : ℝ)
                               (train_speed : ℝ)
                               (bridge_crossing_time : ℝ)
                               (bridge_length : ℝ) :
  carriage_length = 60 →
  engine_length = 60 →
  train_speed = 60 * 1000 / 60 →
  bridge_crossing_time = 5 →
  bridge_length = 3.5 * 1000 →
  ∃ n : ℕ, n = 24 ∧ 
    n * carriage_length + engine_length = 
    train_speed * bridge_crossing_time - bridge_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_carriages_count_l2071_207152


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2071_207118

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 100 = 0 → a^4 - 201*a = 10100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2071_207118


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2071_207116

/-- A quadratic function with axis of symmetry at x = 9.5 and p(1) = 2 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (19 - x)) →  -- symmetry about x = 9.5
  p a b c 1 = 2 →                            -- p(1) = 2
  p a b c 18 = 2 :=                          -- p(18) = 2
by
  sorry

#check quadratic_symmetry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2071_207116


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l2071_207126

/-- Given a rectangle with initial dimensions 4 × 6 inches, if shortening one side by 2 inches
    results in an area of 12 square inches, then shortening the other side by 1 inch
    results in an area of 20 square inches. -/
theorem rectangle_area_problem :
  ∀ (length width : ℝ),
  length = 4 ∧ width = 6 →
  (∃ (shortened_side : ℝ),
    (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
    shortened_side * (if shortened_side = length - 2 then width else length) = 12) →
  (if length - 2 < width - 2 then (length * (width - 1)) else ((length - 1) * width)) = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l2071_207126


namespace NUMINAMATH_CALUDE_product_of_roots_l2071_207156

theorem product_of_roots (x : ℂ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  (∃ p q r : ℂ, x^3 - 15*x^2 + 75*x - 50 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 50) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2071_207156


namespace NUMINAMATH_CALUDE_only_event1_is_random_l2071_207192

/-- Represents an event in a probability space -/
structure Event where
  description : String

/-- Defines what it means for an event to be random -/
def isRandomEvent (e : Event) : Prop :=
  sorry  -- Definition of random event

/-- Event 1: Tossing a coin twice in a row and getting heads both times -/
def event1 : Event := ⟨"Tossing a coin twice in a row and getting heads both times"⟩

/-- Event 2: Opposite charges attract each other -/
def event2 : Event := ⟨"Opposite charges attract each other"⟩

/-- Event 3: Water freezes at 1°C under standard atmospheric pressure -/
def event3 : Event := ⟨"Water freezes at 1°C under standard atmospheric pressure"⟩

/-- Theorem: Only event1 is a random event among the given events -/
theorem only_event1_is_random :
  isRandomEvent event1 ∧ ¬isRandomEvent event2 ∧ ¬isRandomEvent event3 :=
by
  sorry


end NUMINAMATH_CALUDE_only_event1_is_random_l2071_207192


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_5_and_10_l2071_207178

theorem greatest_three_digit_divisible_by_5_and_10 : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  n % 5 = 0 ∧ 
  n % 10 = 0 ∧ 
  ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 5 = 0 ∧ m % 10 = 0) → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_5_and_10_l2071_207178


namespace NUMINAMATH_CALUDE_expression_value_l2071_207151

theorem expression_value (a b : ℝ) (ha : a = 0.137) (hb : b = 0.098) :
  ((a + b)^2 - (a - b)^2) / (a * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2071_207151


namespace NUMINAMATH_CALUDE_xy_value_l2071_207196

theorem xy_value (x y : ℝ) (h : (|x| - 1)^2 + (2*y + 1)^2 = 0) : x * y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2071_207196


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l2071_207121

/-- Converts a binary number represented as a list of bits to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 10011₂ -/
def binary_10011 : List Bool := [true, true, false, false, true]

/-- Theorem stating that 10011₂ is equal to 19 in decimal -/
theorem binary_10011_equals_19 : binary_to_decimal binary_10011 = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l2071_207121


namespace NUMINAMATH_CALUDE_rd_investment_exceeds_200_million_in_2019_l2071_207171

/-- Proves that 2019 is the first year when the annual R&D bonus investment exceeds $200 million -/
theorem rd_investment_exceeds_200_million_in_2019 
  (initial_investment : ℝ) 
  (annual_increase_rate : ℝ) 
  (h1 : initial_investment = 130) 
  (h2 : annual_increase_rate = 0.12) : 
  ∃ (n : ℕ), 
    (n = 2019) ∧ 
    (initial_investment * (1 + annual_increase_rate) ^ (n - 2015) > 200) ∧ 
    (∀ m : ℕ, m < n → initial_investment * (1 + annual_increase_rate) ^ (m - 2015) ≤ 200) := by
  sorry

end NUMINAMATH_CALUDE_rd_investment_exceeds_200_million_in_2019_l2071_207171


namespace NUMINAMATH_CALUDE_minimum_time_for_given_problem_l2071_207145

/-- Represents the problem of replacing shades in chandeliers --/
structure ChandelierProblem where
  num_chandeliers : ℕ
  shades_per_chandelier : ℕ
  time_per_shade : ℕ
  num_electricians : ℕ

/-- Calculates the minimum time required to replace all shades --/
def minimum_replacement_time (p : ChandelierProblem) : ℕ :=
  let total_shades := p.num_chandeliers * p.shades_per_chandelier
  let total_work_time := total_shades * p.time_per_shade
  (total_work_time + p.num_electricians - 1) / p.num_electricians

/-- Theorem stating the minimum time for the given problem --/
theorem minimum_time_for_given_problem :
  let p : ChandelierProblem := {
    num_chandeliers := 60,
    shades_per_chandelier := 4,
    time_per_shade := 5,
    num_electricians := 48
  }
  minimum_replacement_time p = 25 := by sorry


end NUMINAMATH_CALUDE_minimum_time_for_given_problem_l2071_207145


namespace NUMINAMATH_CALUDE_train_length_problem_l2071_207193

/-- Proves that given two trains moving in opposite directions with specified speeds and time to pass,
    the length of the first train is 150 meters. -/
theorem train_length_problem (v1 v2 l2 t : ℝ) (h1 : v1 = 80) (h2 : v2 = 70) (h3 : l2 = 100) 
    (h4 : t = 5.999520038396928) : ∃ l1 : ℝ, l1 = 150 ∧ (v1 + v2) * t * (5/18) = l1 + l2 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l2071_207193


namespace NUMINAMATH_CALUDE_faye_candy_count_l2071_207132

/-- Calculates the remaining candy count after a given number of days -/
def remaining_candy (initial : ℕ) (daily_consumption : ℕ) (daily_addition : ℕ) (days : ℕ) : ℤ :=
  initial + days * daily_addition - days * daily_consumption

/-- Theorem: Faye's remaining candy count after y days -/
theorem faye_candy_count :
  ∀ (x y z : ℕ), remaining_candy 47 x z y = 47 + y * z - y * x :=
by
  sorry

#check faye_candy_count

end NUMINAMATH_CALUDE_faye_candy_count_l2071_207132


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_isosceles_right_triangle_l2071_207139

def isosceles_right_triangle (a c : ℝ) : Prop :=
  a > 0 ∧ c > 0 ∧ c^2 = 2 * a^2

theorem hypotenuse_length_of_isosceles_right_triangle (a c : ℝ) :
  isosceles_right_triangle a c →
  2 * a + c = 8 + 8 * Real.sqrt 2 →
  c = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_of_isosceles_right_triangle_l2071_207139


namespace NUMINAMATH_CALUDE_chord_division_ratio_l2071_207167

/-- Given a circle with radius 11 and a chord of length 18 intersected by a diameter at a point 7 units from the center, 
    the point of intersection divides the chord in a ratio of either 2:1 or 1:2. -/
theorem chord_division_ratio (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
  (h_r : r = 11) (h_chord : chord_length = 18) (h_dist : intersection_distance = 7) :
  ∃ (x y : ℝ), (x + y = chord_length ∧ 
    ((x / y = 2 ∧ y / x = 1/2) ∨ (x / y = 1/2 ∧ y / x = 2)) ∧
    x * y = (r - intersection_distance) * (r + intersection_distance)) :=
sorry

end NUMINAMATH_CALUDE_chord_division_ratio_l2071_207167


namespace NUMINAMATH_CALUDE_smallest_y_for_square_76545_l2071_207129

theorem smallest_y_for_square_76545 :
  ∃ y : ℕ+, 
    (∃ n : ℕ, 76545 * y.val = n^2) ∧ 
    ¬(3 ∣ y.val) ∧ 
    ¬(5 ∣ y.val) ∧
    (∀ z : ℕ+, z < y → ¬(∃ m : ℕ, 76545 * z.val = m^2) ∨ (3 ∣ z.val) ∨ (5 ∣ z.val)) ∧
    y = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_square_76545_l2071_207129


namespace NUMINAMATH_CALUDE_q_investment_time_l2071_207181

/-- Represents a partner in the investment problem -/
structure Partner where
  investment : ℝ
  time : ℝ
  profit : ℝ

/-- The investment problem setup -/
def InvestmentProblem (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 9 ∧
  p.time = 5 ∧
  p.investment * p.time / (q.investment * q.time) = p.profit / q.profit

/-- Theorem stating that Q's investment time is 9 months -/
theorem q_investment_time (p q : Partner) 
  (h : InvestmentProblem p q) : q.time = 9 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_time_l2071_207181


namespace NUMINAMATH_CALUDE_larger_integer_of_product_and_sum_l2071_207149

theorem larger_integer_of_product_and_sum (x y : ℤ) 
  (h_product : x * y = 30) 
  (h_sum : x + y = 13) : 
  max x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_of_product_and_sum_l2071_207149


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2071_207164

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define the interval
def I : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem quadratic_max_value (t : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ I → f t x ≤ m) ∧
  (t < 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ -2*t + 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = -2*t + 2)) ∧
  (t = 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = 2)) ∧
  (t > 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ 2*t + 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = 2*t + 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2071_207164


namespace NUMINAMATH_CALUDE_product_equals_584638125_l2071_207154

theorem product_equals_584638125 : 625 * 935421 = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_584638125_l2071_207154


namespace NUMINAMATH_CALUDE_robbery_trial_l2071_207142

theorem robbery_trial (A B C : Prop) 
  (h1 : (¬A ∨ B) → C)
  (h2 : ¬A → ¬C) : 
  A ∧ C ∧ (B ∨ ¬B) := by
sorry

end NUMINAMATH_CALUDE_robbery_trial_l2071_207142


namespace NUMINAMATH_CALUDE_tournament_committee_count_l2071_207184

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The minimum number of female members in each team -/
def min_females : ℕ := 2

/-- The number of members selected for the committee by the host team -/
def host_committee_size : ℕ := 3

/-- The number of members selected for the committee by non-host teams -/
def non_host_committee_size : ℕ := 2

/-- The total number of members in the tournament committee -/
def total_committee_size : ℕ := 10

/-- The number of possible tournament committees -/
def num_committees : ℕ := 1296540

theorem tournament_committee_count :
  (num_teams > 0) →
  (team_size ≥ host_committee_size) →
  (team_size ≥ non_host_committee_size) →
  (min_females ≥ non_host_committee_size) →
  (min_females < host_committee_size) →
  (num_teams * non_host_committee_size + host_committee_size = total_committee_size) →
  (num_committees = (num_teams - 1) * (Nat.choose team_size host_committee_size) * 
    (Nat.choose team_size non_host_committee_size)^(num_teams - 2) * 
    (Nat.choose min_females non_host_committee_size)) :=
by sorry

#check tournament_committee_count

end NUMINAMATH_CALUDE_tournament_committee_count_l2071_207184


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2071_207176

theorem nested_fraction_equality : 
  1 + 1 / (1 + 1 / (1 + 1 / 2)) = 8 / 5 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2071_207176


namespace NUMINAMATH_CALUDE_exists_more_than_20_components_l2071_207174

/-- A diagonal in a cell can be either left-to-right or right-to-left -/
inductive Diagonal
| LeftToRight
| RightToLeft

/-- A grid is represented as a function from coordinates to diagonals -/
def Grid := Fin 8 → Fin 8 → Diagonal

/-- A point in the grid -/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Two points are connected if there's a path of adjacent diagonals between them -/
def Connected (g : Grid) (p q : Point) : Prop := sorry

/-- A connected component is a maximal set of connected points -/
def ConnectedComponent (g : Grid) (s : Set Point) : Prop := sorry

/-- The number of connected components in a grid -/
def NumComponents (g : Grid) : ℕ := sorry

/-- There exists a configuration with more than 20 connected components -/
theorem exists_more_than_20_components : ∃ g : Grid, NumComponents g > 20 := by sorry

end NUMINAMATH_CALUDE_exists_more_than_20_components_l2071_207174


namespace NUMINAMATH_CALUDE_emily_trivia_score_l2071_207197

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) : 
  first_round + 33 - 48 = 1 → first_round = 16 := by
  sorry

end NUMINAMATH_CALUDE_emily_trivia_score_l2071_207197


namespace NUMINAMATH_CALUDE_parabola_cubic_intersection_l2071_207175

def parabola (x y : ℝ) : Prop := y = 3 * x^2 - 12 * x - 15

def cubic (x y : ℝ) : Prop := y = x^3 - 6 * x^2 + 11 * x - 6

def intersection_points : Set (ℝ × ℝ) := {(-1, 0), (1, -24), (9, 162)}

theorem parabola_cubic_intersection :
  ∀ x y : ℝ, (parabola x y ∧ cubic x y) ↔ (x, y) ∈ intersection_points :=
sorry

end NUMINAMATH_CALUDE_parabola_cubic_intersection_l2071_207175


namespace NUMINAMATH_CALUDE_root_product_sum_l2071_207195

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (Real.sqrt 2023 * x₁^3 - 4047 * x₁^2 + 4046 * x₁ - 1 = 0) ∧
  (Real.sqrt 2023 * x₂^3 - 4047 * x₂^2 + 4046 * x₂ - 1 = 0) ∧
  (Real.sqrt 2023 * x₃^3 - 4047 * x₃^2 + 4046 * x₃ - 1 = 0) →
  x₂ * (x₁ + x₃) = 2 + 1 / 2023 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l2071_207195


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2071_207155

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ (1^2 + 2*1 - m > 0)) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2071_207155


namespace NUMINAMATH_CALUDE_only_first_equation_has_nonzero_solution_l2071_207170

theorem only_first_equation_has_nonzero_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (a^2 + b^2) = a ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = Real.sqrt a * Real.sqrt b → a = 0 ∧ b = 0) ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = a * b → a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_first_equation_has_nonzero_solution_l2071_207170


namespace NUMINAMATH_CALUDE_initial_book_donations_l2071_207198

/-- Proves that the initial number of book donations is 300 given the conditions of the problem. -/
theorem initial_book_donations (
  people_donating : ℕ)
  (books_per_person : ℕ)
  (books_borrowed : ℕ)
  (remaining_books : ℕ)
  (h1 : people_donating = 10)
  (h2 : books_per_person = 5)
  (h3 : books_borrowed = 140)
  (h4 : remaining_books = 210) :
  people_donating * books_per_person + remaining_books + books_borrowed = 300 :=
by sorry


end NUMINAMATH_CALUDE_initial_book_donations_l2071_207198


namespace NUMINAMATH_CALUDE_trees_around_square_theorem_l2071_207137

/-- Represents a rectangle with trees planted along its sides -/
structure TreeRectangle where
  side_ad : ℕ  -- Number of trees along side AD
  side_ab : ℕ  -- Number of trees along side AB

/-- Calculates the number of trees around a square with side length equal to the longer side of the rectangle -/
def trees_around_square (rect : TreeRectangle) : ℕ :=
  4 * (rect.side_ad - 1) + 4

/-- Theorem stating that for a rectangle with 49 trees along AD and 25 along AB,
    the number of trees around the corresponding square is 196 -/
theorem trees_around_square_theorem (rect : TreeRectangle) 
        (h1 : rect.side_ad = 49) (h2 : rect.side_ab = 25) : 
        trees_around_square rect = 196 := by
  sorry

#eval trees_around_square ⟨49, 25⟩

end NUMINAMATH_CALUDE_trees_around_square_theorem_l2071_207137


namespace NUMINAMATH_CALUDE_problem_1_l2071_207124

theorem problem_1 (x : ℝ) (hx : x ≠ 0) :
  (-2 * x^5 + 3 * x^3 - (1/2) * x^2) / ((-1/2 * x)^2) = -8 * x^3 + 12 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2071_207124


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l2071_207180

/-- A circle C tangent to the line x-2=0 at point (2,1) with radius 3 -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the line x-2=0 at point (2,1) -/
  tangent_point : center.1 - 2 = radius ∨ center.1 - 2 = -radius
  /-- The point (2,1) lies on the circle -/
  on_circle : (2 - center.1)^2 + (1 - center.2)^2 = radius^2
  /-- The radius is 3 -/
  radius_is_three : radius = 3

/-- The equation of the circle is either (x+1)^2+(y-1)^2=9 or (x-5)^2+(y-1)^2=9 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) →
  ((∀ x y, (x + 1)^2 + (y - 1)^2 = 9) ∨ (∀ x y, (x - 5)^2 + (y - 1)^2 = 9)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l2071_207180


namespace NUMINAMATH_CALUDE_travel_time_proof_l2071_207182

/-- Given a person traveling at a constant speed, this theorem proves that
    the travel time is 5 hours when the distance is 500 km and the speed is 100 km/hr. -/
theorem travel_time_proof (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 500 ∧ speed = 100 ∧ time = distance / speed → time = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_proof_l2071_207182


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2071_207160

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2071_207160


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2071_207140

theorem expression_equals_zero (x : ℚ) (h : x = 1/3) :
  (2*x + 1) * (2*x - 1) + x * (3 - 4*x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2071_207140


namespace NUMINAMATH_CALUDE_radius_of_circle_in_spherical_coordinates_l2071_207199

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/4) is √2 -/
theorem radius_of_circle_in_spherical_coordinates : 
  let ρ : ℝ := 2
  let φ : ℝ := π / 4
  Real.sqrt (ρ^2 * Real.sin φ^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_in_spherical_coordinates_l2071_207199


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2071_207119

theorem quadratic_equation_m_value : 
  ∀ m : ℝ, 
  (∀ x : ℝ, (m - 1) * x^(|m| + 1) + 2 * m * x + 3 = 0 → 
    (|m| + 1 = 2 ∧ m - 1 ≠ 0)) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2071_207119


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l2071_207163

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - 2*a)^2 = 5

-- Define the points A, B, and P
def point_A : ℝ × ℝ := (3, 2)
def point_B : ℝ × ℝ := (1, 6)
def point_P : ℝ × ℝ := (-1, 3)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 2*x - y + 5 = 0
def tangent_line_2 (x y : ℝ) : Prop := x + 2*y - 5 = 0

theorem circle_and_tangent_lines :
  (circle_C point_A.1 point_A.2) ∧
  (circle_C point_B.1 point_B.2) ∧
  (∀ (x y : ℝ), circle_C x y → (x - 2)^2 + (y - 4)^2 = 5) ∧
  (∀ (x y : ℝ), (tangent_line_1 x y ∨ tangent_line_2 x y) →
    (∃ (t : ℝ), circle_C (t*x + (1-t)*point_P.1) (t*y + (1-t)*point_P.2)) ∧
    (∀ (s : ℝ), s ≠ t → ¬ circle_C (s*x + (1-s)*point_P.1) (s*y + (1-s)*point_P.2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l2071_207163


namespace NUMINAMATH_CALUDE_square_difference_equals_1380_l2071_207186

theorem square_difference_equals_1380 : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_1380_l2071_207186


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2071_207110

theorem polynomial_divisibility (m n : ℕ+) :
  ∃ q : Polynomial ℚ, (X^2 + X + 1) * q = X^(3*m.val + 1) + X^(3*n.val + 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2071_207110


namespace NUMINAMATH_CALUDE_train_speed_l2071_207141

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 140) (h2 : time = 6) :
  length / time = 140 / 6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2071_207141


namespace NUMINAMATH_CALUDE_divisible_by_nine_l2071_207162

theorem divisible_by_nine : ∃ (n : ℕ), 5742 = 9 * n := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l2071_207162


namespace NUMINAMATH_CALUDE_stating_raptors_score_l2071_207189

/-- 
Represents the scores of three teams in a cricket match, 
where the total score is 48 and one team wins over another by 18 points.
-/
structure CricketScores where
  eagles : ℕ
  raptors : ℕ
  hawks : ℕ
  total_is_48 : eagles + raptors + hawks = 48
  eagles_margin : eagles = raptors + 18

/-- 
Theorem stating that the Raptors' score is (30 - hawks) / 2
given the conditions of the cricket match.
-/
theorem raptors_score (scores : CricketScores) : 
  scores.raptors = (30 - scores.hawks) / 2 := by
  sorry

#check raptors_score

end NUMINAMATH_CALUDE_stating_raptors_score_l2071_207189


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l2071_207107

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 1 ∧ min = -17 := by sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l2071_207107


namespace NUMINAMATH_CALUDE_centroid_construction_condition_l2071_207143

/-- A function that checks if a number is divisible by all prime factors of another number -/
def isDivisibleByAllPrimeFactors (m n : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ n → p ∣ m

/-- The main theorem stating the condition for constructing the centroid -/
theorem centroid_construction_condition (n m : ℕ) (h : n ≥ 3) :
  (∃ (construction : Unit), True) ↔ (2 ∣ m ∧ isDivisibleByAllPrimeFactors m n) :=
sorry

end NUMINAMATH_CALUDE_centroid_construction_condition_l2071_207143


namespace NUMINAMATH_CALUDE_M_equals_N_l2071_207165

/-- Definition of set M -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- Definition of set N -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l2071_207165


namespace NUMINAMATH_CALUDE_smallest_prime_with_reverse_composite_l2071_207135

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

theorem smallest_prime_with_reverse_composite : 
  ∀ n : ℕ, 30 ≤ n ∧ n < 41 →
    ¬(is_prime n ∧ 
      is_composite (reverse_digits n) ∧ 
      ∃ m : ℕ, m ≠ 7 ∧ m > 1 ∧ m ∣ reverse_digits n) →
  is_prime 41 ∧ 
  is_composite (reverse_digits 41) ∧ 
  ∃ m : ℕ, m ≠ 7 ∧ m > 1 ∧ m ∣ reverse_digits 41 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_reverse_composite_l2071_207135


namespace NUMINAMATH_CALUDE_certain_number_proof_l2071_207133

theorem certain_number_proof : ∃ (x : ℚ), (2994 / x = 179) → x = 167 / 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2071_207133


namespace NUMINAMATH_CALUDE_trajectory_of_point_M_l2071_207173

/-- The trajectory of point M given the specified conditions -/
theorem trajectory_of_point_M :
  ∀ (M : ℝ × ℝ),
  let A : ℝ × ℝ := (0, -1)
  let B : ℝ × ℝ := (M.1, -3)
  let O : ℝ × ℝ := (0, 0)
  -- MB parallel to OA
  (∃ k : ℝ, B.1 - M.1 = k * A.1 ∧ B.2 - M.2 = k * A.2) →
  -- MA • AB = MB • BA
  ((A.1 - M.1) * (B.1 - A.1) + (A.2 - M.2) * (B.2 - A.2) =
   (B.1 - M.1) * (A.1 - B.1) + (B.2 - M.2) * (A.2 - B.2)) →
  -- Trajectory equation
  M.2 = (1/4) * M.1^2 - 2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_point_M_l2071_207173


namespace NUMINAMATH_CALUDE_student_count_l2071_207191

theorem student_count (total_average : ℝ) (group1_count : ℕ) (group1_average : ℝ)
                      (group2_count : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_count = 8 →
  group1_average = 14 →
  group2_count = 6 →
  group2_average = 16 →
  last_student_age = 17 →
  (group1_count : ℝ) * group1_average + (group2_count : ℝ) * group2_average + last_student_age = 15 * 15 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l2071_207191


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2071_207108

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The right focus F of the ellipse C -/
def F : ℝ × ℝ := (1, 0)

/-- The line L -/
def L (x : ℝ) : Prop := x = 4

/-- The left vertex A of the ellipse C -/
def A : ℝ × ℝ := (-2, 0)

/-- The ratio condition for any point P on C -/
def ratio_condition (P : ℝ × ℝ) : Prop :=
  C P.1 P.2 → 2 * Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = |P.1 - 4|

/-- The theorem to be proved -/
theorem fixed_point_theorem :
  ∀ D E M N : ℝ × ℝ,
  (∃ t : ℝ, C (F.1 + t * (D.1 - F.1)) (F.2 + t * (D.2 - F.2))) →
  (∃ t : ℝ, C (F.1 + t * (E.1 - F.1)) (F.2 + t * (E.2 - F.2))) →
  (∃ t : ℝ, M = (4, A.2 + t * (D.2 - A.2))) →
  (∃ t : ℝ, N = (4, A.2 + t * (E.2 - A.2))) →
  (∀ P : ℝ × ℝ, ratio_condition P) →
  ∃ O : ℝ × ℝ, O = (1, 0) ∧ 
    (M.1 - O.1)^2 + (M.2 - O.2)^2 = (N.1 - O.1)^2 + (N.2 - O.2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2071_207108
