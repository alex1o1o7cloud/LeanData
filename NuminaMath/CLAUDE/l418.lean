import Mathlib

namespace NUMINAMATH_CALUDE_lamp_height_difference_l418_41806

/-- The height difference between two lamps -/
theorem lamp_height_difference (old_height new_height : ℝ) 
  (h1 : old_height = 1) 
  (h2 : new_height = 2.3333333333333335) : 
  new_height - old_height = 1.3333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_lamp_height_difference_l418_41806


namespace NUMINAMATH_CALUDE_remaining_episodes_l418_41853

theorem remaining_episodes (seasons : Nat) (episodes_per_season : Nat) 
  (watched_fraction : Rat) (h1 : seasons = 12) (h2 : episodes_per_season = 20) 
  (h3 : watched_fraction = 1/3) : 
  seasons * episodes_per_season - (seasons * episodes_per_season * watched_fraction).floor = 160 := by
  sorry

end NUMINAMATH_CALUDE_remaining_episodes_l418_41853


namespace NUMINAMATH_CALUDE_find_number_l418_41864

theorem find_number : ∃ x : ℝ, (3 * x / 5 - 220) * 4 + 40 = 360 ∧ x = 500 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l418_41864


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l418_41802

/-- A perfect sphere inscribed in a cube -/
structure InscribedSphere where
  cube_side_length : ℝ
  touches_face_centers : Bool
  radius : ℝ

/-- Theorem: The radius of a perfect sphere inscribed in a cube with side length 2,
    such that it touches the center of each face, is equal to 1 -/
theorem inscribed_sphere_radius
  (s : InscribedSphere)
  (h1 : s.cube_side_length = 2)
  (h2 : s.touches_face_centers = true) :
  s.radius = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l418_41802


namespace NUMINAMATH_CALUDE_difference_number_and_three_fifths_l418_41830

theorem difference_number_and_three_fifths (n : ℚ) : n = 140 → n - (3 / 5 * n) = 56 := by
  sorry

end NUMINAMATH_CALUDE_difference_number_and_three_fifths_l418_41830


namespace NUMINAMATH_CALUDE_baby_nexus_monograms_l418_41882

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of letters to exclude (X and one other) --/
def excluded_letters : ℕ := 2

/-- The number of letters to choose for the monogram (first and middle initials) --/
def letters_to_choose : ℕ := 2

/-- Calculates the number of possible monograms for baby Nexus --/
def monogram_count : ℕ :=
  Nat.choose (alphabet_size - excluded_letters) letters_to_choose

theorem baby_nexus_monograms :
  monogram_count = 253 := by
  sorry

end NUMINAMATH_CALUDE_baby_nexus_monograms_l418_41882


namespace NUMINAMATH_CALUDE_max_y_value_l418_41897

theorem max_y_value (a b y : ℝ) (eq1 : a + b + y = 5) (eq2 : a * b + b * y + a * y = 3) :
  y ≤ 13/3 := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l418_41897


namespace NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l418_41825

theorem cylinder_in_hemisphere_height (r c h : ℝ) : 
  r > 0 ∧ c > 0 ∧ r > c ∧ r = 8 ∧ c = 3 → h = Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l418_41825


namespace NUMINAMATH_CALUDE_tan_60_plus_inverse_sqrt_3_l418_41818

theorem tan_60_plus_inverse_sqrt_3 :
  let tan_60 := Real.sqrt 3
  tan_60 + (Real.sqrt 3)⁻¹ = (4 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_tan_60_plus_inverse_sqrt_3_l418_41818


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_l418_41881

/-- Given a rectangular solid with adjacent face areas √2, √3, and √6,
    the surface area of its circumscribed sphere is 6π. -/
theorem circumscribed_sphere_area (x y z : ℝ) 
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  4 * Real.pi * ((Real.sqrt 6) / 2)^2 = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_circumscribed_sphere_area_l418_41881


namespace NUMINAMATH_CALUDE_percentage_calculation_l418_41869

theorem percentage_calculation (number : ℝ) (result : ℝ) (P : ℝ) : 
  number = 4400 → 
  result = 99 → 
  P * number = result → 
  P = 0.0225 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l418_41869


namespace NUMINAMATH_CALUDE_root_in_interval_l418_41811

-- Define the function f(x) = 2x - 3
def f (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l418_41811


namespace NUMINAMATH_CALUDE_problem_statement_l418_41833

theorem problem_statement (x : ℝ) (h : x = 2) : 4 * x^2 + (1/2) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l418_41833


namespace NUMINAMATH_CALUDE_rectangle_side_length_l418_41826

theorem rectangle_side_length (square_side : ℝ) (rect_side1 : ℝ) (rect_side2 : ℝ) :
  square_side = 5 →
  rect_side1 = 4 →
  square_side * square_side = rect_side1 * rect_side2 →
  rect_side2 = 6.25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l418_41826


namespace NUMINAMATH_CALUDE_right_triangle_area_l418_41843

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) (h5 : a^2 = 36) (h6 : b^2 = 64) : (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l418_41843


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l418_41836

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero. -/
axiom tangent_iff_discriminant_zero (a b c : ℝ) :
  (∃ x y : ℝ, y = a*x + b ∧ y^2 = c*x) →
  (∀ x y : ℝ, y = a*x + b → y^2 = c*x → (a*x + b)^2 = c*x) →
  b^2 = a*c

/-- The main theorem: if y = 3x + c is tangent to y^2 = 12x, then c = 1 -/
theorem tangent_line_to_parabola (c : ℝ) :
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x) →
  (∀ x y : ℝ, y = 3*x + c → y^2 = 12*x → (3*x + c)^2 = 12*x) →
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l418_41836


namespace NUMINAMATH_CALUDE_percentage_of_absent_students_l418_41870

theorem percentage_of_absent_students (total : ℕ) (present : ℕ) : 
  total = 50 → present = 43 → (((total - present : ℚ) / total) * 100 = 14) := by sorry

end NUMINAMATH_CALUDE_percentage_of_absent_students_l418_41870


namespace NUMINAMATH_CALUDE_triangle_side_length_l418_41837

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that if b = 6, c = 4, and A = 2B, then a = 2√15 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → c = 4 → A = 2 * B → 
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = Real.pi) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  a = 2 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l418_41837


namespace NUMINAMATH_CALUDE_real_roots_imply_m_value_l418_41800

theorem real_roots_imply_m_value (x m : ℝ) (i : ℂ) :
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) → m = 1/12 := by
sorry

end NUMINAMATH_CALUDE_real_roots_imply_m_value_l418_41800


namespace NUMINAMATH_CALUDE_line_shift_theorem_l418_41868

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_horizontal (l : Line) (units : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + l.slope * units }

/-- Shifts a line vertically -/
def shift_vertical (l : Line) (units : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - units }

/-- The theorem stating that shifting the line y = 2x - 1 left by 3 units
    and then down by 4 units results in the line y = 2x + 1 -/
theorem line_shift_theorem :
  let initial_line := Line.mk 2 (-1)
  let shifted_left := shift_horizontal initial_line 3
  let final_line := shift_vertical shifted_left 4
  final_line = Line.mk 2 1 := by
  sorry


end NUMINAMATH_CALUDE_line_shift_theorem_l418_41868


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l418_41890

/-- Linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- Theorem: For the linear function f(x) = 2x + 1, if f(-3) = y₁ and f(4) = y₂, then y₁ < y₂ -/
theorem y1_less_than_y2 (y₁ y₂ : ℝ) (h1 : f (-3) = y₁) (h2 : f 4 = y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l418_41890


namespace NUMINAMATH_CALUDE_boxes_left_to_sell_l418_41859

/-- Represents the number of cookie boxes sold to each customer --/
structure CustomerSales where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates the total number of boxes sold --/
def totalSold (sales : CustomerSales) : ℕ :=
  sales.first + sales.second + sales.third + sales.fourth + sales.fifth

/-- Represents Jill's cookie sales --/
def jillSales : CustomerSales where
  first := 5
  second := 4 * 5
  third := (4 * 5) / 2
  fourth := 3 * ((4 * 5) / 2)
  fifth := 10

/-- Jill's sales goal --/
def salesGoal : ℕ := 150

/-- Theorem stating that Jill has 75 boxes left to sell to reach her goal --/
theorem boxes_left_to_sell : salesGoal - totalSold jillSales = 75 := by
  sorry

end NUMINAMATH_CALUDE_boxes_left_to_sell_l418_41859


namespace NUMINAMATH_CALUDE_chris_bluray_purchase_l418_41858

/-- The number of Blu-ray movies Chris bought -/
def num_bluray : ℕ := sorry

/-- The number of DVD movies Chris bought -/
def num_dvd : ℕ := 8

/-- The price of each DVD movie -/
def price_dvd : ℚ := 12

/-- The price of each Blu-ray movie -/
def price_bluray : ℚ := 18

/-- The average price per movie -/
def avg_price : ℚ := 14

theorem chris_bluray_purchase :
  (num_dvd * price_dvd + num_bluray * price_bluray) / (num_dvd + num_bluray) = avg_price ∧
  num_bluray = 4 := by sorry

end NUMINAMATH_CALUDE_chris_bluray_purchase_l418_41858


namespace NUMINAMATH_CALUDE_alien_resource_conversion_l418_41845

/-- Converts a base-5 number represented as a list of digits to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem alien_resource_conversion :
  base5ToBase10 [3, 6, 2] = 83 := by
  sorry

end NUMINAMATH_CALUDE_alien_resource_conversion_l418_41845


namespace NUMINAMATH_CALUDE_chocolate_distribution_l418_41854

/-- The number of ways to distribute n distinct objects among k recipients -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a distribution satisfies the given conditions -/
def validDistribution (d : List ℕ) : Prop :=
  d.length = 3 ∧ d.sum = 8 ∧ d.all (· > 0) ∧ d.Nodup

theorem chocolate_distribution :
  sumOfDigits (distribute 8 3) = 24 :=
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l418_41854


namespace NUMINAMATH_CALUDE_power_division_equals_integer_l418_41820

theorem power_division_equals_integer : 3^18 / 27^2 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_integer_l418_41820


namespace NUMINAMATH_CALUDE_geometric_sequence_divisibility_l418_41886

/-- Given a geometric sequence with first term a₁ and second term a₂, 
    find the smallest n for which the n-th term is divisible by 10⁶ -/
theorem geometric_sequence_divisibility
  (a₁ : ℚ)
  (a₂ : ℕ)
  (h₁ : a₁ = 5 / 8)
  (h₂ : a₂ = 25)
  : ∃ n : ℕ, n > 0 ∧ 
    (∀ k < n, ¬(10^6 ∣ (a₁ * (a₂ / a₁)^(k - 1)))) ∧
    (10^6 ∣ (a₁ * (a₂ / a₁)^(n - 1))) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_divisibility_l418_41886


namespace NUMINAMATH_CALUDE_caravan_hens_count_l418_41841

/-- A caravan with hens, goats, camels, and keepers. -/
structure Caravan where
  hens : ℕ
  goats : ℕ
  camels : ℕ
  keepers : ℕ

/-- Calculate the total number of feet in the caravan. -/
def totalFeet (c : Caravan) : ℕ :=
  2 * c.hens + 4 * c.goats + 4 * c.camels + 2 * c.keepers

/-- Calculate the total number of heads in the caravan. -/
def totalHeads (c : Caravan) : ℕ :=
  c.hens + c.goats + c.camels + c.keepers

/-- The main theorem stating the number of hens in the caravan. -/
theorem caravan_hens_count : ∃ (c : Caravan), 
  c.goats = 45 ∧ 
  c.camels = 8 ∧ 
  c.keepers = 15 ∧ 
  totalFeet c = totalHeads c + 224 ∧ 
  c.hens = 50 := by
  sorry


end NUMINAMATH_CALUDE_caravan_hens_count_l418_41841


namespace NUMINAMATH_CALUDE_zain_total_coins_l418_41803

/-- Represents the number of coins of each type --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ
  halfDollars : ℕ

/-- Calculates the total number of coins --/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.quarters + coins.dimes + coins.nickels + coins.pennies + coins.halfDollars

/-- Represents Emerie's coin count --/
def emerieCoins : CoinCount :=
  { quarters := 6
  , dimes := 7
  , nickels := 5
  , pennies := 10
  , halfDollars := 2 }

/-- Calculates Zain's coin count based on Emerie's --/
def zainCoins (emerie : CoinCount) : CoinCount :=
  { quarters := emerie.quarters + 10
  , dimes := emerie.dimes + 10
  , nickels := emerie.nickels + 10
  , pennies := emerie.pennies + 10
  , halfDollars := emerie.halfDollars + 10 }

/-- Theorem: Zain has 80 coins in total --/
theorem zain_total_coins : totalCoins (zainCoins emerieCoins) = 80 := by
  sorry

end NUMINAMATH_CALUDE_zain_total_coins_l418_41803


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l418_41816

theorem smallest_prime_divisor_of_sum (p : Nat) : 
  Prime p ∧ p ∣ (2^14 + 7^12) ∧ ∀ q, Prime q → q ∣ (2^14 + 7^12) → p ≤ q → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l418_41816


namespace NUMINAMATH_CALUDE_carols_birthday_l418_41813

/-- Represents a date with a month and a day -/
structure Date where
  month : String
  day : Nat

/-- The list of possible dates for Carol's birthday -/
def possible_dates : List Date := [
  ⟨"January", 4⟩, ⟨"March", 8⟩, ⟨"June", 7⟩, ⟨"October", 7⟩,
  ⟨"January", 5⟩, ⟨"April", 8⟩, ⟨"June", 5⟩, ⟨"October", 4⟩,
  ⟨"January", 11⟩, ⟨"April", 9⟩, ⟨"July", 13⟩, ⟨"October", 8⟩
]

/-- Alberto knows the month but not the exact date -/
def alberto_knows_month (d : Date) : Prop :=
  ∃ (other : Date), other ∈ possible_dates ∧ other.month = d.month ∧ other ≠ d

/-- Bernardo knows the day but not the exact date -/
def bernardo_knows_day (d : Date) : Prop :=
  ∃ (other : Date), other ∈ possible_dates ∧ other.day = d.day ∧ other ≠ d

/-- Alberto's first statement: He can't determine the date, and he's sure Bernardo can't either -/
def alberto_statement1 (d : Date) : Prop :=
  alberto_knows_month d ∧ bernardo_knows_day d

/-- After Alberto's statement, Bernardo can determine the date -/
def bernardo_statement (d : Date) : Prop :=
  alberto_statement1 d ∧
  ∀ (other : Date), other ∈ possible_dates → other.day = d.day → alberto_statement1 other → other = d

/-- After Bernardo's statement, Alberto can also determine the date -/
def alberto_statement2 (d : Date) : Prop :=
  bernardo_statement d ∧
  ∀ (other : Date), other ∈ possible_dates → other.month = d.month → bernardo_statement other → other = d

/-- The theorem stating that Carol's birthday must be June 7 -/
theorem carols_birthday :
  ∃! (d : Date), d ∈ possible_dates ∧ alberto_statement2 d ∧ d = ⟨"June", 7⟩ := by
  sorry

end NUMINAMATH_CALUDE_carols_birthday_l418_41813


namespace NUMINAMATH_CALUDE_line_equation_l418_41834

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically --/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy - l.slope * dx }

/-- Check if two lines are identical --/
def Line.identical (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept = l2.intercept

/-- Check if two lines are symmetric about a point --/
def symmetricAbout (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ x y, (y = l1.slope * x + l1.intercept) ↔ 
         (2 * p.2 - y = l2.slope * (2 * p.1 - x) + l2.intercept)

theorem line_equation (l : Line) : 
  (translate (translate l 3 5) 1 (-2)).identical l ∧ 
  symmetricAbout l (translate l 3 5) (2, 3) →
  l.slope = 3/4 ∧ l.intercept = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l418_41834


namespace NUMINAMATH_CALUDE_factorization_equality_l418_41865

theorem factorization_equality (x y : ℝ) : 
  x^2 + 4*y^2 - 4*x*y - 1 = (x - 2*y + 1)*(x - 2*y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l418_41865


namespace NUMINAMATH_CALUDE_two_numbers_with_specific_means_l418_41857

theorem two_numbers_with_specific_means :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = 2 * Real.sqrt 3 ∧
  (a + b) / 2 = 6 ∧
  2 / (1 / a + 1 / b) = 2 ∧
  ((a = 6 - 2 * Real.sqrt 6 ∧ b = 6 + 2 * Real.sqrt 6) ∨
   (a = 6 + 2 * Real.sqrt 6 ∧ b = 6 - 2 * Real.sqrt 6)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_specific_means_l418_41857


namespace NUMINAMATH_CALUDE_train_length_is_160_meters_l418_41822

def train_speed : ℝ := 45 -- km/hr
def crossing_time : ℝ := 30 -- seconds
def bridge_length : ℝ := 215 -- meters

theorem train_length_is_160_meters :
  let speed_mps := train_speed * 1000 / 3600
  let total_distance := speed_mps * crossing_time
  total_distance - bridge_length = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_is_160_meters_l418_41822


namespace NUMINAMATH_CALUDE_parallel_lines_solution_l418_41874

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∀ x y : ℝ, m₁ * x - y = 0 ↔ m₂ * x - y = 0) ↔ m₁ = m₂

/-- Given two parallel lines ax - y + a = 0 and (2a - 3)x + ay - a = 0, prove that a = -3 -/
theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x - y + a = 0 ↔ (2 * a - 3) * x + a * y - a = 0) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_solution_l418_41874


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l418_41831

/-- The line equation passing through a fixed point -/
def line_equation (a x y : ℝ) : Prop :=
  a * y = (3 * a - 1) * x - 1

/-- Theorem stating that the line passes through (-1, -3) for all a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), line_equation a (-1) (-3) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l418_41831


namespace NUMINAMATH_CALUDE_product_18396_9999_l418_41819

theorem product_18396_9999 : 18396 * 9999 = 183962604 := by sorry

end NUMINAMATH_CALUDE_product_18396_9999_l418_41819


namespace NUMINAMATH_CALUDE_expression_equals_100_l418_41878

theorem expression_equals_100 : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_100_l418_41878


namespace NUMINAMATH_CALUDE_symmetric_complex_quotient_l418_41851

/-- Two complex numbers are symmetric about the y-axis if their real parts are negatives of each other and their imaginary parts are equal -/
def symmetric_about_y_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem symmetric_complex_quotient (z₁ z₂ : ℂ) :
  symmetric_about_y_axis z₁ z₂ → z₁ = 1 + I → z₂ / z₁ = I :=
by
  sorry

#check symmetric_complex_quotient

end NUMINAMATH_CALUDE_symmetric_complex_quotient_l418_41851


namespace NUMINAMATH_CALUDE_log_stack_sum_l418_41827

/-- The sum of an arithmetic sequence with first term a, last term l, and n terms -/
def arithmetic_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The number of terms in the sequence of logs -/
def num_terms : ℕ := 15 - 5 + 1

theorem log_stack_sum :
  arithmetic_sum 5 15 num_terms = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l418_41827


namespace NUMINAMATH_CALUDE_three_digit_reverse_double_l418_41899

theorem three_digit_reverse_double (g : ℕ) (a b c : ℕ) : 
  (0 < g) → 
  (a < g) → (b < g) → (c < g) →
  (a * g^2 + b * g + c = 2 * (c * g^2 + b * g + a)) →
  ∃ k : ℕ, (k > 0) ∧ (g = 3 * k + 2) := by
sorry


end NUMINAMATH_CALUDE_three_digit_reverse_double_l418_41899


namespace NUMINAMATH_CALUDE_polygon_area_l418_41801

structure Polygon :=
  (sides : ℕ)
  (side_length : ℝ)
  (perimeter : ℝ)
  (is_rectangular_with_removed_corners : Prop)

def area_of_polygon (p : Polygon) : ℝ :=
  20 * p.side_length^2

theorem polygon_area (p : Polygon) 
  (h1 : p.sides = 20)
  (h2 : p.perimeter = 60)
  (h3 : p.is_rectangular_with_removed_corners)
  (h4 : p.side_length = p.perimeter / p.sides) :
  area_of_polygon p = 180 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_l418_41801


namespace NUMINAMATH_CALUDE_quadratic_minimum_l418_41835

/-- The quadratic function f(x) = 3x^2 - 8x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 7

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x - 8

theorem quadratic_minimum :
  ∃ (x_min : ℝ), x_min = 4/3 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_minimum_l418_41835


namespace NUMINAMATH_CALUDE_complementary_event_l418_41862

/-- The sample space of outcomes when two students purchase a beverage with a chance of winning a prize -/
inductive Outcome
  | BothWin
  | AWinsBLoses
  | ALosesBWins
  | BothLose

/-- The event where both students win a prize -/
def bothWin (o : Outcome) : Prop :=
  o = Outcome.BothWin

/-- The event where at most one student wins a prize -/
def atMostOneWins (o : Outcome) : Prop :=
  o = Outcome.AWinsBLoses ∨ o = Outcome.ALosesBWins ∨ o = Outcome.BothLose

/-- Theorem stating that the complementary event to "both win" is "at most one wins" -/
theorem complementary_event :
  ∀ o : Outcome, ¬(bothWin o) ↔ atMostOneWins o :=
sorry


end NUMINAMATH_CALUDE_complementary_event_l418_41862


namespace NUMINAMATH_CALUDE_final_digit_is_nine_l418_41891

/-- Represents a sequence of digits -/
def DigitSequence := List Nat

/-- Constructs the initial sequence N₁ by concatenating numbers from 1 to 1995 -/
def constructN1 : DigitSequence := sorry

/-- Removes digits at even positions from a given sequence -/
def removeEvenPositions (seq : DigitSequence) : DigitSequence := sorry

/-- Removes digits at odd positions from a given sequence -/
def removeOddPositions (seq : DigitSequence) : DigitSequence := sorry

/-- Applies the alternating removal process until only one digit remains -/
def processSequence (seq : DigitSequence) : Nat := sorry

theorem final_digit_is_nine :
  processSequence constructN1 = 9 := by sorry

end NUMINAMATH_CALUDE_final_digit_is_nine_l418_41891


namespace NUMINAMATH_CALUDE_power_of_fraction_three_fourths_five_l418_41846

theorem power_of_fraction_three_fourths_five :
  (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_three_fourths_five_l418_41846


namespace NUMINAMATH_CALUDE_positive_solution_x_l418_41840

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 10 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (x_pos : x > 0) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l418_41840


namespace NUMINAMATH_CALUDE_total_cost_of_fruits_l418_41887

def cost_of_grapes : ℚ := 12.08
def cost_of_cherries : ℚ := 9.85

theorem total_cost_of_fruits : cost_of_grapes + cost_of_cherries = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_fruits_l418_41887


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l418_41804

/-- The focal length of the hyperbola -/
def focal_length : ℝ := 4

/-- The equation of circle C₂ -/
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- The equation of hyperbola C₁ -/
def hyperbola_equation (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote_equation (a b x y : ℝ) : Prop := b * x = a * y ∨ b * x = -a * y

/-- The asymptotes are tangent to the circle -/
def asymptotes_tangent_to_circle (a b : ℝ) : Prop :=
  ∀ x y, asymptote_equation a b x y → (abs (-2 * b) / Real.sqrt (a^2 + b^2) = 1)

theorem hyperbola_equation_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : a^2 - b^2 = focal_length^2 / 4)
  (h_tangent : asymptotes_tangent_to_circle a b) :
  ∀ x y, hyperbola_equation a b x y ↔ x^2 / 3 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l418_41804


namespace NUMINAMATH_CALUDE_power_four_congruence_l418_41805

theorem power_four_congruence (n : ℕ) (a : ℤ) (hn : n > 0) (ha : a^3 ≡ 1 [ZMOD n]) :
  a^4 ≡ a [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_power_four_congruence_l418_41805


namespace NUMINAMATH_CALUDE_f_negative_six_equals_negative_one_l418_41832

def f (x : ℝ) : ℝ := sorry

theorem f_negative_six_equals_negative_one :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, f (x + 6) = f x) →  -- f has period 6
  (∀ x, -3 ≤ x ∧ x ≤ 3 → f x = (x + 1) * (x - 1)) →  -- f(x) = (x+1)(x-a) for -3 ≤ x ≤ 3, where a = 1
  f (-6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_six_equals_negative_one_l418_41832


namespace NUMINAMATH_CALUDE_card_game_problem_l418_41807

/-- The card game problem -/
theorem card_game_problem (T : ℚ) :
  -- Initial ratios
  let initial_aldo : ℚ := 7 / 18 * T
  let initial_bernardo : ℚ := 6 / 18 * T
  let initial_carlos : ℚ := 5 / 18 * T
  -- Final ratios
  let final_aldo : ℚ := 6 / 15 * T
  let final_bernardo : ℚ := 5 / 15 * T
  let final_carlos : ℚ := 4 / 15 * T
  -- One player won 12 reais
  (∃ (winner : ℚ), winner - (winner - 12) = 12) →
  -- The changes in amounts
  (final_aldo - initial_aldo = 12 ∨
   final_bernardo - initial_bernardo = 12 ∨
   final_carlos - initial_carlos = 12) →
  -- Prove the final amounts
  (final_aldo = 432 ∧ final_bernardo = 360 ∧ final_carlos = 288) := by
sorry


end NUMINAMATH_CALUDE_card_game_problem_l418_41807


namespace NUMINAMATH_CALUDE_angle_c_is_60_degrees_l418_41883

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the concept of an angle in a quadrilateral
def angle (q : Quadrilateral) (v : Fin 4) : ℝ := sorry

-- State the theorem
theorem angle_c_is_60_degrees (q : Quadrilateral) :
  angle q 0 + 60 = angle q 1 →  -- Angle A is 60° smaller than angle B
  angle q 2 = 60 := by  -- Angle C is 60°
  sorry

end NUMINAMATH_CALUDE_angle_c_is_60_degrees_l418_41883


namespace NUMINAMATH_CALUDE_marbles_left_l418_41879

theorem marbles_left (initial_marbles given_marbles : ℝ) :
  initial_marbles = 9.0 →
  given_marbles = 3.0 →
  initial_marbles - given_marbles = 6.0 := by
sorry

end NUMINAMATH_CALUDE_marbles_left_l418_41879


namespace NUMINAMATH_CALUDE_egg_problem_solution_l418_41856

/-- Represents the number of eggs of each type --/
structure EggCounts where
  newLaid : ℕ
  fresh : ℕ
  ordinary : ℕ

/-- Checks if the given egg counts satisfy all problem constraints --/
def satisfiesConstraints (counts : EggCounts) : Prop :=
  counts.newLaid + counts.fresh + counts.ordinary = 100 ∧
  5 * counts.newLaid + counts.fresh + (counts.ordinary / 2) = 100 ∧
  (counts.newLaid = counts.fresh ∨ counts.newLaid = counts.ordinary ∨ counts.fresh = counts.ordinary)

/-- The unique solution to the egg problem --/
def eggSolution : EggCounts :=
  { newLaid := 10, fresh := 10, ordinary := 80 }

/-- Theorem stating that the egg solution is unique and satisfies all constraints --/
theorem egg_problem_solution :
  satisfiesConstraints eggSolution ∧
  ∀ counts : EggCounts, satisfiesConstraints counts → counts = eggSolution := by
  sorry


end NUMINAMATH_CALUDE_egg_problem_solution_l418_41856


namespace NUMINAMATH_CALUDE_skateboarder_distance_is_3720_l418_41848

/-- Represents the skateboarder's journey -/
structure SkateboarderJourney where
  initial_distance : ℕ  -- Distance covered in the first second
  distance_increase : ℕ  -- Increase in distance each second on the ramp
  ramp_time : ℕ  -- Time spent on the ramp
  flat_time : ℕ  -- Time spent on the flat stretch

/-- Calculates the total distance traveled by the skateboarder -/
def total_distance (journey : SkateboarderJourney) : ℕ :=
  let ramp_distance := journey.ramp_time * (journey.initial_distance + (journey.ramp_time - 1) * journey.distance_increase / 2)
  let final_speed := journey.initial_distance + (journey.ramp_time - 1) * journey.distance_increase
  let flat_distance := final_speed * journey.flat_time
  ramp_distance + flat_distance

/-- Theorem stating that the total distance traveled is 3720 meters -/
theorem skateboarder_distance_is_3720 (journey : SkateboarderJourney) 
  (h1 : journey.initial_distance = 10)
  (h2 : journey.distance_increase = 9)
  (h3 : journey.ramp_time = 20)
  (h4 : journey.flat_time = 10) : 
  total_distance journey = 3720 := by
  sorry

end NUMINAMATH_CALUDE_skateboarder_distance_is_3720_l418_41848


namespace NUMINAMATH_CALUDE_find_k_l418_41898

/-- The sum of the first n terms of the sequence {a_n} -/
def S (n : ℕ) (k : ℝ) : ℝ := 5 * n^2 + k * n

/-- The nth term of the sequence {a_n} -/
def a (n : ℕ) (k : ℝ) : ℝ := S n k - S (n-1) k

theorem find_k : ∃ k : ℝ, (∀ n : ℕ, S n k = 5 * n^2 + k * n) ∧ a 2 k = 18 → k = 3 :=
sorry

end NUMINAMATH_CALUDE_find_k_l418_41898


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l418_41861

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ 
  (∀ (z : ℚ), x ≠ (z : ℝ)) ∧
  (∀ (a b : ℕ), x ≠ Real.sqrt (a / b))

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 3) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 9) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.1) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l418_41861


namespace NUMINAMATH_CALUDE_field_trip_adults_l418_41838

/-- Given a field trip scenario, prove the number of adults attending. -/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 9 → num_students = 40 → num_vans = 6 → 
  (num_vans * van_capacity - num_students : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l418_41838


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l418_41817

/-- The ratio of a boy's faster walking rate to his usual walking rate, given his usual time and early arrival time. -/
theorem walking_rate_ratio (usual_time early_time : ℕ) : 
  usual_time = 42 → early_time = 6 → (usual_time : ℚ) / (usual_time - early_time) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l418_41817


namespace NUMINAMATH_CALUDE_expression_evaluation_l418_41867

theorem expression_evaluation :
  let x : ℤ := -1
  (x + 1) * (x - 2) + 2 * (x + 4) * (x - 4) = -30 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l418_41867


namespace NUMINAMATH_CALUDE_soap_cost_per_pound_l418_41842

theorem soap_cost_per_pound 
  (num_bars : ℕ) 
  (weight_per_bar : ℝ) 
  (total_cost : ℝ) 
  (h1 : num_bars = 20)
  (h2 : weight_per_bar = 1.5)
  (h3 : total_cost = 15) : 
  total_cost / (num_bars * weight_per_bar) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_soap_cost_per_pound_l418_41842


namespace NUMINAMATH_CALUDE_pastries_sold_equals_initial_l418_41810

/-- Represents the number of pastries and cakes made and sold by a baker -/
structure BakerInventory where
  initialPastries : ℕ
  initialCakes : ℕ
  cakesSold : ℕ
  cakesRemaining : ℕ

/-- Theorem stating that the number of pastries sold is equal to the initial number of pastries made -/
theorem pastries_sold_equals_initial (inventory : BakerInventory)
  (h1 : inventory.initialPastries = 61)
  (h2 : inventory.initialCakes = 167)
  (h3 : inventory.cakesSold = 108)
  (h4 : inventory.cakesRemaining = 59)
  : inventory.initialPastries = inventory.initialPastries := by
  sorry

end NUMINAMATH_CALUDE_pastries_sold_equals_initial_l418_41810


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l418_41863

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8) (h2 : x - y = 3/8) : x^2 - y^2 = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l418_41863


namespace NUMINAMATH_CALUDE_floor_of_e_eq_two_l418_41894

/-- The floor of Euler's number is 2 -/
theorem floor_of_e_eq_two : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_eq_two_l418_41894


namespace NUMINAMATH_CALUDE_complex_exponent_calculation_l418_41877

theorem complex_exponent_calculation : 3 * 3^6 - 9^60 / 9^58 + 4^3 = 2170 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponent_calculation_l418_41877


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l418_41828

/-- Given the selling price and profit per meter of cloth, calculate the cost price per meter. -/
theorem cost_price_per_meter
  (selling_price : ℚ)
  (cloth_length : ℚ)
  (profit_per_meter : ℚ)
  (h1 : selling_price = 8925)
  (h2 : cloth_length = 85)
  (h3 : profit_per_meter = 25) :
  (selling_price - cloth_length * profit_per_meter) / cloth_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l418_41828


namespace NUMINAMATH_CALUDE_circle_plus_five_two_l418_41821

/-- The custom binary operation ⊕ -/
def circle_plus (x y : ℝ) : ℝ := (x + y + 1) * (x - y)

/-- Theorem stating that 5 ⊕ 2 = 24 -/
theorem circle_plus_five_two : circle_plus 5 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_five_two_l418_41821


namespace NUMINAMATH_CALUDE_test_coincidences_l418_41824

theorem test_coincidences (n : ℕ) (p_vasya p_misha : ℝ) 
  (hn : n = 20) 
  (hv : p_vasya = 6 / 20) 
  (hm : p_misha = 8 / 20) : 
  n * (p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)) = 10.8 := by
  sorry

end NUMINAMATH_CALUDE_test_coincidences_l418_41824


namespace NUMINAMATH_CALUDE_min_distance_complex_circles_l418_41847

theorem min_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4 * Complex.I)) = 2)
  (hw : Complex.abs (w - (6 - 5 * Complex.I)) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ z' w' : ℂ, 
      Complex.abs (z' - (2 - 4 * Complex.I)) = 2 → 
      Complex.abs (w' - (6 - 5 * Complex.I)) = 4 → 
      Complex.abs (z' - w') ≥ min_dist) ∧
    Complex.abs (z - w) ≥ min_dist ∧
    min_dist = Real.sqrt 17 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_circles_l418_41847


namespace NUMINAMATH_CALUDE_lea_notebooks_l418_41815

/-- The number of notebooks Léa bought -/
def notebooks : ℕ := sorry

/-- The cost of the book Léa bought -/
def book_cost : ℕ := 16

/-- The number of binders Léa bought -/
def num_binders : ℕ := 3

/-- The cost of each binder -/
def binder_cost : ℕ := 2

/-- The cost of each notebook -/
def notebook_cost : ℕ := 1

/-- The total cost of Léa's purchases -/
def total_cost : ℕ := 28

theorem lea_notebooks : 
  notebooks = 6 ∧
  book_cost + num_binders * binder_cost + notebooks * notebook_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_lea_notebooks_l418_41815


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_when_not_p_necessary_not_sufficient_for_not_q_l418_41860

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x - 72 ≤ 0 ∧ x^2 + x - 6 > 0

-- Part 1
theorem range_of_x_when_a_is_neg_one :
  (∀ x : ℝ, p x (-1) → q x) →
  ∀ x : ℝ, (x ∈ Set.Icc (-6) (-3) ∪ Set.Ioc 1 12) ↔ (p x (-1) ∨ q x) :=
sorry

-- Part 2
theorem range_of_a_when_not_p_necessary_not_sufficient_for_not_q :
  (∀ x a : ℝ, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x a : ℝ, ¬(p x a) ∧ q x) →
  ∀ a : ℝ, a ∈ Set.Icc (-4) (-2) ↔ (∃ x : ℝ, p x a ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_when_not_p_necessary_not_sufficient_for_not_q_l418_41860


namespace NUMINAMATH_CALUDE_specific_parallelepiped_volume_l418_41852

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  /-- Length of one side of the base -/
  sideA : ℝ
  /-- Length of the other side of the base -/
  sideB : ℝ
  /-- Angle between the sides of the base in radians -/
  baseAngle : ℝ
  /-- The smaller diagonal of the parallelepiped -/
  smallerDiagonal : ℝ

/-- The volume of the right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific parallelepiped -/
theorem specific_parallelepiped_volume :
  ∃ (p : RightParallelepiped),
    p.sideA = 3 ∧
    p.sideB = 4 ∧
    p.baseAngle = 2 * π / 3 ∧
    p.smallerDiagonal = Real.sqrt (p.sideA ^ 2 + p.sideB ^ 2 - 2 * p.sideA * p.sideB * Real.cos p.baseAngle) ∧
    volume p = 36 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_specific_parallelepiped_volume_l418_41852


namespace NUMINAMATH_CALUDE_train_tunnel_time_l418_41850

/-- Proves that a train of given length and speed passing through a tunnel of given length takes 1 minute to completely clear the tunnel. -/
theorem train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length_km : ℝ) :
  train_length = 100 →
  train_speed_kmh = 72 →
  tunnel_length_km = 1.1 →
  (tunnel_length_km * 1000 + train_length) / (train_speed_kmh * 1000 / 60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_train_tunnel_time_l418_41850


namespace NUMINAMATH_CALUDE_nadine_dog_cleaning_time_l418_41896

/-- The time Nadine spends cleaning her dog -/
def dog_cleaning_time (hosing_time shampoo_time shampoo_count : ℕ) : ℕ :=
  hosing_time + shampoo_time * shampoo_count

theorem nadine_dog_cleaning_time :
  dog_cleaning_time 10 15 3 = 55 :=
by sorry

end NUMINAMATH_CALUDE_nadine_dog_cleaning_time_l418_41896


namespace NUMINAMATH_CALUDE_calculation_proof_l418_41873

theorem calculation_proof :
  ((-48) * 0.125 + 48 * (11/8) + (-48) * (5/4) = 0) ∧
  (|(-7/9)| / ((2/3) - (1/5)) - (1/3) * ((-4)^2) = -11/3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l418_41873


namespace NUMINAMATH_CALUDE_other_jelly_correct_l418_41814

/-- Given a total amount of jelly and the amount of one type, 
    calculate the amount of the other type -/
def other_jelly_amount (total : ℕ) (one_type : ℕ) : ℕ :=
  total - one_type

/-- Theorem: The amount of the other type of jelly is the difference
    between the total amount and the amount of one type -/
theorem other_jelly_correct (total : ℕ) (one_type : ℕ) 
  (h : one_type ≤ total) : 
  other_jelly_amount total one_type = total - one_type :=
by
  sorry

#eval other_jelly_amount 6310 4518

end NUMINAMATH_CALUDE_other_jelly_correct_l418_41814


namespace NUMINAMATH_CALUDE_alpha_epsilon_shorter_l418_41855

/-- The running times of three movies in minutes -/
structure MovieTimes where
  millennium : ℕ
  alphaEpsilon : ℕ
  beastOfWar : ℕ

/-- Conditions for the movie running times -/
def validMovieTimes (t : MovieTimes) : Prop :=
  t.millennium = 120 ∧
  t.beastOfWar = 100 ∧
  t.beastOfWar = t.alphaEpsilon + 10 ∧
  t.alphaEpsilon < t.millennium

theorem alpha_epsilon_shorter (t : MovieTimes) (h : validMovieTimes t) :
  t.millennium - t.alphaEpsilon = 30 := by
  sorry

#check alpha_epsilon_shorter

end NUMINAMATH_CALUDE_alpha_epsilon_shorter_l418_41855


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l418_41808

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for part I
theorem range_of_a : 
  {a : ℝ | ∃ x, f x < |1 - 2*a|} = {a : ℝ | a < -3/2 ∨ a > 5/2} :=
sorry

-- Theorem for part II
theorem range_of_m :
  {m : ℝ | ∃ t, t^2 - 2*Real.sqrt 6*t + f m = 0} = {m : ℝ | -1 ≤ m ∧ m ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l418_41808


namespace NUMINAMATH_CALUDE_complex_equation_solution_l418_41895

theorem complex_equation_solution (z : ℂ) (h : Complex.I * (z + 1) = 1 + 2 * Complex.I) :
  z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l418_41895


namespace NUMINAMATH_CALUDE_infinite_solutions_l418_41871

/-- Standard prime factorization of a positive integer -/
def prime_factorization (n : ℕ+) : List (ℕ × ℕ) := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := 
  let factors := prime_factorization n
  (factors.map (fun (p, α) => α)).prod * 
  (factors.map (fun (p, α) => p^(α - 1))).prod

/-- The set of positive integers n satisfying f(n+1) = f(n) + 1 -/
def S : Set ℕ+ := {n | f (n + 1) = f n + 1}

/-- The main theorem to be proved -/
theorem infinite_solutions : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l418_41871


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l418_41866

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_pos : ∀ n, a n > 0)
  (h_mean : Real.sqrt (a 4 * a 14) = 2 * Real.sqrt 2) :
  (2 * a 7 + a 11 ≥ 8) ∧ ∃ x, 2 * x + (a 11) = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l418_41866


namespace NUMINAMATH_CALUDE_divisibility_problem_l418_41872

theorem divisibility_problem (m n : ℕ) (h : m * n ∣ m + n) :
  (Nat.Prime n → n ∣ m) ∧
  (∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q → ¬(n ∣ m)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l418_41872


namespace NUMINAMATH_CALUDE_min_k_for_inequality_l418_41839

theorem min_k_for_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ k : ℝ, ∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (5 * x + y)) ↔
  (∃ k : ℝ, k ≥ Real.sqrt 30 / 5 ∧ 
    ∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (5 * x + y)) :=
by sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l418_41839


namespace NUMINAMATH_CALUDE_julia_total_food_expense_l418_41876

/-- Represents the weekly food cost and number of weeks for an animal -/
structure AnimalExpense where
  weeklyFoodCost : ℕ
  numberOfWeeks : ℕ

/-- Calculates the total food expense for Julia's animals -/
def totalFoodExpense (animals : List AnimalExpense) : ℕ :=
  animals.map (fun a => a.weeklyFoodCost * a.numberOfWeeks) |>.sum

/-- The list of Julia's animals with their expenses -/
def juliaAnimals : List AnimalExpense := [
  ⟨15, 3⟩,  -- Parrot
  ⟨12, 5⟩,  -- Rabbit
  ⟨8, 2⟩,   -- Turtle
  ⟨5, 6⟩    -- Guinea pig
]

/-- Theorem stating that Julia's total food expense is $151 -/
theorem julia_total_food_expense :
  totalFoodExpense juliaAnimals = 151 := by
  sorry

end NUMINAMATH_CALUDE_julia_total_food_expense_l418_41876


namespace NUMINAMATH_CALUDE_sum_of_leading_digits_is_14_l418_41889

/-- M is a 400-digit positive integer where each digit is 8 -/
def M : ℕ := sorry

/-- g(r) is the leading digit of the r-th root of M -/
def g (r : ℕ) : ℕ := sorry

/-- The sum of g(r) for r from 2 to 8 is 14 -/
theorem sum_of_leading_digits_is_14 : 
  g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_leading_digits_is_14_l418_41889


namespace NUMINAMATH_CALUDE_opposite_pairs_l418_41892

theorem opposite_pairs :
  (- (-2) = - (- (-2))) ∧
  ((-1)^2 = - ((-1)^2)) ∧
  ((-2)^3 ≠ -6) ∧
  ((-2)^7 = -2^7) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l418_41892


namespace NUMINAMATH_CALUDE_intersection_implies_p_value_l418_41809

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define the parabola equation
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- State the theorem
theorem intersection_implies_p_value 
  (p : ℝ) 
  (h_p_pos : p > 0) 
  (A B : ℝ × ℝ) 
  (h_A_ellipse : ellipse A.1 A.2)
  (h_A_parabola : parabola p A.1 A.2)
  (h_B_ellipse : ellipse B.1 B.2)
  (h_B_parabola : parabola p B.1 B.2)
  (h_distance : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4) :
  p = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_p_value_l418_41809


namespace NUMINAMATH_CALUDE_append_digit_square_difference_l418_41812

theorem append_digit_square_difference (x y : ℕ) : 
  x > 0 → y ≤ 9 → (10 * x + y - x^2 = 8 * x) → 
  ((x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8)) := by
  sorry

end NUMINAMATH_CALUDE_append_digit_square_difference_l418_41812


namespace NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l418_41823

theorem cos_alpha_plus_seven_pi_twelfths (α : ℝ) 
  (h : Real.sin (α + π / 12) = 1 / 3) : 
  Real.cos (α + 7 * π / 12) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l418_41823


namespace NUMINAMATH_CALUDE_square_sum_from_means_l418_41875

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20) 
  (h_geometric : Real.sqrt (a * b) = 16) : 
  a^2 + b^2 = 1088 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l418_41875


namespace NUMINAMATH_CALUDE_solution_set_t_3_nonnegative_for_all_x_l418_41880

-- Define the function f
def f (t x : ℝ) : ℝ := x^2 - (t + 1)*x + t

-- Theorem 1: Solution set when t = 3
theorem solution_set_t_3 :
  {x : ℝ | f 3 x > 0} = Set.Iio 1 ∪ Set.Ioi 3 :=
sorry

-- Theorem 2: Condition for f(x) ≥ 0 for all real x
theorem nonnegative_for_all_x :
  (∀ x : ℝ, f t x ≥ 0) ↔ t = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_t_3_nonnegative_for_all_x_l418_41880


namespace NUMINAMATH_CALUDE_method_doubles_method_power_of_two_l418_41844

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a row of coins -/
def CoinRow (N : ℕ) := Fin N → CoinState

/-- Represents a method for the magician to guess the number -/
structure GuessMethod (N : ℕ) :=
(guess : CoinRow N → Fin N)

/-- States that if a method exists for N coins, it exists for 2N coins -/
theorem method_doubles {N : ℕ} (h : GuessMethod N) : GuessMethod (2 * N) :=
sorry

/-- States that the method only works for powers of 2 -/
theorem method_power_of_two {N : ℕ} : GuessMethod N → ∃ k : ℕ, N = 2^k :=
sorry

end NUMINAMATH_CALUDE_method_doubles_method_power_of_two_l418_41844


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_over_six_l418_41885

noncomputable def f (x : ℝ) : ℝ := (1/2) * x - 2 * Real.cos x

theorem tangent_slope_at_pi_over_six :
  deriv f (π/6) = 3/2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_over_six_l418_41885


namespace NUMINAMATH_CALUDE_dans_limes_l418_41829

theorem dans_limes (initial_limes : ℝ) (given_limes : ℝ) (remaining_limes : ℝ) : 
  initial_limes = 9 → given_limes = 4.5 → remaining_limes = initial_limes - given_limes → remaining_limes = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_dans_limes_l418_41829


namespace NUMINAMATH_CALUDE_intersection_M_N_l418_41849

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

theorem intersection_M_N : M ∩ N = {y : ℝ | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l418_41849


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l418_41884

-- Define the polynomials
def p (z : ℝ) : ℝ := 3 * z^2 - 4 * z + 1
def q (z : ℝ) : ℝ := 4 * z^3 + z^2 - 5 * z + 3

-- State the theorem
theorem polynomial_multiplication :
  ∀ z : ℝ, p z * q z = 12 * z^5 + 3 * z^4 + 32 * z^3 + z^2 - 7 * z + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l418_41884


namespace NUMINAMATH_CALUDE_marta_textbook_problem_l418_41888

theorem marta_textbook_problem :
  ∀ (sale_books online_books bookstore_books : ℕ)
    (sale_price online_total bookstore_total total_spent : ℚ),
    sale_books = 5 →
    sale_price = 10 →
    online_books = 2 →
    online_total = 40 →
    bookstore_total = 3 * online_total →
    total_spent = 210 →
    sale_books * sale_price + online_total + bookstore_total = total_spent →
    bookstore_books = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_marta_textbook_problem_l418_41888


namespace NUMINAMATH_CALUDE_tournament_rankings_l418_41893

/-- Represents a team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match :=
(team1 : Team)
(team2 : Team)

/-- Represents the tournament structure -/
structure Tournament :=
(saturday_matches : List Match)
(no_ties : Bool)

/-- Represents the final ranking of teams -/
structure Ranking :=
(first : Team)
(second : Team)
(third : Team)
(fourth : Team)
(fifth : Team)
(sixth : Team)

/-- Counts the number of possible ranking sequences for the given tournament -/
def countPossibleRankings (t : Tournament) : Nat :=
  sorry

/-- The main theorem stating the number of possible ranking sequences -/
theorem tournament_rankings (t : Tournament) 
  (h1 : t.saturday_matches = [Match.mk Team.A Team.B, Match.mk Team.C Team.D, Match.mk Team.E Team.F])
  (h2 : t.no_ties = true) : 
  countPossibleRankings t = 288 :=
sorry

end NUMINAMATH_CALUDE_tournament_rankings_l418_41893
