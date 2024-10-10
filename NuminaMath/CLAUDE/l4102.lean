import Mathlib

namespace vacation_days_l4102_410270

theorem vacation_days (rainy_days clear_mornings clear_afternoons : ℕ) 
  (h1 : rainy_days = 13)
  (h2 : clear_mornings = 11)
  (h3 : clear_afternoons = 12)
  (h4 : ∀ d, (d ≤ rainy_days ↔ (d ≤ rainy_days - clear_mornings ∨ d ≤ rainy_days - clear_afternoons) ∧
                               ¬(d ≤ rainy_days - clear_mornings ∧ d ≤ rainy_days - clear_afternoons))) :
  rainy_days + clear_mornings = 18 :=
by sorry

end vacation_days_l4102_410270


namespace greatest_3digit_base8_divisible_by_7_l4102_410281

/-- Converts a base 8 number to base 10 --/
def base8To10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number --/
def is3DigitBase8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : Nat, is3DigitBase8 n → base8To10 n % 7 = 0 → n ≤ 777 :=
sorry

end greatest_3digit_base8_divisible_by_7_l4102_410281


namespace unique_xxyy_square_l4102_410229

def is_xxyy_form (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = 1000 * x + 100 * x + 10 * y + y

theorem unique_xxyy_square : 
  ∀ n : ℕ, is_xxyy_form n ∧ ∃ m : ℕ, n = m^2 → n = 7744 :=
sorry

end unique_xxyy_square_l4102_410229


namespace opposite_seats_imply_38_seats_l4102_410294

/-- Represents a round table with equally spaced seats -/
structure RoundTable where
  total_seats : ℕ
  seats_numbered_clockwise : Bool

/-- Defines two people sitting opposite each other on a round table -/
structure OppositeSeats (table : RoundTable) where
  seat1 : ℕ
  seat2 : ℕ
  are_opposite : seat2 - seat1 = table.total_seats / 2

/-- Theorem stating that if two people sit in seats 10 and 29 opposite each other,
    then the total number of seats is 38 -/
theorem opposite_seats_imply_38_seats (table : RoundTable)
  (opposite_pair : OppositeSeats table)
  (h1 : opposite_pair.seat1 = 10)
  (h2 : opposite_pair.seat2 = 29)
  (h3 : table.seats_numbered_clockwise = true) :
  table.total_seats = 38 := by
  sorry

end opposite_seats_imply_38_seats_l4102_410294


namespace jake_present_weight_l4102_410200

/-- Jake's present weight in pounds -/
def jake_weight : ℕ := sorry

/-- Jake's sister's weight in pounds -/
def sister_weight : ℕ := sorry

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℕ := 224

theorem jake_present_weight : 
  (jake_weight - 20 = 2 * sister_weight) ∧ 
  (jake_weight + sister_weight = combined_weight) → 
  jake_weight = 156 := by
sorry

end jake_present_weight_l4102_410200


namespace f_is_quadratic_l4102_410277

/-- Definition of a quadratic equation in x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 3x -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f :=
sorry

end f_is_quadratic_l4102_410277


namespace inscribed_circle_segment_ratios_l4102_410216

/-- Given a triangle with sides in ratio 5:4:3, prove the ratios of segments divided by tangent points of inscribed circle -/
theorem inscribed_circle_segment_ratios (a b c : ℝ) (h : a / b = 5 / 4 ∧ b / c = 4 / 3) :
  let r := (a + b - c) / 2
  let s := (a + b + c) / 2
  (r / (s - b), r / (s - c), (s - c) / (s - b)) = (1 / 3, 1 / 2, 2 / 3) := by
  sorry


end inscribed_circle_segment_ratios_l4102_410216


namespace line_through_points_l4102_410264

theorem line_through_points (a n : ℝ) :
  (∀ x y, x = 3 * y + 5 → 
    ((x = a ∧ y = n) ∨ (x = a + 2 ∧ y = n + 2/3))) →
  a = 3 * n + 5 :=
by sorry

end line_through_points_l4102_410264


namespace johns_remaining_money_l4102_410214

def remaining_money (initial_amount : ℚ) (snack_fraction : ℚ) (necessity_fraction : ℚ) : ℚ :=
  let after_snacks := initial_amount * (1 - snack_fraction)
  after_snacks * (1 - necessity_fraction)

theorem johns_remaining_money :
  remaining_money 20 (1/5) (3/4) = 4 := by
  sorry

end johns_remaining_money_l4102_410214


namespace fraction_sum_equality_l4102_410244

theorem fraction_sum_equality : 
  (1/4 - 1/5) / (2/5 - 1/4) + (1/6) / (1/3 - 1/4) = 7/3 := by
  sorry

end fraction_sum_equality_l4102_410244


namespace ellipse_min_distance_sum_l4102_410231

theorem ellipse_min_distance_sum (x y : ℝ) : 
  (x^2 / 2 + y^2 = 1) →  -- Point (x, y) is on the ellipse
  (∃ (min : ℝ), (∀ (x' y' : ℝ), x'^2 / 2 + y'^2 = 1 → 
    (x'^2 + y'^2) + ((x' + 1)^2 + y'^2) ≥ min) ∧ 
    min = 2) := by
  sorry

end ellipse_min_distance_sum_l4102_410231


namespace factorial_sum_equals_power_of_two_l4102_410203

theorem factorial_sum_equals_power_of_two (a b c : ℕ+) : 
  (Nat.factorial a.val + Nat.factorial b.val = 2^(Nat.factorial c.val)) ↔ 
  ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 2)) := by
sorry

end factorial_sum_equals_power_of_two_l4102_410203


namespace divisible_by_two_and_three_implies_divisible_by_six_l4102_410255

theorem divisible_by_two_and_three_implies_divisible_by_six (n : ℕ) :
  (n % 2 = 0 ∧ n % 3 = 0) → n % 6 = 0 := by
  sorry

end divisible_by_two_and_three_implies_divisible_by_six_l4102_410255


namespace janous_inequality_l4102_410238

theorem janous_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5 ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end janous_inequality_l4102_410238


namespace no_primes_in_range_l4102_410271

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ p, Prime p → ¬(n! + 2 < p ∧ p < n! + n + 1) :=
sorry

end no_primes_in_range_l4102_410271


namespace matrix_equation_proof_l4102_410272

open Matrix

theorem matrix_equation_proof :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  M^3 - 3 • M^2 + 4 • M = !![6, 12; 3, 6] := by sorry

end matrix_equation_proof_l4102_410272


namespace attic_items_count_l4102_410292

theorem attic_items_count (total : ℝ) (useful_percent : ℝ) (heirloom_percent : ℝ) (junk_percent : ℝ) (junk_count : ℝ) :
  useful_percent = 0.20 →
  heirloom_percent = 0.10 →
  junk_percent = 0.70 →
  junk_count = 28 →
  junk_percent * total = junk_count →
  useful_percent * total = 8 :=
by
  sorry

end attic_items_count_l4102_410292


namespace sum_of_fraction_and_constant_l4102_410254

theorem sum_of_fraction_and_constant (x : Real) (h : x = 8.0) : 0.75 * x + 2 = 8.0 := by
  sorry

end sum_of_fraction_and_constant_l4102_410254


namespace equal_area_floors_width_l4102_410208

/-- Represents the dimensions of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the area of a rectangular floor -/
def area (f : Floor) : ℝ := f.length * f.width

theorem equal_area_floors_width :
  ∀ (X Y : Floor),
  area X = area Y →
  X.length = 18 →
  X.width = 10 →
  Y.length = 20 →
  Y.width = 9 :=
by sorry

end equal_area_floors_width_l4102_410208


namespace non_zero_digits_after_decimal_l4102_410295

theorem non_zero_digits_after_decimal (n : ℕ) (d : ℕ) : 
  (720 : ℚ) / (2^5 * 5^9) = n / (10^d) ∧ 
  n % 10 ≠ 0 ∧
  n < 10^4 ∧ 
  n ≥ 10^3 →
  d = 8 :=
sorry

end non_zero_digits_after_decimal_l4102_410295


namespace power_of_two_plus_one_square_l4102_410222

theorem power_of_two_plus_one_square (k z : ℕ) :
  2^k + 1 = z^2 → k = 2 ∧ z = 3 := by
  sorry

end power_of_two_plus_one_square_l4102_410222


namespace median_salary_is_25000_l4102_410282

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions. -/
def medianSalary (positions : List Position) : Nat :=
  sorry

/-- The list of positions in the company. -/
def companyPositions : List Position := [
  { title := "President", count := 1, salary := 135000 },
  { title := "Vice-President", count := 4, salary := 92000 },
  { title := "Director", count := 15, salary := 78000 },
  { title := "Associate Director", count := 8, salary := 55000 },
  { title := "Administrative Specialist", count := 30, salary := 25000 },
  { title := "Customer Service Representative", count := 12, salary := 20000 }
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat :=
  (companyPositions.map (·.count)).sum

theorem median_salary_is_25000 :
  totalEmployees = 70 ∧ medianSalary companyPositions = 25000 := by
  sorry

end median_salary_is_25000_l4102_410282


namespace duct_tape_cutting_time_l4102_410219

/-- The time required to cut all strands of duct tape -/
def cutting_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  total_strands / (hannah_rate + son_rate)

/-- Theorem stating the time required to cut all strands -/
theorem duct_tape_cutting_time :
  cutting_time 22 8 3 = 2 := by
  sorry

end duct_tape_cutting_time_l4102_410219


namespace perfect_square_fraction_l4102_410290

theorem perfect_square_fraction (n : ℤ) : 
  n > 2020 → 
  (∃ m : ℤ, (n - 2020) / (2120 - n) = m^2) → 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := by
sorry

end perfect_square_fraction_l4102_410290


namespace train_crossing_time_l4102_410217

/-- Proves that a train of length 500 m, traveling at 180 km/h, takes 10 seconds to cross an electric pole. -/
theorem train_crossing_time :
  let train_length : ℝ := 500  -- Length of the train in meters
  let train_speed_kmh : ℝ := 180  -- Speed of the train in km/h
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600  -- Speed in m/s
  let crossing_time : ℝ := train_length / train_speed_ms  -- Time to cross the pole
  crossing_time = 10 := by sorry

end train_crossing_time_l4102_410217


namespace price_restoration_l4102_410278

theorem price_restoration (original_price : ℝ) (markup_percentage : ℝ) (reduction_percentage : ℝ) : 
  markup_percentage = 25 →
  reduction_percentage = 20 →
  original_price * (1 + markup_percentage / 100) * (1 - reduction_percentage / 100) = original_price :=
by
  sorry

end price_restoration_l4102_410278


namespace space_shuttle_speed_conversion_l4102_410293

-- Define the speed in kilometers per second
def speed_km_per_second : ℝ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℝ := 3600

-- Theorem to prove
theorem space_shuttle_speed_conversion :
  speed_km_per_second * seconds_per_hour = 21600 := by
  sorry

end space_shuttle_speed_conversion_l4102_410293


namespace polynomial_division_remainder_l4102_410201

theorem polynomial_division_remainder :
  ∀ (R : Polynomial ℤ) (Q : Polynomial ℤ),
    (Polynomial.degree R < 2) →
    (x^101 : Polynomial ℤ) = (x^2 - 3*x + 2) * Q + R →
    R = 2^101 * (x - 1) - (x - 2) := by sorry

end polynomial_division_remainder_l4102_410201


namespace square_circle_union_area_l4102_410236

theorem square_circle_union_area : 
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π := by
  sorry

end square_circle_union_area_l4102_410236


namespace special_right_triangle_pair_theorem_l4102_410202

/-- Two right triangles with specific properties -/
structure SpecialRightTrianglePair where
  /-- The length of the common leg -/
  x : ℝ
  /-- The length of the other leg of T₁ -/
  y : ℝ
  /-- The length of the hypotenuse of T₁ -/
  w : ℝ
  /-- The length of the other leg of T₂ -/
  v : ℝ
  /-- The length of the hypotenuse of T₂ -/
  z : ℝ
  /-- Area of T₁ is 3 -/
  area_t1 : x * y / 2 = 3
  /-- Area of T₂ is 4 -/
  area_t2 : x * v / 2 = 4
  /-- Hypotenuse of T₁ is twice the length of the hypotenuse of T₂ -/
  hypotenuse_relation : w = 2 * z
  /-- Pythagorean theorem for T₁ -/
  pythagorean_t1 : x^2 + y^2 = w^2
  /-- Pythagorean theorem for T₂ -/
  pythagorean_t2 : x^2 + v^2 = z^2

/-- The square of the product of the third sides is 2304/25 -/
theorem special_right_triangle_pair_theorem (t : SpecialRightTrianglePair) :
  (t.y * t.v)^2 = 2304/25 := by
  sorry

end special_right_triangle_pair_theorem_l4102_410202


namespace mans_speed_in_still_water_l4102_410279

/-- Proves that given a man who swims downstream 36 km in 6 hours and upstream 18 km in 6 hours, his speed in still water is 4.5 km/h. -/
theorem mans_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h1 : downstream_distance = 36) 
  (h2 : upstream_distance = 18) 
  (h3 : time = 6) : 
  ∃ (speed_still_water : ℝ) (stream_speed : ℝ),
    speed_still_water + stream_speed = downstream_distance / time ∧ 
    speed_still_water - stream_speed = upstream_distance / time ∧
    speed_still_water = 4.5 := by
  sorry

end mans_speed_in_still_water_l4102_410279


namespace polynomial_division_remainder_l4102_410213

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^5 + 1 = (x^2 - 3*x + 5) * q + (11*x - 14) := by sorry

end polynomial_division_remainder_l4102_410213


namespace pet_store_cats_l4102_410280

theorem pet_store_cats (siamese_cats : ℕ) (cats_sold : ℕ) (cats_left : ℕ) (house_cats : ℕ) : 
  siamese_cats = 38 → 
  cats_sold = 45 → 
  cats_left = 18 → 
  siamese_cats + house_cats - cats_sold = cats_left → 
  house_cats = 25 := by
sorry

end pet_store_cats_l4102_410280


namespace area_between_circles_and_xaxis_l4102_410226

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_xaxis :
  let c1_center : ℝ × ℝ := (5, 5)
  let c2_center : ℝ × ℝ := (14, 5)
  let radius : ℝ := 3
  let rectangle_area : ℝ := (14 - 5) * 5
  let circle_segment_area : ℝ := 2 * (π * radius^2 / 4)
  rectangle_area - circle_segment_area = 45 - 9 * π / 2 := by
  sorry

end area_between_circles_and_xaxis_l4102_410226


namespace angle_equivalence_l4102_410206

/-- Given α = 2022°, if β has the same terminal side as α and β ∈ (0, 2π), then β = 37π/30 radians. -/
theorem angle_equivalence (α β : Real) : 
  α = 2022 * (π / 180) →  -- Convert 2022° to radians
  (∃ k : ℤ, β = α + 2 * π * k) →  -- Same terminal side
  0 < β ∧ β < 2 * π →  -- β ∈ (0, 2π)
  β = 37 * π / 30 := by
sorry

end angle_equivalence_l4102_410206


namespace triangle_theorem_l4102_410263

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - t.a^2 = (4 * Real.sqrt 2 / 3) * t.b * t.c)
  (h2 : 3 * t.c / t.a = Real.sqrt 2 * Real.sin t.B / Real.sin t.A)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 2) :
  Real.sin t.A = 1/3 ∧ 
  t.c = 2 * Real.sqrt 2 ∧ 
  Real.sin (2 * t.C - π/6) = (10 * Real.sqrt 6 - 23) / 54 := by
  sorry


end triangle_theorem_l4102_410263


namespace calvins_collection_size_l4102_410259

/-- Calculates the total number of insects in Calvin's collection. -/
def calvinsTotalInsects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := scorpions * 2
  roaches + scorpions + crickets + caterpillars

/-- Proves that Calvin has 27 insects in his collection. -/
theorem calvins_collection_size :
  calvinsTotalInsects 12 3 = 27 := by
  sorry

#eval calvinsTotalInsects 12 3

end calvins_collection_size_l4102_410259


namespace right_rectangular_prism_volume_l4102_410273

theorem right_rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 18)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    a * b * c = 24 * Real.sqrt 3 :=
by sorry

end right_rectangular_prism_volume_l4102_410273


namespace cube_equation_solution_l4102_410283

theorem cube_equation_solution :
  ∃! x : ℝ, (x + 3)^3 = (1/27)⁻¹ :=
by
  -- The unique solution is x = 0
  use 0
  constructor
  · -- Prove that x = 0 satisfies the equation
    sorry
  · -- Prove that this is the only solution
    sorry

end cube_equation_solution_l4102_410283


namespace isosceles_triangle_condition_l4102_410284

/-- If a, b, c are the sides of a triangle and satisfy the given equation,
    then the triangle is isosceles. -/
theorem isosceles_triangle_condition (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  3 * a^3 + 6 * a^2 * b - 3 * a^2 * c - 6 * a * b * c = 0 →
  (a = b ∨ b = c ∨ c = a) :=
by sorry

end isosceles_triangle_condition_l4102_410284


namespace polynomial_root_implication_l4102_410239

theorem polynomial_root_implication (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + a * (2 - 3 * Complex.I : ℂ) ^ 2 + 3 * (2 - 3 * Complex.I : ℂ) + b = 0 →
  a = -3/2 ∧ b = 65/2 := by
sorry

end polynomial_root_implication_l4102_410239


namespace root_product_theorem_l4102_410225

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - x^3 + 2*x^2 - x + 1

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0) (h₄ : f x₄ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ = 667 := by
  sorry

end root_product_theorem_l4102_410225


namespace expression_equals_73_l4102_410234

def x : ℤ := 2
def y : ℤ := -3
def z : ℤ := 6

theorem expression_equals_73 : x^2 + y^2 + z^2 + 2*x*y - 2*y*z = 73 := by
  sorry

end expression_equals_73_l4102_410234


namespace find_b_l4102_410228

theorem find_b (a b c : ℚ) 
  (sum_eq : a + b + c = 150)
  (equal_after_changes : a + 10 = b - 10 ∧ b - 10 = 3 * c) : 
  b = 520 / 7 := by
sorry

end find_b_l4102_410228


namespace parabola_equation_l4102_410230

/-- A parabola with x-axis as axis of symmetry and vertex at origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- The parabola passes through the point (-2, -4) -/
def passes_through (par : Parabola) : Prop :=
  par.eq (-2) (-4)

/-- The standard equation of the parabola is y^2 = -8x -/
def standard_equation (par : Parabola) : Prop :=
  par.p = -4

theorem parabola_equation :
  ∃ (par : Parabola), passes_through par ∧ standard_equation par :=
sorry

end parabola_equation_l4102_410230


namespace complement_A_in_U_l4102_410210

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≥ 1}

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- Theorem statement
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -1 ∨ x = 1 ∨ x > 2} := by sorry

end complement_A_in_U_l4102_410210


namespace balloon_expenses_l4102_410227

/-- The problem of calculating the total money Harry and Kevin brought to the store -/
theorem balloon_expenses (sheet_cost rope_cost propane_cost : ℕ)
  (helium_cost_per_oz : ℚ)
  (height_per_oz : ℕ)
  (max_height : ℕ) :
  sheet_cost = 42 →
  rope_cost = 18 →
  propane_cost = 14 →
  helium_cost_per_oz = 3/2 →
  height_per_oz = 113 →
  max_height = 9492 →
  ∃ (total_money : ℕ), total_money = 200 := by
  sorry

end balloon_expenses_l4102_410227


namespace fraction_solution_l4102_410265

theorem fraction_solution : ∃ x : ℝ, (x - 4) / (x^2) = 0 ∧ x = 4 := by
  sorry

end fraction_solution_l4102_410265


namespace ellipse_chords_and_bisector_l4102_410268

/-- Given an ellipse x²/2 + y² = 1, this theorem proves:
    1. The trajectory of midpoint of parallel chords with slope 2
    2. The trajectory of midpoint of chord defined by line passing through A(2,1)
    3. The line passing through P(1/2, 1/2) and bisected by P -/
theorem ellipse_chords_and_bisector 
  (x y : ℝ) (h : x^2/2 + y^2 = 1) :
  (∃ t : ℝ, x + 4*y = t) ∧ 
  (∃ s : ℝ, x^2 + 2*y^2 - 2*x - 2*y = s) ∧
  (2*x + 4*y - 3 = 0 → 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      x₁^2/2 + y₁^2 = 1 ∧ 
      x₂^2/2 + y₂^2 = 1 ∧ 
      (x₁ + x₂)/2 = 1/2 ∧ 
      (y₁ + y₂)/2 = 1/2) :=
by sorry

end ellipse_chords_and_bisector_l4102_410268


namespace meaningful_fraction_l4102_410276

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by sorry

end meaningful_fraction_l4102_410276


namespace no_valid_n_l4102_410289

theorem no_valid_n : ¬ ∃ (n : ℕ), 
  n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
sorry

end no_valid_n_l4102_410289


namespace bird_puzzle_solution_l4102_410235

theorem bird_puzzle_solution :
  ∃! (x y z : ℕ),
    x + y + z = 30 ∧
    (x : ℚ) / 3 + (y : ℚ) / 2 + 2 * (z : ℚ) = 30 ∧
    x = 9 ∧ y = 10 ∧ z = 11 := by
  sorry

end bird_puzzle_solution_l4102_410235


namespace apple_cider_volume_l4102_410240

/-- The volume of apple cider in a cylindrical pot -/
theorem apple_cider_volume (h : Real) (d : Real) (fill_ratio : Real) (cider_ratio : Real) :
  h = 9 →
  d = 4 →
  fill_ratio = 2/3 →
  cider_ratio = 2/7 →
  (fill_ratio * h * π * (d/2)^2) * cider_ratio = 48*π/7 :=
by sorry

end apple_cider_volume_l4102_410240


namespace cos_240_degrees_l4102_410247

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l4102_410247


namespace inequality_and_minimum_value_l4102_410269

theorem inequality_and_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ x : ℝ, 0 < x → x < 1 → (1 - x)^2 / x + x^2 / (1 - x) ≥ 1) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (1 - x)^2 / x + x^2 / (1 - x) = 1) :=
by sorry

end inequality_and_minimum_value_l4102_410269


namespace randy_initial_money_l4102_410258

theorem randy_initial_money :
  ∀ (initial_money : ℝ),
  let lunch_cost : ℝ := 10
  let remaining_after_lunch : ℝ := initial_money - lunch_cost
  let ice_cream_cost : ℝ := 5
  let ice_cream_fraction : ℝ := 1/4
  ice_cream_cost = ice_cream_fraction * remaining_after_lunch →
  initial_money = 30 :=
λ initial_money =>
  let lunch_cost : ℝ := 10
  let remaining_after_lunch : ℝ := initial_money - lunch_cost
  let ice_cream_cost : ℝ := 5
  let ice_cream_fraction : ℝ := 1/4
  λ h : ice_cream_cost = ice_cream_fraction * remaining_after_lunch =>
  sorry

#check randy_initial_money

end randy_initial_money_l4102_410258


namespace large_bucket_relation_tank_capacity_is_21_l4102_410261

/-- The capacity of a small bucket in liters -/
def small_bucket_capacity : ℝ := 0.5

/-- The capacity of a large bucket in liters -/
def large_bucket_capacity : ℝ := 4

/-- The number of small buckets used to fill the tank -/
def num_small_buckets : ℕ := 2

/-- The number of large buckets used to fill the tank -/
def num_large_buckets : ℕ := 5

/-- The relationship between small and large bucket capacities -/
theorem large_bucket_relation : large_bucket_capacity = 2 * small_bucket_capacity + 3 := by sorry

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := num_small_buckets * small_bucket_capacity + num_large_buckets * large_bucket_capacity

theorem tank_capacity_is_21 : tank_capacity = 21 := by sorry

end large_bucket_relation_tank_capacity_is_21_l4102_410261


namespace square_roots_problem_l4102_410204

theorem square_roots_problem (x m : ℝ) : 
  x > 0 ∧ 
  (m + 3)^2 = x ∧ 
  (2*m - 15)^2 = x ∧ 
  m + 3 ≠ 2*m - 15 → 
  x = 49 := by
sorry

end square_roots_problem_l4102_410204


namespace average_weight_increase_l4102_410211

theorem average_weight_increase (initial_count : ℕ) (original_weight replaced_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  original_weight = 101 →
  (original_weight - replaced_weight) / initial_count = 4.5 := by
  sorry

end average_weight_increase_l4102_410211


namespace fibonacci_inequality_l4102_410250

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) :
  (fibonacci (n + 1) : ℝ) ^ (1 / n : ℝ) ≥ 1 + 1 / ((fibonacci n : ℝ) ^ (1 / n : ℝ)) := by
  sorry

end fibonacci_inequality_l4102_410250


namespace train_A_time_l4102_410245

/-- Represents the properties of a train journey --/
structure TrainJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup --/
def trainProblem (routeLength : ℝ) (meetingPoint : ℝ) (trainBTime : ℝ) : Prop :=
  ∃ (trainA trainB : TrainJourney),
    -- Total route length
    routeLength = 75 ∧
    -- Train B's journey
    trainB.distance = routeLength ∧
    trainB.time = trainBTime ∧
    trainB.speed = trainB.distance / trainB.time ∧
    -- Train A's journey
    trainA.distance = routeLength ∧
    -- Meeting point
    meetingPoint = 30 ∧
    -- Trains meet at the same time
    meetingPoint / trainA.speed = (routeLength - meetingPoint) / trainB.speed ∧
    -- Train A's time is the total distance divided by its speed
    trainA.time = trainA.distance / trainA.speed

/-- The theorem to prove --/
theorem train_A_time : 
  ∀ (routeLength meetingPoint trainBTime : ℝ),
    trainProblem routeLength meetingPoint trainBTime →
    ∃ (trainA : TrainJourney), trainA.time = 3 := by
  sorry

end train_A_time_l4102_410245


namespace negation_of_universal_proposition_l4102_410243

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 2*x₀ + 4 > 0) := by
  sorry

end negation_of_universal_proposition_l4102_410243


namespace files_deleted_l4102_410248

theorem files_deleted (initial_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) : 
  initial_files = 27 → files_per_folder = 6 → num_folders = 3 →
  initial_files - (files_per_folder * num_folders) = 9 :=
by
  sorry

end files_deleted_l4102_410248


namespace boa_constrictor_length_is_70_l4102_410215

/-- The length of the garden snake in inches -/
def garden_snake_length : ℕ := 10

/-- The factor by which the boa constrictor is longer than the garden snake -/
def boa_length_factor : ℕ := 7

/-- The length of the boa constrictor in inches -/
def boa_constrictor_length : ℕ := garden_snake_length * boa_length_factor

theorem boa_constrictor_length_is_70 : boa_constrictor_length = 70 := by
  sorry

end boa_constrictor_length_is_70_l4102_410215


namespace fixed_point_of_arcsin_function_l4102_410291

theorem fixed_point_of_arcsin_function (m : ℝ) :
  ∃ (P : ℝ × ℝ), P = (0, -1) ∧ ∀ x : ℝ, m * Real.arcsin x - 1 = P.2 ↔ x = P.1 := by
  sorry

end fixed_point_of_arcsin_function_l4102_410291


namespace sum_of_first_and_third_l4102_410241

theorem sum_of_first_and_third (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B / C = 5 / 8 →
  B = 30 →
  A + C = 68 := by
sorry

end sum_of_first_and_third_l4102_410241


namespace fractional_equation_solution_l4102_410288

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 2) / (x - 1) = 0 ∧ x = -2 :=
by sorry

end fractional_equation_solution_l4102_410288


namespace inverse_of_7_mod_45_l4102_410205

theorem inverse_of_7_mod_45 : ∃ x : ℤ, 0 ≤ x ∧ x < 45 ∧ (7 * x) % 45 = 1 :=
  by
  use 32
  sorry

end inverse_of_7_mod_45_l4102_410205


namespace pauls_hourly_rate_is_35_l4102_410299

/-- Paul's Plumbing's hourly labor charge -/
def pauls_hourly_rate : ℝ := 35

/-- Paul's Plumbing's site visit fee -/
def pauls_visit_fee : ℝ := 55

/-- Reliable Plumbing's site visit fee -/
def reliable_visit_fee : ℝ := 75

/-- Reliable Plumbing's hourly labor charge -/
def reliable_hourly_rate : ℝ := 30

/-- The number of hours worked -/
def hours_worked : ℝ := 4

theorem pauls_hourly_rate_is_35 :
  pauls_hourly_rate = 35 ∧
  pauls_visit_fee + hours_worked * pauls_hourly_rate =
  reliable_visit_fee + hours_worked * reliable_hourly_rate :=
by sorry

end pauls_hourly_rate_is_35_l4102_410299


namespace laundry_time_calculation_l4102_410296

theorem laundry_time_calculation (loads : ℕ) (wash_time : ℕ) (dry_time : ℕ) :
  loads = 8 ∧ wash_time = 45 ∧ dry_time = 60 →
  (loads * (wash_time + dry_time)) / 60 = 14 := by
  sorry

end laundry_time_calculation_l4102_410296


namespace count_happy_license_plates_l4102_410246

/-- The set of allowed letters on the license plate -/
def allowed_letters : Finset Char := {'А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'}

/-- The set of consonant letters from the allowed letters -/
def consonant_letters : Finset Char := {'В', 'К', 'М', 'Н', 'Р', 'С', 'Т', 'Х'}

/-- The set of odd digits -/
def odd_digits : Finset Nat := {1, 3, 5, 7, 9}

/-- A license plate is a tuple of 3 letters and 3 digits -/
structure LicensePlate :=
  (letter1 : Char)
  (letter2 : Char)
  (digit1 : Nat)
  (digit2 : Nat)
  (digit3 : Nat)
  (letter3 : Char)

/-- A license plate is happy if the first two letters are consonants and the third digit is odd -/
def is_happy (plate : LicensePlate) : Prop :=
  plate.letter1 ∈ consonant_letters ∧
  plate.letter2 ∈ consonant_letters ∧
  plate.digit3 ∈ odd_digits

/-- The set of all valid license plates -/
def all_license_plates : Finset LicensePlate :=
  sorry

/-- The set of all happy license plates -/
def happy_license_plates : Finset LicensePlate :=
  sorry

/-- The main theorem: there are 384000 happy license plates -/
theorem count_happy_license_plates :
  Finset.card happy_license_plates = 384000 :=
sorry

end count_happy_license_plates_l4102_410246


namespace area_regular_hexagon_inscribed_circle_l4102_410256

/-- The area of a regular hexagon inscribed in a circle with radius 3 units -/
theorem area_regular_hexagon_inscribed_circle (r : ℝ) (h : r = 3) : 
  (6 : ℝ) * ((r^2 * Real.sqrt 3) / 4) = (27 * Real.sqrt 3) / 2 := by
  sorry

end area_regular_hexagon_inscribed_circle_l4102_410256


namespace simplify_expression_l4102_410285

theorem simplify_expression (x : ℝ) : 3*x + 2*x^2 + 5*x - x^2 + 7 = x^2 + 8*x + 7 := by
  sorry

end simplify_expression_l4102_410285


namespace increasing_function_sum_inequality_l4102_410220

/-- An increasing function on the real line. -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: For an increasing function f and real numbers a and b,
    if a + b ≥ 0, then f(a) + f(b) ≥ f(-a) + f(-b). -/
theorem increasing_function_sum_inequality
  (f : ℝ → ℝ) (hf : IncreasingFunction f) (a b : ℝ) :
  a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b) := by
  sorry

end increasing_function_sum_inequality_l4102_410220


namespace S_seven_two_l4102_410218

def S (a b : ℕ) : ℕ := 3 * a + 5 * b

theorem S_seven_two : S 7 2 = 31 := by
  sorry

end S_seven_two_l4102_410218


namespace three_points_determine_plane_l4102_410209

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space using the general equation ax + by + cz + d = 0
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if two planes are perpendicular
def perpendicularPlanes (p1 p2 : Plane) : Prop :=
  p1.a * p2.a + p1.b * p2.b + p1.c * p2.c = 0

-- Function to check if a point lies on a plane
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

-- Theorem statement
theorem three_points_determine_plane 
  (p1 p2 p3 : Plane) 
  (point1 point2 point3 : Point3D) : 
  perpendicularPlanes p1 p2 ∧ 
  perpendicularPlanes p2 p3 ∧ 
  perpendicularPlanes p3 p1 ∧ 
  pointOnPlane point1 p1 ∧ 
  pointOnPlane point2 p2 ∧ 
  pointOnPlane point3 p3 → 
  ∃! (resultPlane : Plane), 
    pointOnPlane point1 resultPlane ∧ 
    pointOnPlane point2 resultPlane ∧ 
    pointOnPlane point3 resultPlane :=
by
  sorry

end three_points_determine_plane_l4102_410209


namespace largest_s_value_l4102_410262

theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (59 * (s - 2) * r = 58 * s * (r - 2)) → s ≤ 117 ∧ ∃ r', r' ≥ s ∧ 59 * (117 - 2) * r' = 58 * 117 * (r' - 2) := by
  sorry

#check largest_s_value

end largest_s_value_l4102_410262


namespace john_chips_bought_l4102_410266

-- Define the cost of chips and corn chips
def chip_cost : ℚ := 2
def corn_chip_cost : ℚ := 3/2

-- Define John's budget
def budget : ℚ := 45

-- Define the number of corn chips John can buy with remaining money
def corn_chips_bought : ℚ := 10

-- Define the function to calculate the number of corn chips that can be bought with remaining money
def corn_chips_buyable (x : ℚ) : ℚ := (budget - chip_cost * x) / corn_chip_cost

-- Theorem statement
theorem john_chips_bought : 
  ∃ (x : ℚ), x = 15 ∧ corn_chips_buyable x = corn_chips_bought :=
sorry

end john_chips_bought_l4102_410266


namespace same_heads_probability_is_three_sixteenths_l4102_410297

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 6

/-- The probability of Ephraim getting the same number of heads as Keiko -/
def same_heads_probability : ℚ :=
  favorable_outcomes / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies)

theorem same_heads_probability_is_three_sixteenths :
  same_heads_probability = 3 / 16 := by sorry

end same_heads_probability_is_three_sixteenths_l4102_410297


namespace train_speed_kmph_l4102_410274

/-- Converts speed from meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 37.503

/-- Theorem: The train's speed in kilometers per hour is 135.0108 -/
theorem train_speed_kmph : mps_to_kmph train_speed_mps = 135.0108 := by
  sorry

end train_speed_kmph_l4102_410274


namespace bucket_weight_l4102_410223

theorem bucket_weight (c d : ℝ) : 
  (∃ (x y : ℝ), x + 3/4 * y = c ∧ x + 1/3 * y = d) → 
  (∃ (full_weight : ℝ), full_weight = (8*c - 3*d) / 5) :=
by sorry

end bucket_weight_l4102_410223


namespace max_rectangle_area_l4102_410237

/-- The maximum area of a rectangle with perimeter 156 feet and natural number sides --/
theorem max_rectangle_area (l w : ℕ) : 
  (2 * (l + w) = 156) → l * w ≤ 1521 := by
  sorry

end max_rectangle_area_l4102_410237


namespace sqrt_expression_value_l4102_410275

theorem sqrt_expression_value : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt 0.49) = 158 / 63 := by
  sorry

end sqrt_expression_value_l4102_410275


namespace sams_age_l4102_410298

theorem sams_age (sam drew alex jordan : ℕ) : 
  sam + drew + alex + jordan = 142 →
  sam = drew / 2 →
  alex = sam + 3 →
  jordan = 2 * alex →
  sam = 22 :=
by
  sorry

end sams_age_l4102_410298


namespace product_of_distinct_non_trivial_primes_last_digit_l4102_410232

def is_non_trivial_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 10

def last_digit (n : ℕ) : ℕ :=
  n % 10

theorem product_of_distinct_non_trivial_primes_last_digit 
  (p q : ℕ) (hp : is_non_trivial_prime p) (hq : is_non_trivial_prime q) (hpq : p ≠ q) :
  ∃ d : ℕ, d ∈ [1, 3, 7, 9] ∧ last_digit (p * q) = d :=
sorry

end product_of_distinct_non_trivial_primes_last_digit_l4102_410232


namespace expression_value_l4102_410207

theorem expression_value (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end expression_value_l4102_410207


namespace distributive_property_fraction_l4102_410233

theorem distributive_property_fraction (x y : ℝ) :
  (x + y) / 2 = x / 2 + y / 2 := by sorry

end distributive_property_fraction_l4102_410233


namespace function_non_negative_implies_bounds_l4102_410221

theorem function_non_negative_implies_bounds 
  (a b A B : ℝ) 
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
sorry

end function_non_negative_implies_bounds_l4102_410221


namespace triangle_problem_l4102_410242

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    prove the measure of angle A and the area of the triangle 
    under specific conditions. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) →  -- Given condition
  (0 < A ∧ A < π) →                          -- Angle A is in (0, π)
  (0 < B ∧ B < π) →                          -- Angle B is in (0, π)
  (0 < C ∧ C < π) →                          -- Angle C is in (0, π)
  (A + B + C = π) →                          -- Sum of angles in a triangle
  (a * Real.sin B = b * Real.sin A) →        -- Law of sines
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) → -- Law of cosines
  (A = π / 6) ∧                              -- First part of the theorem
  ((Real.cos B = 2 * Real.sqrt 2 / 3 ∧ a = Real.sqrt 2) →
   (1 / 2 * a * b * Real.sin C = (2 * Real.sqrt 2 + Real.sqrt 3) / 9)) -- Second part
  := by sorry

end triangle_problem_l4102_410242


namespace intersection_with_complement_l4102_410224

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end intersection_with_complement_l4102_410224


namespace april_rainfall_calculation_l4102_410251

/-- The amount of rainfall in March, in inches -/
def march_rainfall : ℝ := 0.81

/-- The difference in rainfall between March and April, in inches -/
def rainfall_difference : ℝ := 0.35

/-- The amount of rainfall in April, in inches -/
def april_rainfall : ℝ := march_rainfall - rainfall_difference

theorem april_rainfall_calculation :
  april_rainfall = 0.46 := by sorry

end april_rainfall_calculation_l4102_410251


namespace sixty_degrees_to_radians_l4102_410257

theorem sixty_degrees_to_radians :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian 60 = π / 3 := by
  sorry

end sixty_degrees_to_radians_l4102_410257


namespace mistaken_divisor_l4102_410249

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = correct_divisor * 20 →
  dividend = mistaken_divisor * 35 →
  mistaken_divisor = 12 := by
sorry

end mistaken_divisor_l4102_410249


namespace largest_power_divides_product_l4102_410252

/-- The largest power of the largest prime that divides n -/
def pow (n : ℕ) : ℕ :=
  sorry

/-- The product of pow(n) for n from 2 to 7200 -/
def product_pow : ℕ :=
  sorry

/-- 2020 raised to the power of the result -/
def power_2020 (m : ℕ) : ℕ :=
  2020^m

theorem largest_power_divides_product :
  ∃ m : ℕ, m = 72 ∧
  power_2020 m ∣ product_pow ∧
  ∀ k > m, ¬(power_2020 k ∣ product_pow) :=
sorry

end largest_power_divides_product_l4102_410252


namespace marcus_pebble_ratio_l4102_410287

def pebble_ratio (initial : ℕ) (received : ℕ) (final : ℕ) : Prop :=
  let skipped := initial + received - final
  (2 * skipped = initial) ∧ (skipped ≠ 0)

theorem marcus_pebble_ratio :
  pebble_ratio 18 30 39 := by
  sorry

end marcus_pebble_ratio_l4102_410287


namespace quartic_sum_l4102_410260

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, Q x = a * x^4 + b * x^3 + c * x^2 + d * x + m
  at_zero : Q 0 = m
  at_one : Q 1 = 3 * m
  at_neg_one : Q (-1) = 4 * m
  at_two : Q 2 = 5 * m

/-- The sum of the polynomial evaluated at 3 and -3 equals 407m -/
theorem quartic_sum (m : ℝ) (P : QuarticPolynomial m) : P.Q 3 + P.Q (-3) = 407 * m := by
  sorry

end quartic_sum_l4102_410260


namespace largest_product_of_three_primes_digit_sum_l4102_410212

def is_single_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n a b : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    a ≠ b ∧
    is_prime a ∧
    is_prime b ∧
    is_prime (a + b) ∧
    n = a * b * (a + b) ∧
    (∀ (m : ℕ), 
      (∃ (x y : ℕ), 
        is_single_digit x ∧ 
        is_single_digit y ∧ 
        x ≠ y ∧ 
        is_prime x ∧ 
        is_prime y ∧ 
        is_prime (x + y) ∧ 
        m = x * y * (x + y)) → m ≤ n) ∧
    sum_of_digits n = 7 :=
by sorry

end largest_product_of_three_primes_digit_sum_l4102_410212


namespace employee_count_l4102_410253

theorem employee_count (average_salary : ℕ) (new_average_salary : ℕ) (manager_salary : ℕ) :
  average_salary = 2400 →
  new_average_salary = 2500 →
  manager_salary = 4900 →
  ∃ n : ℕ, n * average_salary + manager_salary = (n + 1) * new_average_salary ∧ n = 24 :=
by sorry

end employee_count_l4102_410253


namespace max_subset_size_2021_l4102_410267

/-- Given a natural number N, returns the maximum size of a subset A of {1, ..., N}
    such that any two numbers in A are neither coprime nor have a divisibility relationship. -/
def maxSubsetSize (N : ℕ) : ℕ :=
  sorry

/-- The maximum subset size for N = 2021 is 505. -/
theorem max_subset_size_2021 : maxSubsetSize 2021 = 505 := by
  sorry

end max_subset_size_2021_l4102_410267


namespace exponent_multiplication_l4102_410286

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l4102_410286
