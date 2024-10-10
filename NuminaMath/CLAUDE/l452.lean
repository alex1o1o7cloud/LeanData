import Mathlib

namespace angle_C_measure_l452_45246

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem angle_C_measure (abc : Triangle) (h1 : abc.A = 50) (h2 : abc.B = 60) : abc.C = 70 := by
  sorry

end angle_C_measure_l452_45246


namespace square_plus_divisor_not_perfect_square_plus_divisor_perfect_iff_l452_45200

def is_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = m^2

theorem square_plus_divisor_not_perfect (n d : ℕ) (hn : n > 0) (hd : d > 0) (hdiv : d ∣ 2*n^2) :
  ¬ is_perfect_square (n^2 + d) := by sorry

theorem square_plus_divisor_perfect_iff (n d : ℕ) (hn : n > 0) (hd : d > 0) (hdiv : d ∣ 3*n^2) :
  is_perfect_square (n^2 + d) ↔ d = 3*n^2 := by sorry

end square_plus_divisor_not_perfect_square_plus_divisor_perfect_iff_l452_45200


namespace nisos_population_meets_capacity_l452_45236

/-- Represents the state of Nisos island at a given time -/
structure NisosState where
  year : ℕ
  population : ℕ

/-- Calculates the population after a given number of 20-year periods -/
def population_after (initial_population : ℕ) (periods : ℕ) : ℕ :=
  initial_population * (4 ^ periods)

/-- Theorem: Nisos island population meets capacity limit after 60 years -/
theorem nisos_population_meets_capacity : 
  ∀ (initial_state : NisosState),
    initial_state.year = 1998 →
    initial_state.population = 100 →
    ∃ (final_state : NisosState),
      final_state.year = initial_state.year + 60 ∧
      final_state.population ≥ 7500 ∧
      final_state.population < population_after 100 4 :=
sorry

/-- The land area of Nisos island in hectares -/
def nisos_area : ℕ := 15000

/-- The land area required per person in hectares -/
def land_per_person : ℕ := 2

/-- The capacity of Nisos island -/
def nisos_capacity : ℕ := nisos_area / land_per_person

/-- The population growth factor per 20-year period -/
def growth_factor : ℕ := 4

/-- The number of 20-year periods in 60 years -/
def periods_in_60_years : ℕ := 3

end nisos_population_meets_capacity_l452_45236


namespace mix_g_weekly_amount_l452_45295

/-- Calculates the weekly amount of Mix G birdseed needed for pigeons -/
def weekly_mix_g_amount (num_pigeons : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  num_pigeons * daily_consumption * days

/-- Theorem stating that the weekly amount of Mix G birdseed needed is 168 grams -/
theorem mix_g_weekly_amount :
  weekly_mix_g_amount 6 4 7 = 168 :=
by sorry

end mix_g_weekly_amount_l452_45295


namespace sum_of_digits_divisible_by_nine_l452_45226

/-- The sum of digits in a number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The concatenation of numbers from 1 to n -/
def concatenateNumbers (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in the concatenation of numbers from 1 to 2015 is divisible by 9 -/
theorem sum_of_digits_divisible_by_nine :
  ∃ k : ℕ, sumOfDigits (concatenateNumbers 2015) = 9 * k := by sorry

end sum_of_digits_divisible_by_nine_l452_45226


namespace triangle_side_mod_three_l452_45224

/-- 
Given two triangles with the same perimeter, where the first is equilateral
with integer side lengths and the second has integer side lengths with one
side of length 1 and another of length d, then d ≡ 1 (mod 3).
-/
theorem triangle_side_mod_three (a d : ℕ) : 
  (3 * a = 2 * d + 1) → (d % 3 = 1) := by
  sorry

end triangle_side_mod_three_l452_45224


namespace M_equals_N_l452_45265

/-- Set M of integers defined as 12m + 8n + 4l where m, n, l are integers -/
def M : Set ℤ := {u : ℤ | ∃ (m n l : ℤ), u = 12*m + 8*n + 4*l}

/-- Set N of integers defined as 20p + 16q + 12r where p, q, r are integers -/
def N : Set ℤ := {u : ℤ | ∃ (p q r : ℤ), u = 20*p + 16*q + 12*r}

/-- Theorem stating that set M is equal to set N -/
theorem M_equals_N : M = N := by sorry

end M_equals_N_l452_45265


namespace homeroom_teacher_selection_count_l452_45263

/-- The number of ways to arrange k elements from n distinct elements -/
def arrangementCount (n k : ℕ) : ℕ := sorry

/-- The number of valid selection schemes for homeroom teachers -/
def validSelectionCount (maleTotalCount femaleTotalCount selectCount : ℕ) : ℕ :=
  arrangementCount (maleTotalCount + femaleTotalCount) selectCount -
  (arrangementCount maleTotalCount selectCount + arrangementCount femaleTotalCount selectCount)

theorem homeroom_teacher_selection_count :
  validSelectionCount 5 4 3 = 420 := by sorry

end homeroom_teacher_selection_count_l452_45263


namespace sum_of_digits_inequality_l452_45266

/-- Sum of digits function -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: For any positive integer n, s(n) ≤ 8 * s(8n) -/
theorem sum_of_digits_inequality (n : ℕ+) : sum_of_digits n ≤ 8 * sum_of_digits (8 * n) := by
  sorry

end sum_of_digits_inequality_l452_45266


namespace combined_paint_cost_l452_45215

/-- Represents the dimensions and painting cost of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (breadth : ℝ)
  (paint_rate : ℝ)

/-- Calculates the area of a rectangular floor -/
def floor_area (f : Floor) : ℝ := f.length * f.breadth

/-- Calculates the cost to paint a floor -/
def paint_cost (f : Floor) : ℝ := floor_area f * f.paint_rate

/-- Represents the two-story building -/
structure Building :=
  (first_floor : Floor)
  (second_floor : Floor)

/-- The main theorem to prove -/
theorem combined_paint_cost (b : Building) : ℝ :=
  let f1 := b.first_floor
  let f2 := b.second_floor
  have h1 : f1.length = 3 * f1.breadth := by sorry
  have h2 : paint_cost f1 = 484 := by sorry
  have h3 : f1.paint_rate = 3 := by sorry
  have h4 : f2.length = 0.8 * f1.length := by sorry
  have h5 : f2.breadth = 1.3 * f1.breadth := by sorry
  have h6 : f2.paint_rate = 5 := by sorry
  have h7 : paint_cost f1 + paint_cost f2 = 1320.8 := by sorry
  1320.8

#check combined_paint_cost

end combined_paint_cost_l452_45215


namespace hyperbola_properties_l452_45202

/-- Given a hyperbola with equation (x^2 / 9) - (y^2 / 16) = 1, 
    prove its eccentricity and asymptote equations -/
theorem hyperbola_properties :
  let hyperbola := fun (x y : ℝ) => (x^2 / 9) - (y^2 / 16) = 1
  let eccentricity := 5/3
  let asymptote := fun (x : ℝ) => (4/3) * x
  (∀ x y, hyperbola x y → 
    (∃ c, c^2 = 25 ∧ eccentricity = c / 3)) ∧
  (∀ x, hyperbola x (asymptote x) ∨ hyperbola x (-asymptote x)) :=
by sorry

end hyperbola_properties_l452_45202


namespace pure_imaginary_fraction_l452_45288

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end pure_imaginary_fraction_l452_45288


namespace giant_kite_area_l452_45277

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a kite given its four vertices -/
def kiteArea (p1 p2 p3 p4 : Point) : ℝ :=
  let base := p3.x - p1.x
  let height := p2.y - p1.y
  base * height

/-- Theorem: The area of the specified kite is 72 square inches -/
theorem giant_kite_area :
  let p1 : Point := ⟨2, 12⟩
  let p2 : Point := ⟨8, 18⟩
  let p3 : Point := ⟨14, 12⟩
  let p4 : Point := ⟨8, 2⟩
  kiteArea p1 p2 p3 p4 = 72 := by
  sorry

end giant_kite_area_l452_45277


namespace line_slope_angle_l452_45253

theorem line_slope_angle (x y : ℝ) : 
  y - Real.sqrt 3 * x + 5 = 0 → 
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ Real.tan α = Real.sqrt 3 ∧ α = π / 3 :=
by sorry

end line_slope_angle_l452_45253


namespace power_sum_of_i_l452_45286

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^20 + i^39 = 1 - i := by
  sorry

end power_sum_of_i_l452_45286


namespace sum_of_solutions_l452_45268

theorem sum_of_solutions (x y : ℝ) 
  (hx : x^3 - 6*x^2 + 12*x = 13) 
  (hy : y^3 + 3*y - 3*y^2 = -4) : 
  x + y = 3 := by
sorry

end sum_of_solutions_l452_45268


namespace largest_divisor_with_equal_quotient_remainder_l452_45248

theorem largest_divisor_with_equal_quotient_remainder :
  ∀ (A B C : ℕ),
    (10 = A * B + C) →
    (B = C) →
    A ≤ 9 ∧
    (∃ (A' : ℕ), A' = 9 ∧ ∃ (B' C' : ℕ), 10 = A' * B' + C' ∧ B' = C') :=
by sorry

end largest_divisor_with_equal_quotient_remainder_l452_45248


namespace correct_observation_value_l452_45252

theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (corrected_mean : ℝ)
  (h_n : n = 20)
  (h_initial_mean : initial_mean = 36)
  (h_wrong_value : wrong_value = 40)
  (h_corrected_mean : corrected_mean = 34.9) :
  (n : ℝ) * initial_mean - wrong_value + (n : ℝ) * corrected_mean - ((n : ℝ) * initial_mean - wrong_value) = 18 := by
  sorry

end correct_observation_value_l452_45252


namespace pattern_equality_l452_45208

theorem pattern_equality (n : ℤ) : n * (n + 2) - (n + 1)^2 = -1 := by
  sorry

end pattern_equality_l452_45208


namespace nilpotent_matrix_square_zero_l452_45244

theorem nilpotent_matrix_square_zero 
  (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : 
  B ^ 2 = 0 := by
sorry

end nilpotent_matrix_square_zero_l452_45244


namespace cooler_capacity_increase_l452_45242

/-- Given three coolers with specific capacity relationships, prove the percentage increase from the first to the second cooler --/
theorem cooler_capacity_increase (a b c : ℝ) : 
  a = 100 → 
  b > a → 
  c = b / 2 → 
  a + b + c = 325 → 
  (b - a) / a * 100 = 50 := by
  sorry

end cooler_capacity_increase_l452_45242


namespace cos_30_degrees_l452_45285

theorem cos_30_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_30_degrees_l452_45285


namespace fraction_equality_l452_45228

theorem fraction_equality (a b : ℝ) (h : a / (a + b) = 3 / 4) : a / b = 3 := by
  sorry

end fraction_equality_l452_45228


namespace homework_completion_l452_45227

theorem homework_completion (total : ℝ) (h : total > 0) : 
  let monday := (3 / 5 : ℝ) * total
  let tuesday := (1 / 3 : ℝ) * (total - monday)
  let wednesday := total - monday - tuesday
  wednesday = (4 / 15 : ℝ) * total := by
  sorry

end homework_completion_l452_45227


namespace root_product_value_l452_45235

theorem root_product_value (p q : ℝ) : 
  3 * p ^ 2 + 9 * p - 21 = 0 →
  3 * q ^ 2 + 9 * q - 21 = 0 →
  (3 * p - 4) * (6 * q - 8) = -22 := by
sorry

end root_product_value_l452_45235


namespace cubic_equation_roots_l452_45289

theorem cubic_equation_roots (x : ℝ) : ∃ (a b : ℝ),
  (x^3 - x^2 - 2*x + 1 = 0) ∧ (a^3 - a^2 - 2*a + 1 = 0) ∧ (b^3 - b^2 - 2*b + 1 = 0) ∧ (a - a*b = 1) :=
by sorry

end cubic_equation_roots_l452_45289


namespace souvenir_problem_l452_45225

/-- Represents the cost and selling prices of souvenirs -/
structure SouvenirPrices where
  costA : ℝ
  costB : ℝ
  sellingB : ℝ

/-- Represents the quantity and profit of souvenirs -/
structure SouvenirQuantities where
  totalQuantity : ℕ
  minQuantityA : ℕ

/-- Theorem stating the properties of the souvenir problem -/
theorem souvenir_problem 
  (prices : SouvenirPrices) 
  (quantities : SouvenirQuantities) 
  (h1 : prices.costA = prices.costB + 30)
  (h2 : 1000 / prices.costA = 400 / prices.costB)
  (h3 : quantities.totalQuantity = 200)
  (h4 : quantities.minQuantityA ≥ 60)
  (h5 : quantities.minQuantityA < quantities.totalQuantity - quantities.minQuantityA)
  (h6 : prices.sellingB = 30) :
  prices.costA = 50 ∧ 
  prices.costB = 20 ∧
  (∃ x : ℝ, x = 65 ∧ (x - prices.costA) * (400 - 5*x) = 1125) ∧
  (∃ y : ℝ, y = 2480 ∧ 
    y = (68 - prices.costA) * (400 - 5*68) + 
        (prices.sellingB - prices.costB) * (quantities.totalQuantity - (400 - 5*68))) :=
by sorry

end souvenir_problem_l452_45225


namespace twins_age_problem_l452_45213

theorem twins_age_problem (age : ℕ) : 
  (age * age) + 5 = ((age + 1) * (age + 1)) → age = 2 := by
  sorry

end twins_age_problem_l452_45213


namespace positive_integer_N_equals_121_l452_45239

theorem positive_integer_N_equals_121 :
  ∃ (N : ℕ), N > 0 ∧ 33^2 * 55^2 = 15^2 * N^2 ∧ N = 121 := by
  sorry

end positive_integer_N_equals_121_l452_45239


namespace veranda_width_l452_45283

/-- Given a rectangular room with length 19 m and width 12 m, surrounded by a veranda on all sides
    with an area of 140 m², prove that the width of the veranda is 2 m. -/
theorem veranda_width (room_length : ℝ) (room_width : ℝ) (veranda_area : ℝ) :
  room_length = 19 →
  room_width = 12 →
  veranda_area = 140 →
  ∃ (w : ℝ), w = 2 ∧
    (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width = veranda_area :=
by sorry

end veranda_width_l452_45283


namespace twenty_is_eighty_percent_of_twentyfive_l452_45257

theorem twenty_is_eighty_percent_of_twentyfive : ∃ y : ℝ, y > 0 ∧ 20 / y = 80 / 100 → y = 25 := by
  sorry

end twenty_is_eighty_percent_of_twentyfive_l452_45257


namespace problem_solution_l452_45229

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)

theorem problem_solution (a : ℕ → ℝ) (h : sequence_property a) (h10 : a 10 = 10) : 
  a 22 = 10 := by
sorry

end problem_solution_l452_45229


namespace speed_difference_l452_45241

/-- Proves that the difference between the car's and truck's average speeds is 18 km/h -/
theorem speed_difference (truck_distance : ℝ) (truck_time : ℝ) (car_time : ℝ) (distance_difference : ℝ)
  (h1 : truck_distance = 296)
  (h2 : truck_time = 8)
  (h3 : car_time = 5.5)
  (h4 : distance_difference = 6.5)
  (h5 : (truck_distance + distance_difference) / car_time > truck_distance / truck_time) :
  (truck_distance + distance_difference) / car_time - truck_distance / truck_time = 18 := by
  sorry

end speed_difference_l452_45241


namespace bananas_left_l452_45221

theorem bananas_left (initial : ℕ) (eaten : ℕ) : 
  initial = 12 → eaten = 4 → initial - eaten = 8 := by
  sorry

end bananas_left_l452_45221


namespace plane_equation_l452_45294

/-- A plane in 3D space defined by a normal vector and a point it passes through. -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Checks if a given point lies on the plane. -/
def Plane.contains (π : Plane) (p : ℝ × ℝ × ℝ) : Prop :=
  let (nx, ny, nz) := π.normal
  let (ax, ay, az) := π.point
  let (x, y, z) := p
  nx * (x - ax) + ny * (y - ay) + nz * (z - az) = 0

/-- The main theorem stating the equation of the plane. -/
theorem plane_equation (π : Plane) (h : π.normal = (1, -1, 2) ∧ π.point = (0, 3, 1)) :
  ∀ p : ℝ × ℝ × ℝ, π.contains p ↔ p.1 - p.2.1 + 2 * p.2.2 + 1 = 0 := by
  sorry

end plane_equation_l452_45294


namespace simplify_and_evaluate_l452_45267

theorem simplify_and_evaluate (a : ℚ) (h : a = -3/2) :
  (a + 2)^2 - (a + 1)*(a - 1) = -1 := by
  sorry

end simplify_and_evaluate_l452_45267


namespace quadratic_two_real_roots_l452_45279

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x - 1 = 0 ∧ k * y^2 - 6 * y - 1 = 0) ↔
  (k ≥ -9 ∧ k ≠ 0) :=
by sorry

end quadratic_two_real_roots_l452_45279


namespace percentage_without_muffin_l452_45293

theorem percentage_without_muffin (muffin yogurt fruit granola : ℝ) :
  muffin = 38 →
  yogurt = 10 →
  fruit = 27 →
  granola = 25 →
  muffin + yogurt + fruit + granola = 100 →
  100 - muffin = 62 :=
by sorry

end percentage_without_muffin_l452_45293


namespace betty_age_l452_45216

/-- Given the ages of Albert, Mary, and Betty, prove that Betty is 4 years old. -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 8) : 
  betty = 4 := by
sorry

end betty_age_l452_45216


namespace robins_haircut_l452_45278

theorem robins_haircut (initial_length current_length : ℕ) 
  (h1 : initial_length = 17)
  (h2 : current_length = 13) :
  initial_length - current_length = 4 := by
  sorry

end robins_haircut_l452_45278


namespace stock_income_calculation_l452_45299

/-- Calculates the income derived from a stock investment --/
theorem stock_income_calculation
  (interest_rate : ℝ)
  (investment_amount : ℝ)
  (brokerage_rate : ℝ)
  (market_value_per_100 : ℝ)
  (h1 : interest_rate = 0.105)
  (h2 : investment_amount = 6000)
  (h3 : brokerage_rate = 0.0025)
  (h4 : market_value_per_100 = 83.08333333333334) :
  let brokerage_fee := investment_amount * brokerage_rate
  let actual_investment := investment_amount - brokerage_fee
  let num_units := actual_investment / market_value_per_100
  let face_value := num_units * 100
  let income := face_value * interest_rate
  income = 756 := by sorry

end stock_income_calculation_l452_45299


namespace line_intersects_circle_l452_45219

theorem line_intersects_circle (a : ℝ) (h : a ≥ 0) :
  ∃ (x y : ℝ), (a * x - y + Real.sqrt 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end line_intersects_circle_l452_45219


namespace strip_length_is_14_l452_45245

/-- Represents a folded rectangular strip of paper -/
structure FoldedStrip :=
  (width : ℝ)
  (ap_length : ℝ)
  (bm_length : ℝ)

/-- Calculates the total length of the folded strip -/
def total_length (strip : FoldedStrip) : ℝ :=
  strip.ap_length + strip.width + strip.bm_length

/-- Theorem: The length of the rectangular strip is 14 cm -/
theorem strip_length_is_14 (strip : FoldedStrip) 
  (h_width : strip.width = 4)
  (h_ap : strip.ap_length = 5)
  (h_bm : strip.bm_length = 5) : 
  total_length strip = 14 :=
by
  sorry

#eval total_length { width := 4, ap_length := 5, bm_length := 5 }

end strip_length_is_14_l452_45245


namespace age_change_proof_l452_45287

theorem age_change_proof (n : ℕ) (A : ℝ) : 
  ((n + 1) * (A + 7) = n * A + 39) →
  ((n + 1) * (A - 1) = n * A + 15) →
  n = 2 := by
  sorry

end age_change_proof_l452_45287


namespace number_relationship_l452_45231

theorem number_relationship (s l : ℕ) : 
  s + l = 124 → s = 31 → l = s + 62 := by
  sorry

end number_relationship_l452_45231


namespace delta_sports_club_ratio_l452_45259

/-- Proves that the ratio of female to male members is 2/3 given the average ages --/
theorem delta_sports_club_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) : 
  (35 : ℝ) * f + 30 * m = 32 * (f + m) → f / m = 2 / 3 := by
  sorry

end delta_sports_club_ratio_l452_45259


namespace remainder_1999_11_mod_8_l452_45243

theorem remainder_1999_11_mod_8 : 1999^11 % 8 = 7 := by
  sorry

end remainder_1999_11_mod_8_l452_45243


namespace proposition_is_true_l452_45230

theorem proposition_is_true : ∀ x : ℝ, x > 2 → Real.log (x - 1) + x^2 + 4 > 4*x := by
  sorry

end proposition_is_true_l452_45230


namespace sum_of_rectangle_areas_l452_45250

def first_six_odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11]

def rectangle_areas (base_width : ℕ) (lengths : List ℕ) : List ℕ :=
  lengths.map (λ l => base_width * l^2)

theorem sum_of_rectangle_areas :
  let base_width := 2
  let areas := rectangle_areas base_width first_six_odd_numbers
  List.sum areas = 572 := by sorry

end sum_of_rectangle_areas_l452_45250


namespace sqrt_200_equals_10_l452_45232

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end sqrt_200_equals_10_l452_45232


namespace function_inequality_l452_45205

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_ineq : ∀ x, x ≠ 1 → (x - 1) * deriv f x < 0)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = f 0.5)
  (h_b : b = f (4/3))
  (h_c : c = f 3) :
  b > a ∧ a > c := by sorry

end function_inequality_l452_45205


namespace gcd_consecutive_b_terms_is_one_l452_45260

def b (n : ℕ) : ℕ := 2 * n.factorial + n

theorem gcd_consecutive_b_terms_is_one (n : ℕ) : 
  Nat.gcd (b n) (b (n + 1)) = 1 := by sorry

end gcd_consecutive_b_terms_is_one_l452_45260


namespace triangle_side_length_l452_45261

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → B = π/4 → b = Real.sqrt 6 - Real.sqrt 2 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b / Real.sin B = c / Real.sin C →
  c = Real.sqrt 2 := by
sorry

end triangle_side_length_l452_45261


namespace complex_cube_root_sum_l452_45255

theorem complex_cube_root_sum (z : ℂ) (h1 : z^3 = 1) (h2 : z ≠ 1) :
  z^103 + z^104 + z^105 + z^106 + z^107 + z^108 = 0 := by
  sorry

end complex_cube_root_sum_l452_45255


namespace triangle_inequality_l452_45203

/-- A structure representing a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A structure representing a line in a 2D plane -/
structure Line where
  m : ℝ
  n : ℝ
  p : ℝ

/-- Function to calculate the area of a triangle -/
def areaOfTriangle (t : Triangle) : ℝ := sorry

/-- Function to calculate the tangent of an angle in a triangle -/
def tanAngle (t : Triangle) (vertex : Fin 3) : ℝ := sorry

/-- Function to calculate the perpendicular distance from a point to a line -/
def perpDistance (point : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- The main theorem -/
theorem triangle_inequality (t : Triangle) (l : Line) :
  let u := perpDistance t.A l
  let v := perpDistance t.B l
  let w := perpDistance t.C l
  let S := areaOfTriangle t
  u^2 * tanAngle t 0 + v^2 * tanAngle t 1 + w^2 * tanAngle t 2 ≥ 2 * S := by
  sorry

end triangle_inequality_l452_45203


namespace find_second_number_l452_45222

theorem find_second_number (G N : ℕ) (h1 : G = 101) (h2 : 4351 % G = 8) (h3 : N % G = 10) :
  N = 4359 := by
  sorry

end find_second_number_l452_45222


namespace seokgi_jumped_furthest_l452_45234

/-- Represents the jump distances of three people -/
structure JumpDistances where
  yooseung : ℚ
  shinyoung : ℚ
  seokgi : ℚ

/-- Given the jump distances, proves that Seokgi jumped the furthest -/
theorem seokgi_jumped_furthest (j : JumpDistances)
  (h1 : j.yooseung = 15/8)
  (h2 : j.shinyoung = 2)
  (h3 : j.seokgi = 17/8) :
  j.seokgi > j.yooseung ∧ j.seokgi > j.shinyoung :=
by sorry

end seokgi_jumped_furthest_l452_45234


namespace inequality_proof_l452_45291

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end inequality_proof_l452_45291


namespace largest_quantity_l452_45214

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2012 / 2011 + 2010 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : C > A ∧ C > B := by sorry

end largest_quantity_l452_45214


namespace inner_polygon_smaller_perimeter_l452_45207

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool
  
/-- Calculate the perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : Real :=
  sorry

/-- Check if one polygon is contained within another -/
def is_contained_in (inner outer : ConvexPolygon) : Prop :=
  sorry

/-- Theorem: The perimeter of an inner convex polygon is smaller than that of the outer convex polygon -/
theorem inner_polygon_smaller_perimeter
  (inner outer : ConvexPolygon)
  (h_inner_convex : inner.is_convex = true)
  (h_outer_convex : outer.is_convex = true)
  (h_contained : is_contained_in inner outer) :
  perimeter inner < perimeter outer :=
sorry

end inner_polygon_smaller_perimeter_l452_45207


namespace platform_length_l452_45282

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 27 →
  pole_time = 18 →
  (train_length * platform_time / pole_time) - train_length = 150 := by
sorry

end platform_length_l452_45282


namespace prime_squared_minus_one_divisibility_l452_45281

theorem prime_squared_minus_one_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_ge_7 : p ≥ 7) :
  (∃ q : ℕ, Nat.Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Nat.Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) :=
sorry

end prime_squared_minus_one_divisibility_l452_45281


namespace max_clock_digit_sum_l452_45292

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def clock_digit_sum (h m : ℕ) : ℕ := digit_sum h + digit_sum m

theorem max_clock_digit_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  clock_digit_sum h m ≥ clock_digit_sum h' m' ∧
  clock_digit_sum h m = 24 :=
sorry

end max_clock_digit_sum_l452_45292


namespace train_average_speed_with_stoppages_l452_45209

theorem train_average_speed_with_stoppages 
  (speed_without_stoppages : ℝ)
  (stop_time_per_hour : ℝ)
  (h1 : speed_without_stoppages = 100)
  (h2 : stop_time_per_hour = 3)
  : (speed_without_stoppages * (60 - stop_time_per_hour) / 60) = 95 := by
  sorry

end train_average_speed_with_stoppages_l452_45209


namespace function_monotonicity_l452_45262

theorem function_monotonicity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x^2 - 3*x + 2) * (deriv (deriv f) x) < 0) :
  ∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 := by sorry

end function_monotonicity_l452_45262


namespace weight_2019_is_9_5_l452_45211

/-- The weight of a stick in kilograms -/
def stick_weight : ℝ := 0.5

/-- The number of sticks in each digit -/
def sticks_in_digit : Fin 10 → ℕ
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0

/-- The weight of the number 2019 in kilograms -/
def weight_2019 : ℝ :=
  (sticks_in_digit 2 + sticks_in_digit 0 + sticks_in_digit 1 + sticks_in_digit 9) * stick_weight

/-- Theorem: The weight of the number 2019 is 9.5 kg -/
theorem weight_2019_is_9_5 : weight_2019 = 9.5 := by
  sorry

end weight_2019_is_9_5_l452_45211


namespace exists_vector_not_in_span_l452_45233

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (-4, -2)

/-- The statement to be proven -/
theorem exists_vector_not_in_span : ∃ d : ℝ × ℝ, ∀ k₁ k₂ : ℝ, d ≠ k₁ • b + k₂ • c := by
  sorry

end exists_vector_not_in_span_l452_45233


namespace cyclist_speed_proof_l452_45210

/-- Represents the speed of the east-bound cyclist in mph -/
def east_speed : ℝ := 18

/-- Represents the speed of the west-bound cyclist in mph -/
def west_speed : ℝ := east_speed + 4

/-- Represents the time traveled in hours -/
def time : ℝ := 5

/-- Represents the total distance between the cyclists after the given time -/
def total_distance : ℝ := 200

theorem cyclist_speed_proof :
  east_speed * time + west_speed * time = total_distance :=
by sorry

end cyclist_speed_proof_l452_45210


namespace hexadecagon_triangles_l452_45251

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- Represents that no three vertices of the hexadecagon are collinear -/
axiom no_collinear_vertices : True

/-- The number of triangles that can be formed using the vertices of a regular hexadecagon -/
def num_triangles : ℕ := Nat.choose n 3

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end hexadecagon_triangles_l452_45251


namespace unique_number_satisfies_equation_l452_45274

theorem unique_number_satisfies_equation : ∃! x : ℝ, (60 + 12) / 3 = (x - 12) * 3 := by
  sorry

end unique_number_satisfies_equation_l452_45274


namespace no_solutions_for_inequality_system_l452_45247

theorem no_solutions_for_inequality_system :
  ¬ ∃ (x y : ℝ), (11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3) ∧ (5 * x + y ≤ -10) := by
  sorry

end no_solutions_for_inequality_system_l452_45247


namespace quadratic_inequality_solution_set_l452_45280

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1 < x ∧ x < 2) :
  ∀ x : ℝ, 2*x^2 + b*x + a < 0 ↔ -1 < x ∧ x < 1/2 :=
sorry

end quadratic_inequality_solution_set_l452_45280


namespace quadratic_function_property_l452_45273

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_at_neg_two : (fun x => a * x^2 + b * x + c) (-2) = a^2
  passes_through_point : a * (-1)^2 + b * (-1) + c = 6

/-- Theorem stating that for a quadratic function with given properties, (a+c)/b = 1/2 -/
theorem quadratic_function_property (f : QuadraticFunction) : (f.a + f.c) / f.b = 1/2 := by
  sorry

end quadratic_function_property_l452_45273


namespace parabolas_intersection_l452_45284

-- Define the parabolas
def parabola1 (a x y : ℝ) : Prop := y = x^2 + x + a
def parabola2 (a x y : ℝ) : Prop := x = 4*y^2 + 3*y + a

-- Define the condition of four intersection points
def has_four_intersections (a : ℝ) : Prop := ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
  (parabola1 a x1 y1 ∧ parabola2 a x1 y1) ∧
  (parabola1 a x2 y2 ∧ parabola2 a x2 y2) ∧
  (parabola1 a x3 y3 ∧ parabola2 a x3 y3) ∧
  (parabola1 a x4 y4 ∧ parabola2 a x4 y4) ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x1 ≠ x3 ∨ y1 ≠ y3) ∧ (x1 ≠ x4 ∨ y1 ≠ y4) ∧
  (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x2 ≠ x4 ∨ y2 ≠ y4) ∧ (x3 ≠ x4 ∨ y3 ≠ y4)

-- Define the range of a
def a_range (a : ℝ) : Prop := (a < -1/2 ∨ (-1/2 < a ∧ a < -7/16))

-- Define the condition for points being concyclic
def concyclic (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  ∃ cx cy r : ℝ, 
    (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
    (x2 - cx)^2 + (y2 - cy)^2 = r^2 ∧
    (x3 - cx)^2 + (y3 - cy)^2 = r^2 ∧
    (x4 - cx)^2 + (y4 - cy)^2 = r^2

-- The main theorem
theorem parabolas_intersection (a : ℝ) :
  has_four_intersections a →
  (a_range a ∧
   ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
     (parabola1 a x1 y1 ∧ parabola2 a x1 y1) ∧
     (parabola1 a x2 y2 ∧ parabola2 a x2 y2) ∧
     (parabola1 a x3 y3 ∧ parabola2 a x3 y3) ∧
     (parabola1 a x4 y4 ∧ parabola2 a x4 y4) ∧
     concyclic x1 y1 x2 y2 x3 y3 x4 y4 ∧
     ∃ cx cy : ℝ, cx = -3/8 ∧ cy = 1/8) :=
by sorry

end parabolas_intersection_l452_45284


namespace cone_surface_area_l452_45297

/-- The surface area of a cone with given height and base area -/
theorem cone_surface_area (h : ℝ) (base_area : ℝ) (h_pos : h > 0) (base_pos : base_area > 0) :
  let r := Real.sqrt (base_area / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  h = 4 ∧ base_area = 9 * Real.pi → Real.pi * r * l + base_area = 24 * Real.pi := by
  sorry

end cone_surface_area_l452_45297


namespace ratio_and_quadratic_equation_solution_l452_45201

theorem ratio_and_quadratic_equation_solution (x y z a : ℤ) : 
  (∃ k : ℚ, x = 4 * k ∧ y = 6 * k ∧ z = 10 * k) →
  y^2 = 40 * a - 20 →
  a = 1 := by
sorry

end ratio_and_quadratic_equation_solution_l452_45201


namespace prob_both_white_one_third_l452_45290

/-- Represents a bag of balls -/
structure Bag where
  white : Nat
  yellow : Nat

/-- Calculates the probability of drawing a white ball from a bag -/
def probWhite (bag : Bag) : Rat :=
  bag.white / (bag.white + bag.yellow)

/-- The probability of drawing white balls from both bags -/
def probBothWhite (bagA bagB : Bag) : Rat :=
  probWhite bagA * probWhite bagB

theorem prob_both_white_one_third :
  let bagA : Bag := { white := 1, yellow := 1 }
  let bagB : Bag := { white := 2, yellow := 1 }
  probBothWhite bagA bagB = 1/3 := by
  sorry

end prob_both_white_one_third_l452_45290


namespace total_toy_count_l452_45220

def toy_count (jerry gabriel jaxon sarah emily : ℕ) : Prop :=
  jerry = gabriel + 8 ∧
  gabriel = 2 * jaxon ∧
  jaxon = 15 ∧
  sarah = jerry - 5 ∧
  sarah = emily + 3 ∧
  emily = 2 * gabriel

theorem total_toy_count :
  ∀ jerry gabriel jaxon sarah emily : ℕ,
  toy_count jerry gabriel jaxon sarah emily →
  jerry + gabriel + jaxon + sarah + emily = 176 :=
by
  sorry

end total_toy_count_l452_45220


namespace winter_clothing_count_l452_45271

theorem winter_clothing_count (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : 
  num_boxes = 4 → scarves_per_box = 2 → mittens_per_box = 6 →
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 32 := by
  sorry

end winter_clothing_count_l452_45271


namespace quadratic_solution_form_l452_45223

theorem quadratic_solution_form (x : ℝ) : 
  (5 * x^2 - 11 * x + 2 = 0) →
  ∃ (m n p : ℕ), 
    x = (m + Real.sqrt n) / p ∧ 
    m = 20 ∧ 
    n = 0 ∧ 
    p = 10 ∧
    m + n + p = 30 ∧
    Nat.gcd m (Nat.gcd n p) = 1 :=
by sorry

end quadratic_solution_form_l452_45223


namespace inequality_proof_l452_45218

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x / (y^2 + z)) + (y / (z^2 + x)) + (z / (x^2 + y)) ≥ 9/4 := by
  sorry

end inequality_proof_l452_45218


namespace brandon_rabbit_catching_l452_45212

/-- The number of squirrels Brandon can catch in an hour -/
def squirrels_per_hour : ℕ := 6

/-- The number of calories in each squirrel -/
def calories_per_squirrel : ℕ := 300

/-- The number of calories in each rabbit -/
def calories_per_rabbit : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits -/
def additional_calories : ℕ := 200

/-- The number of rabbits Brandon can catch in an hour -/
def rabbits_per_hour : ℕ := 2

theorem brandon_rabbit_catching :
  squirrels_per_hour * calories_per_squirrel =
  rabbits_per_hour * calories_per_rabbit + additional_calories :=
by sorry

end brandon_rabbit_catching_l452_45212


namespace parkers_richies_ratio_l452_45254

/-- Given that Parker's share is $50 and the total shared amount is $125,
    prove that the ratio of Parker's share to Richie's share is 2:3. -/
theorem parkers_richies_ratio (parker_share : ℝ) (total_share : ℝ) :
  parker_share = 50 →
  total_share = 125 →
  parker_share < total_share →
  ∃ (a b : ℕ), a = 2 ∧ b = 3 ∧ parker_share / (total_share - parker_share) = a / b :=
by sorry

end parkers_richies_ratio_l452_45254


namespace f_properties_l452_45240

def f (x : ℝ) := -7 * x

theorem f_properties :
  (∀ x y : ℝ, (x > 0 ∧ f x < 0) ∨ (x < 0 ∧ f x > 0)) ∧
  f 1 = -7 ∧
  (∀ x y : ℝ, x < y → f x > f y) :=
sorry

end f_properties_l452_45240


namespace conditional_probability_B_given_A_l452_45272

-- Define the sample space
def Ω : Type := Fin 3 → Fin 3

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A: sum of numbers drawn is 6
def A : Set Ω := {ω : Ω | ω 0 + ω 1 + ω 2 = 5}

-- Define event B: number 2 is drawn three times
def B : Set Ω := {ω : Ω | ∀ i, ω i = 1}

-- State the theorem
theorem conditional_probability_B_given_A :
  P B / P A = 1 / 7 := by sorry

end conditional_probability_B_given_A_l452_45272


namespace rectangle_area_l452_45296

/-- Given a rectangle with perimeter 24 and one side length x (x > 0),
    prove that its area y is equal to (12-x)x -/
theorem rectangle_area (x : ℝ) (hx : x > 0) : 
  let perimeter : ℝ := 24
  let y : ℝ := x * (perimeter / 2 - x)
  y = (12 - x) * x :=
by sorry

end rectangle_area_l452_45296


namespace min_value_inequality_l452_45276

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + b / a) * (1 + 4 * a / b) ≥ 9 := by
  sorry

end min_value_inequality_l452_45276


namespace danjiangkou_tourists_scientific_notation_l452_45217

/-- Converts a positive integer to scientific notation -/
def to_scientific_notation (n : ℕ) : ℚ × ℤ :=
  sorry

theorem danjiangkou_tourists_scientific_notation :
  to_scientific_notation 456000 = (4.56, 5) :=
sorry

end danjiangkou_tourists_scientific_notation_l452_45217


namespace regular_polygon_sides_l452_45256

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 140 → n = 9 :=
by
  sorry

end regular_polygon_sides_l452_45256


namespace calculate_y_investment_y_investment_proof_l452_45204

/-- Calculates the investment amount of partner y in a business partnership --/
theorem calculate_y_investment (x_investment : ℕ) (total_profit : ℕ) (x_profit_share : ℕ) : ℕ :=
  let y_profit_share := total_profit - x_profit_share
  let y_investment := (y_profit_share * x_investment) / x_profit_share
  y_investment

/-- Proves that y's investment is 15000 given the problem conditions --/
theorem y_investment_proof :
  calculate_y_investment 5000 1600 400 = 15000 := by
  sorry

end calculate_y_investment_y_investment_proof_l452_45204


namespace quadratic_factorization_l452_45269

theorem quadratic_factorization (x : ℝ) : 
  x^2 - 6*x - 6 = 0 ↔ (x - 3)^2 = 15 := by sorry

end quadratic_factorization_l452_45269


namespace saturday_sales_77_l452_45249

/-- Represents the number of boxes sold on each day --/
structure DailySales where
  saturday : ℕ
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total sales over 5 days --/
def totalSales (sales : DailySales) : ℕ :=
  sales.saturday + sales.sunday + sales.monday + sales.tuesday + sales.wednesday

/-- Checks if the sales follow the given percentage increases --/
def followsPercentageIncreases (sales : DailySales) : Prop :=
  sales.sunday = (sales.saturday * 3) / 2 ∧
  sales.monday = (sales.sunday * 13) / 10 ∧
  sales.tuesday = (sales.monday * 6) / 5 ∧
  sales.wednesday = (sales.tuesday * 11) / 10

theorem saturday_sales_77 (sales : DailySales) :
  followsPercentageIncreases sales →
  totalSales sales = 720 →
  sales.saturday = 77 := by
  sorry


end saturday_sales_77_l452_45249


namespace quadratic_symmetry_l452_45298

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: For a quadratic function p(x) with axis of symmetry at x = 8.5 and p(-1) = -4, p(18) = -4 -/
theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c (17 - x) = p a b c x) →  -- axis of symmetry at x = 8.5
  p a b c (-1) = -4 →
  p a b c 18 = -4 :=
by sorry

end quadratic_symmetry_l452_45298


namespace greatest_sum_consecutive_integers_l452_45237

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m > n → m * (m + 1) ≥ 500) → n + (n + 1) = 43 :=
by sorry

end greatest_sum_consecutive_integers_l452_45237


namespace parallel_vectors_k_value_l452_45270

/-- Given two vectors a and b in ℝ³, if k * a + b is parallel to 2 * a - b, then k = -2 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 1, 0)) 
    (h2 : b = (-1, 0, -2)) 
    (h_parallel : ∃ (t : ℝ), t • (k • a + b) = 2 • a - b) : 
  k = -2 := by
  sorry

end parallel_vectors_k_value_l452_45270


namespace fraction_sum_inequality_l452_45206

theorem fraction_sum_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
  x / y + y / x ≥ 2 ∧ (x / y + y / x = 2 ↔ x = y) := by
  sorry

end fraction_sum_inequality_l452_45206


namespace toms_age_ratio_l452_45264

theorem toms_age_ratio (T N : ℕ) : 
  (∃ (x y z : ℕ), T = x + y + z) →  -- T is the sum of three children's ages
  (T - N = 2 * ((T - N) - 3 * N)) →  -- N years ago, Tom's age was twice the sum of his children's ages
  T / N = 5 := by
sorry

end toms_age_ratio_l452_45264


namespace f_max_min_on_interval_l452_45258

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = min) ∧
    max = 16 ∧ min = 0 := by
  sorry

end f_max_min_on_interval_l452_45258


namespace inscribed_cylinder_radius_l452_45238

/-- Represents a right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ

/-- The condition that the cylinder's diameter equals its height -/
def cylinder_diameter_equals_height (c : InscribedCylinder) : Prop :=
  2 * c.cylinder_radius = 2 * c.cylinder_radius

/-- The condition that the cone's diameter is 15 -/
def cone_diameter_is_15 (c : InscribedCylinder) : Prop :=
  c.cone_diameter = 15

/-- The condition that the cone's altitude is 15 -/
def cone_altitude_is_15 (c : InscribedCylinder) : Prop :=
  c.cone_altitude = 15

/-- The main theorem: the radius of the inscribed cylinder is 15/4 -/
theorem inscribed_cylinder_radius (c : InscribedCylinder) 
  (h1 : cylinder_diameter_equals_height c)
  (h2 : cone_diameter_is_15 c)
  (h3 : cone_altitude_is_15 c) :
  c.cylinder_radius = 15 / 4 := by
  sorry


end inscribed_cylinder_radius_l452_45238


namespace odd_count_after_ten_operations_l452_45275

/-- Represents the state of the board after n operations -/
structure BoardState (n : ℕ) where
  odd_count : ℕ  -- Number of odd numbers on the board
  total_count : ℕ  -- Total number of numbers on the board

/-- Performs one operation on the board -/
def next_state (state : BoardState n) : BoardState (n + 1) :=
  sorry

/-- Initial state of the board with 0 and 1 -/
def initial_state : BoardState 0 :=
  { odd_count := 1, total_count := 2 }

/-- The state of the board after n operations -/
def board_state (n : ℕ) : BoardState n :=
  match n with
  | 0 => initial_state
  | n + 1 => next_state (board_state n)

theorem odd_count_after_ten_operations :
  (board_state 10).odd_count = 683 :=
sorry

end odd_count_after_ten_operations_l452_45275
