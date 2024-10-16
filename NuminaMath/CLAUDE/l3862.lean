import Mathlib

namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l3862_386226

theorem alcohol_water_ratio (alcohol_fraction water_fraction : ℚ) 
  (h1 : alcohol_fraction = 3/5)
  (h2 : water_fraction = 2/5)
  (h3 : alcohol_fraction + water_fraction = 1) :
  alcohol_fraction / water_fraction = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l3862_386226


namespace NUMINAMATH_CALUDE_slow_clock_catch_up_l3862_386224

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents how many minutes the clock is slow per hour -/
def slow_rate : ℕ := 4

/-- Represents the current time on the slow clock in minutes past 11:00 -/
def current_slow_time : ℕ := 46

/-- Represents the target time on the slow clock in minutes past 11:00 -/
def target_slow_time : ℕ := 60

/-- Theorem stating that it takes 15 minutes of correct time for the slow clock to reach 12:00 -/
theorem slow_clock_catch_up :
  (target_slow_time - current_slow_time) * minutes_per_hour / (minutes_per_hour - slow_rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_slow_clock_catch_up_l3862_386224


namespace NUMINAMATH_CALUDE_unique_m_opens_downwards_l3862_386200

/-- A function f(x) = (m + 1)x^(|m|) that opens downwards -/
def opens_downwards (m : ℝ) : Prop :=
  (abs m = 2) ∧ (m + 1 < 0)

/-- The unique value of m for which the function opens downwards is -2 -/
theorem unique_m_opens_downwards :
  ∃! m : ℝ, opens_downwards m :=
sorry

end NUMINAMATH_CALUDE_unique_m_opens_downwards_l3862_386200


namespace NUMINAMATH_CALUDE_correct_passwords_count_l3862_386216

def total_passwords : ℕ := 10000

def invalid_passwords : ℕ := 10

theorem correct_passwords_count :
  total_passwords - invalid_passwords = 9990 :=
by sorry

end NUMINAMATH_CALUDE_correct_passwords_count_l3862_386216


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3862_386299

theorem partial_fraction_decomposition_product (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (35 * x - 29) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -246 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3862_386299


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3862_386248

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 5 ∧ 
  k = 0 ∧ 
  c = 10 ∧ 
  a = 5 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 10 + 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3862_386248


namespace NUMINAMATH_CALUDE_bens_car_cost_ratio_l3862_386259

theorem bens_car_cost_ratio :
  let old_car_cost : ℚ := 1800
  let new_car_cost : ℚ := 2000 + 1800
  (new_car_cost / old_car_cost) = 19 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bens_car_cost_ratio_l3862_386259


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l3862_386231

theorem sum_of_two_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l3862_386231


namespace NUMINAMATH_CALUDE_server_data_requests_l3862_386293

/-- The number of data requests processed by a server in 24 hours -/
def data_requests_per_day (requests_per_minute : ℕ) : ℕ :=
  requests_per_minute * (24 * 60)

/-- Theorem stating that a server processing 15,000 data requests per minute
    will process 21,600,000 data requests in 24 hours -/
theorem server_data_requests :
  data_requests_per_day 15000 = 21600000 := by
  sorry

end NUMINAMATH_CALUDE_server_data_requests_l3862_386293


namespace NUMINAMATH_CALUDE_set_A_properties_l3862_386273

/-- Property P: For any i, j (1 ≤ i ≤ j ≤ n), at least one of aᵢaⱼ and aⱼ/aᵢ belongs to A -/
def property_P (A : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ A → y ∈ A → x ≤ y → (x * y ∈ A ∨ y / x ∈ A)

theorem set_A_properties {n : ℕ} (A : Set ℝ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_A : A = {x | ∃ i, i ∈ Finset.range n ∧ x = a i})
  (h_sorted : ∀ i j, i < j → j < n → a i < a j)
  (h_P : property_P A) :
  (a 0 = 1) ∧ 
  ((Finset.range n).sum a / (Finset.range n).sum (λ i => (a i)⁻¹) = a (n - 1)) ∧
  (n = 5 → ∃ r : ℝ, ∀ i, i < 4 → a (i + 1) = r * a i) :=
by sorry

end NUMINAMATH_CALUDE_set_A_properties_l3862_386273


namespace NUMINAMATH_CALUDE_cubic_function_m_value_l3862_386241

theorem cubic_function_m_value (d e f g m : ℤ) :
  let g : ℝ → ℝ := λ x => (d : ℝ) * x^3 + (e : ℝ) * x^2 + (f : ℝ) * x + (g : ℝ)
  g 1 = 0 ∧
  70 < g 5 ∧ g 5 < 80 ∧
  120 < g 6 ∧ g 6 < 130 ∧
  10000 * (m : ℝ) < g 50 ∧ g 50 < 10000 * ((m + 1) : ℝ) →
  m = 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_m_value_l3862_386241


namespace NUMINAMATH_CALUDE_count_100_digit_even_numbers_l3862_386247

/-- A function that represents the count of n-digit even numbers where each digit is 0, 1, or 3 -/
def countEvenNumbers (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * 3^(n - 2)

/-- Theorem stating that the count of 100-digit even numbers where each digit is 0, 1, or 3 is 2 * 3^98 -/
theorem count_100_digit_even_numbers :
  countEvenNumbers 100 = 2 * 3^98 := by
  sorry


end NUMINAMATH_CALUDE_count_100_digit_even_numbers_l3862_386247


namespace NUMINAMATH_CALUDE_hockey_league_games_l3862_386281

/-- The number of games played in a hockey league season -/
def number_of_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 19 teams, where each team faces every other team 10 times, 
    the total number of games played in the season is 1710 -/
theorem hockey_league_games : number_of_games 19 10 = 1710 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l3862_386281


namespace NUMINAMATH_CALUDE_max_product_sum_2004_l3862_386294

theorem max_product_sum_2004 :
  ∃ (a b : ℤ), a + b = 2004 ∧
  ∀ (x y : ℤ), x + y = 2004 → x * y ≤ a * b ∧
  a * b = 1004004 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_2004_l3862_386294


namespace NUMINAMATH_CALUDE_numerical_puzzle_solution_l3862_386220

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that checks if two digits are different -/
def differentDigits (a b : ℕ) : Prop := a ≠ b ∧ a < 10 ∧ b < 10

/-- The main theorem stating the solution to the numerical puzzle -/
theorem numerical_puzzle_solution :
  ∀ (a b : ℕ), differentDigits a b →
    isTwoDigit (10 * a + b) →
    (10 * a + b = b ^ (10 * a + b)) ↔ 
    ((a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_numerical_puzzle_solution_l3862_386220


namespace NUMINAMATH_CALUDE_max_value_x_cubed_minus_y_cubed_l3862_386239

theorem max_value_x_cubed_minus_y_cubed (x y : ℝ) (h : x^2 + y^2 = x + y) :
  ∃ (M : ℝ), M = 1 ∧ x^3 - y^3 ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = x₀ + y₀ ∧ x₀^3 - y₀^3 = M :=
sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_minus_y_cubed_l3862_386239


namespace NUMINAMATH_CALUDE_periodic_odd_function_property_l3862_386234

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_property (f : ℝ → ℝ) (a : ℝ) 
    (h_periodic : is_periodic f 3)
    (h_odd : is_odd f)
    (h_f1 : f 1 > 1)
    (h_f2 : f 2 = a) :
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_property_l3862_386234


namespace NUMINAMATH_CALUDE_symmetric_curve_is_correct_l3862_386228

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the symmetric curve
def symmetric_curve (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 1

-- Theorem statement
theorem symmetric_curve_is_correct : 
  ∀ (x y x' y' : ℝ), 
    given_circle x y → 
    symmetry_line ((x + x') / 2) ((y + y') / 2) → 
    symmetric_curve x' y' :=
sorry

end NUMINAMATH_CALUDE_symmetric_curve_is_correct_l3862_386228


namespace NUMINAMATH_CALUDE_square_field_area_l3862_386263

/-- Prove that a square field with the given conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 1.4 →
  total_cost = 932.4 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    total_cost = wire_cost_per_meter * (4 * side_length - num_gates * gate_width) ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l3862_386263


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l3862_386279

theorem baseball_card_value_decrease (initial_value : ℝ) (first_year_decrease : ℝ) (total_decrease : ℝ) 
  (h1 : first_year_decrease = 60)
  (h2 : total_decrease = 64)
  (h3 : initial_value > 0) :
  let value_after_first_year := initial_value * (1 - first_year_decrease / 100)
  let final_value := initial_value * (1 - total_decrease / 100)
  let second_year_decrease := (value_after_first_year - final_value) / value_after_first_year * 100
  second_year_decrease = 10 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l3862_386279


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3862_386298

/-- Given a triangle ABC where angle A is 60° and side a is 4, 
    the maximum perimeter of the triangle is 12. -/
theorem triangle_max_perimeter (b c : ℝ) : 
  let A : ℝ := 60 * π / 180  -- Convert 60° to radians
  let a : ℝ := 4
  b > 0 → c > 0 →   -- Ensure positive side lengths
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- Cosine theorem
  a + b + c ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3862_386298


namespace NUMINAMATH_CALUDE_cuboid_breadth_proof_l3862_386270

/-- The surface area of a cuboid given its length, width, and height. -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The breadth of a cuboid with surface area 700 m², length 12 m, and height 7 m is 14 m. -/
theorem cuboid_breadth_proof :
  ∃ w : ℝ, cuboidSurfaceArea 12 w 7 = 700 ∧ w = 14 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_breadth_proof_l3862_386270


namespace NUMINAMATH_CALUDE_total_students_l3862_386287

/-- Represents the age groups in the school -/
inductive AgeGroup
  | Below8
  | Exactly8
  | Between9And10
  | Above10

/-- Represents the school with its student distribution -/
structure School where
  totalStudents : ℕ
  ageDistribution : AgeGroup → ℚ
  exactly8Count : ℕ

/-- The conditions of the problem -/
def schoolConditions (s : School) : Prop :=
  s.ageDistribution AgeGroup.Below8 = 1/5 ∧
  s.ageDistribution AgeGroup.Exactly8 = 1/4 ∧
  s.ageDistribution AgeGroup.Between9And10 = 7/20 ∧
  s.ageDistribution AgeGroup.Above10 = 1/5 ∧
  s.exactly8Count = 15

/-- The theorem to prove -/
theorem total_students (s : School) (h : schoolConditions s) : s.totalStudents = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l3862_386287


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3862_386244

/-- Given vectors m and n in ℝ², prove that if m + n is perpendicular to m - n, then t = -3 -/
theorem vector_perpendicular (t : ℝ) : 
  let m : Fin 2 → ℝ := ![t + 1, 1]
  let n : Fin 2 → ℝ := ![t + 2, 2]
  (m + n) • (m - n) = 0 → t = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3862_386244


namespace NUMINAMATH_CALUDE_freds_change_is_correct_l3862_386233

/-- The amount of change Fred received after buying movie tickets and borrowing a movie -/
def freds_change (ticket_price : ℚ) (num_tickets : ℕ) (borrowed_movie_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price)

/-- Theorem: Fred's change is $1.37 -/
theorem freds_change_is_correct : 
  freds_change (92/100 + 5) 2 (79/100 + 6) 20 = 37/100 + 1 :=
by sorry

end NUMINAMATH_CALUDE_freds_change_is_correct_l3862_386233


namespace NUMINAMATH_CALUDE_inequality_proof_l3862_386260

theorem inequality_proof (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  2 * m + 1 / (m^2 - 2*m*n + n^2) ≥ 2 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3862_386260


namespace NUMINAMATH_CALUDE_triangle_side_length_l3862_386249

/-- Theorem: In a triangle ABC, if c + b = 12, A = 60°, and B = 30°, then c = 8 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  c + b = 12 → A = 60 → B = 30 → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3862_386249


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l3862_386269

/-- The decimal representation 0.142857142857... as a real number -/
def a : ℚ := 142857 / 999999

/-- The decimal representation 0.285714285714... as a real number -/
def b : ℚ := 285714 / 999999

/-- Theorem stating that the ratio of the two repeating decimals is 1/2 -/
theorem repeating_decimal_ratio : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l3862_386269


namespace NUMINAMATH_CALUDE_gcd_of_product_form_l3862_386267

def product_form (a b c d : ℤ) : ℤ :=
  (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b)

theorem gcd_of_product_form :
  ∃ (g : ℤ), g > 0 ∧ 
  (∀ (a b c d : ℤ), g ∣ product_form a b c d) ∧
  (∀ (h : ℤ), h > 0 → (∀ (a b c d : ℤ), h ∣ product_form a b c d) → h ∣ g) ∧
  g = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_product_form_l3862_386267


namespace NUMINAMATH_CALUDE_solve_equation_l3862_386209

theorem solve_equation (t x : ℝ) : 2*t + 2*x - t - 3*x + 4*x + 2*t = 30 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3862_386209


namespace NUMINAMATH_CALUDE_revenue_maximized_at_five_l3862_386256

def revenue (x : ℝ) : ℝ := (400 - 20*x) * (50 + 5*x)

theorem revenue_maximized_at_five :
  ∃ (max : ℝ), revenue 5 = max ∧ ∀ (x : ℝ), revenue x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_five_l3862_386256


namespace NUMINAMATH_CALUDE_zero_real_necessary_not_sufficient_for_purely_imaginary_l3862_386252

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

theorem zero_real_necessary_not_sufficient_for_purely_imaginary :
  ∃ (a b : ℝ), (isPurelyImaginary (Complex.mk a b) → a = 0) ∧
                ¬(a = 0 → isPurelyImaginary (Complex.mk a b)) :=
by sorry

end NUMINAMATH_CALUDE_zero_real_necessary_not_sufficient_for_purely_imaginary_l3862_386252


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3862_386235

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3862_386235


namespace NUMINAMATH_CALUDE_min_cross_section_area_and_volume_ratio_l3862_386264

/-- A regular triangular pyramid inscribed in a sphere -/
structure RegularTriangularPyramid (R : ℝ) where
  /-- The radius of the circumscribing sphere -/
  radius : ℝ
  /-- The height of the pyramid -/
  height : ℝ
  /-- The height is 4R/3 -/
  height_eq : height = 4 * R / 3

/-- A cross-section of the pyramid passing through a median of its base -/
structure CrossSection (R : ℝ) (pyramid : RegularTriangularPyramid R) where
  /-- The area of the cross-section -/
  area : ℝ
  /-- The ratio of the volumes of the two parts divided by the cross-section -/
  volume_ratio : ℚ × ℚ

/-- The theorem stating the minimum area of the cross-section and the volume ratio -/
theorem min_cross_section_area_and_volume_ratio (R : ℝ) (pyramid : RegularTriangularPyramid R) :
  ∃ (cs : CrossSection R pyramid),
    cs.area = 2 * Real.sqrt 2 / Real.sqrt 33 * R^2 ∧
    cs.volume_ratio = (3, 19) ∧
    ∀ (other_cs : CrossSection R pyramid), cs.area ≤ other_cs.area :=
sorry

end NUMINAMATH_CALUDE_min_cross_section_area_and_volume_ratio_l3862_386264


namespace NUMINAMATH_CALUDE_units_sold_to_A_is_three_l3862_386218

/-- Represents the number of units sold to Customer A in a phone store scenario. -/
def units_sold_to_A (total_phones defective_phones units_sold_to_B units_sold_to_C : ℕ) : ℕ :=
  total_phones - defective_phones - units_sold_to_B - units_sold_to_C

/-- Theorem stating that given the specific conditions of the problem, 
    the number of units sold to Customer A is 3. -/
theorem units_sold_to_A_is_three :
  units_sold_to_A 20 5 5 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_sold_to_A_is_three_l3862_386218


namespace NUMINAMATH_CALUDE_six_solutions_l3862_386201

/-- The number of ordered pairs of positive integers (m,n) satisfying 6/m + 3/n = 1 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (m, n) := p
    m > 0 ∧ n > 0 ∧ 6 * n + 3 * m = m * n) (Finset.product (Finset.range 25) (Finset.range 22))).card

/-- The theorem stating that there are exactly 6 solutions -/
theorem six_solutions : solution_count = 6 := by
  sorry


end NUMINAMATH_CALUDE_six_solutions_l3862_386201


namespace NUMINAMATH_CALUDE_system_four_solutions_l3862_386217

theorem system_four_solutions (a : ℝ) (ha : a > 0) :
  ∃! (solutions : Finset (ℝ × ℝ)), 
    solutions.card = 4 ∧
    ∀ (x y : ℝ), (x, y) ∈ solutions ↔ 
      (y = a * x^2 ∧ y^2 + 3 = x^2 + 4*y) :=
sorry

end NUMINAMATH_CALUDE_system_four_solutions_l3862_386217


namespace NUMINAMATH_CALUDE_units_digit_of_5_to_4_l3862_386282

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_5_to_4 : unitsDigit (5^4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_5_to_4_l3862_386282


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l3862_386280

theorem not_p_necessary_not_sufficient_for_not_p_or_q (p q : Prop) :
  (∀ (h : ¬p ∨ q), ¬p) ∧ 
  ¬(∀ (h : ¬p), ¬(p ∨ q)) :=
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l3862_386280


namespace NUMINAMATH_CALUDE_inconsistent_school_population_l3862_386296

theorem inconsistent_school_population (total_students : Real) 
  (boy_percentage : Real) (representative_students : Nat) : 
  total_students = 113.38934190276818 → 
  boy_percentage = 0.70 → 
  representative_students = 90 → 
  (representative_students : Real) / (total_students * boy_percentage) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_school_population_l3862_386296


namespace NUMINAMATH_CALUDE_sqrt_19_minus_1_squared_plus_2x_plus_2_l3862_386227

theorem sqrt_19_minus_1_squared_plus_2x_plus_2 :
  let x : ℝ := Real.sqrt 19 - 1
  x^2 + 2*x + 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_sqrt_19_minus_1_squared_plus_2x_plus_2_l3862_386227


namespace NUMINAMATH_CALUDE_consecutive_discounts_l3862_386265

theorem consecutive_discounts (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_discount := original_price * (1 - 0.3)
  let price_after_second_discount := price_after_first_discount * (1 - 0.2)
  let final_price := price_after_second_discount * (1 - 0.1)
  (original_price - final_price) / original_price = 0.496 := by
sorry

end NUMINAMATH_CALUDE_consecutive_discounts_l3862_386265


namespace NUMINAMATH_CALUDE_fraction_power_simplification_l3862_386274

theorem fraction_power_simplification :
  9 * (1 / 7)^4 = 9 / 2401 :=
by sorry

end NUMINAMATH_CALUDE_fraction_power_simplification_l3862_386274


namespace NUMINAMATH_CALUDE_incircle_touch_point_distance_special_triangle_incircle_touch_point_distance_l3862_386215

/-- Given a triangle with sides a, b, c, and an incircle that touches side c at point P,
    the distance from one endpoint of side c to P is (a + b + c) / 2 - b -/
theorem incircle_touch_point_distance (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  let s := (a + b + c) / 2
  (s - b) = ((a + b + c) / 2) - b :=
by sorry

/-- In a triangle with sides 4, 5, and 6, the distance from one vertex to the point 
    where the incircle touches the opposite side is 2.5 -/
theorem special_triangle_incircle_touch_point_distance :
  let a := 4
  let b := 5
  let c := 6
  let s := (a + b + c) / 2
  (s - b) = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_incircle_touch_point_distance_special_triangle_incircle_touch_point_distance_l3862_386215


namespace NUMINAMATH_CALUDE_square_inequality_l3862_386207

theorem square_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l3862_386207


namespace NUMINAMATH_CALUDE_negation_of_existential_l3862_386289

theorem negation_of_existential (p : Prop) :
  (¬∃ (x : ℝ), x = Real.sin x) ↔ (∀ (x : ℝ), x ≠ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_l3862_386289


namespace NUMINAMATH_CALUDE_samara_oil_spending_l3862_386275

/-- The amount Alberto spent on his car -/
def alberto_spent : ℕ := 2457

/-- The amount Samara spent on tires -/
def samara_tires : ℕ := 467

/-- The amount Samara spent on detailing -/
def samara_detailing : ℕ := 79

/-- The difference between Alberto's and Samara's spending -/
def spending_difference : ℕ := 1886

/-- The amount Samara spent on oil -/
def samara_oil : ℕ := 25

theorem samara_oil_spending : 
  alberto_spent = samara_oil + samara_tires + samara_detailing + spending_difference :=
by sorry

end NUMINAMATH_CALUDE_samara_oil_spending_l3862_386275


namespace NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l3862_386230

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (student : U → Prop)
variable (club_member : U → Prop)
variable (fraternity_member : U → Prop)
variable (honest : U → Prop)

-- State the theorem
theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, club_member x → student x)
  (h2 : ∀ x, club_member x → ¬honest x)
  (h3 : ∀ x, fraternity_member x → honest x) :
  ∀ x, fraternity_member x → ¬club_member x :=
by
  sorry


end NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l3862_386230


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3862_386236

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3862_386236


namespace NUMINAMATH_CALUDE_tinas_mile_time_l3862_386278

theorem tinas_mile_time (tony tina tom : ℝ) 
  (h1 : tony = tina / 2)  -- Tony is twice as fast as Tina
  (h2 : tina = 3 * tom)   -- Tina is one-third as fast as Tom
  (h3 : tony + tina + tom = 11) -- Sum of their times is 11 minutes
  : tina = 6 := by
  sorry

end NUMINAMATH_CALUDE_tinas_mile_time_l3862_386278


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_eq_l3862_386288

theorem product_of_solutions_abs_eq : ∃ (a b : ℝ), 
  (∀ x : ℝ, (|x| = 3 * (|x| - 4)) ↔ (x = a ∨ x = b)) ∧ (a * b = -36) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_eq_l3862_386288


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3862_386292

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3862_386292


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l3862_386286

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 48) 
  (h_gcd : Nat.gcd x y = 8) : 
  x * y = 384 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l3862_386286


namespace NUMINAMATH_CALUDE_sonika_deposit_l3862_386238

/-- Calculates the final amount after simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem sonika_deposit :
  ∀ (P R : ℝ),
  simpleInterest P (R / 100) 3 = 10200 →
  simpleInterest P ((R + 2) / 100) 3 = 10680 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_sonika_deposit_l3862_386238


namespace NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l3862_386213

-- Define a color type
inductive Color
| White
| Black

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  is_equilateral : sorry

-- Theorem statement
theorem exists_monochromatic_equilateral_triangle :
  ∃ (t : EquilateralTriangle), 
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l3862_386213


namespace NUMINAMATH_CALUDE_library_book_sorting_l3862_386253

theorem library_book_sorting (damaged : ℕ) (obsolete : ℕ) : 
  obsolete = 6 * damaged - 8 →
  damaged + obsolete = 69 →
  damaged = 11 := by
sorry

end NUMINAMATH_CALUDE_library_book_sorting_l3862_386253


namespace NUMINAMATH_CALUDE_complex_number_proof_l3862_386261

theorem complex_number_proof : 
  ∀ (z : ℂ), (Complex.im ((1 + 2*Complex.I) * z) = 0) → (Complex.abs z = Real.sqrt 5) → 
  (z = 1 - 2*Complex.I ∨ z = -1 + 2*Complex.I) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_proof_l3862_386261


namespace NUMINAMATH_CALUDE_erased_length_is_24cm_l3862_386277

/-- The length of the erased portion of a line --/
def erased_length (original_length : ℝ) (final_length : ℝ) : ℝ :=
  original_length - final_length

/-- Theorem: The erased length is 24 cm when the original length is 1 m and the final length is 76 cm --/
theorem erased_length_is_24cm :
  erased_length 100 76 = 24 := by
  sorry

#check erased_length_is_24cm

end NUMINAMATH_CALUDE_erased_length_is_24cm_l3862_386277


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3862_386245

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x + y > 1 ∧ x^2 + y^2 ≤ 1) ∧ 
  (∃ u v : ℝ, u^2 + v^2 > 1 ∧ u + v ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3862_386245


namespace NUMINAMATH_CALUDE_square_sum_and_product_l3862_386205

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l3862_386205


namespace NUMINAMATH_CALUDE_sequence_a_bounds_l3862_386212

def sequence_a : ℕ → ℚ
  | 0     => 1/2
  | (n+1) => sequence_a n + (1 / (n+1)^2) * (sequence_a n)^2

theorem sequence_a_bounds (n : ℕ) : 
  1 - 1 / (2^(n+1)) ≤ sequence_a n ∧ sequence_a n < 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_bounds_l3862_386212


namespace NUMINAMATH_CALUDE_power_equality_l3862_386266

theorem power_equality (x y : ℕ) (h1 : 8^x = 2^y) (h2 : x = 3) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3862_386266


namespace NUMINAMATH_CALUDE_exists_surjective_function_with_property_l3862_386204

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f (x + y) - f x - f y) ∈ ({0, 1} : Set ℝ)

-- State the theorem
theorem exists_surjective_function_with_property :
  ∃ f : ℝ → ℝ, Function.Surjective f ∧ has_property f :=
sorry

end NUMINAMATH_CALUDE_exists_surjective_function_with_property_l3862_386204


namespace NUMINAMATH_CALUDE_circle_area_to_circumference_l3862_386285

theorem circle_area_to_circumference (A : ℝ) (h : A = 196 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ A = Real.pi * r^2 ∧ 2 * Real.pi * r = 28 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_to_circumference_l3862_386285


namespace NUMINAMATH_CALUDE_investment_income_l3862_386225

theorem investment_income
  (total_investment : ℝ)
  (first_investment : ℝ)
  (first_rate : ℝ)
  (second_rate : ℝ)
  (h1 : total_investment = 8000)
  (h2 : first_investment = 3000)
  (h3 : first_rate = 0.085)
  (h4 : second_rate = 0.064) :
  first_investment * first_rate + (total_investment - first_investment) * second_rate = 575 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_l3862_386225


namespace NUMINAMATH_CALUDE_circle_area_when_equal_to_circumference_l3862_386262

/-- Given a circle where the circumference and area are numerically equal,
    and the diameter is 4, prove that the area is 4π. -/
theorem circle_area_when_equal_to_circumference (r : ℝ) : 
  2 * Real.pi * r = Real.pi * r^2 →   -- Circumference equals area
  4 = 2 * r →                         -- Diameter is 4
  Real.pi * r^2 = 4 * Real.pi :=      -- Area is 4π
by
  sorry

#check circle_area_when_equal_to_circumference

end NUMINAMATH_CALUDE_circle_area_when_equal_to_circumference_l3862_386262


namespace NUMINAMATH_CALUDE_edward_money_problem_l3862_386210

theorem edward_money_problem (initial spent received final : ℤ) :
  spent = 17 →
  received = 10 →
  final = 7 →
  initial - spent + received = final →
  initial = 14 := by
sorry

end NUMINAMATH_CALUDE_edward_money_problem_l3862_386210


namespace NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_ln_a_negative_l3862_386255

theorem a_less_than_one_necessary_not_sufficient_for_ln_a_negative :
  (∀ a : ℝ, (Real.log a < 0) → (a < 1)) ∧
  (∃ a : ℝ, a < 1 ∧ ¬(Real.log a < 0)) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_ln_a_negative_l3862_386255


namespace NUMINAMATH_CALUDE_equation_solutions_l3862_386254

theorem equation_solutions : 
  {x : ℝ | (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 
             1 / ((x - 5) * (x - 7)) + 1 / ((x - 7) * (x - 9)) = 1 / 8)} = {13, -3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3862_386254


namespace NUMINAMATH_CALUDE_max_abc_value_l3862_386202

theorem max_abc_value (a b c : ℕ+) 
  (h1 : a * b + b * c = 518)
  (h2 : a * b - a * c = 360) :
  ∀ x y z : ℕ+, x * y * z ≤ a * b * c → x * y + y * z = 518 → x * y - x * z = 360 → 
  a * b * c = 1008 := by
sorry

end NUMINAMATH_CALUDE_max_abc_value_l3862_386202


namespace NUMINAMATH_CALUDE_braking_velocities_l3862_386237

/-- The displacement function representing the braking system -/
def s (t : ℝ) : ℝ := -3 * t^3 + t^2 + 20

/-- The velocity function (derivative of displacement) -/
def v (t : ℝ) : ℝ := -9 * t^2 + 2 * t

/-- Theorem stating the average and instantaneous velocities during braking -/
theorem braking_velocities :
  (∀ t ∈ Set.Icc 0 2, s t ≥ 0) →  -- Braking completes within 2 seconds
  ((s 1 - s 0) / 1 = -2) ∧        -- Average velocity in first second
  ((s 2 - s 1) / 1 = -18) ∧       -- Average velocity between 1 and 2 seconds
  (v 1 = -7)                      -- Instantaneous velocity at 1 second
:= by sorry

end NUMINAMATH_CALUDE_braking_velocities_l3862_386237


namespace NUMINAMATH_CALUDE_least_divisible_by_three_l3862_386243

theorem least_divisible_by_three (x : ℕ) : (∃ y : ℕ, y > 0 ∧ 23 * y % 3 = 0) → x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_three_l3862_386243


namespace NUMINAMATH_CALUDE_painting_wings_count_l3862_386257

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  total_wings : Nat
  artifacts_per_wing : Nat
  large_painting_wings : Nat
  small_painting_wings : Nat
  paintings_per_small_wing : Nat

/-- The number of wings dedicated to paintings in the museum -/
def painting_wings (m : Museum) : Nat :=
  m.large_painting_wings + m.small_painting_wings

/-- The number of wings dedicated to artifacts in the museum -/
def artifact_wings (m : Museum) : Nat :=
  m.total_wings - painting_wings m

/-- The total number of paintings in the museum -/
def total_paintings (m : Museum) : Nat :=
  m.large_painting_wings + m.small_painting_wings * m.paintings_per_small_wing

/-- The total number of artifacts in the museum -/
def total_artifacts (m : Museum) : Nat :=
  m.artifacts_per_wing * artifact_wings m

theorem painting_wings_count (m : Museum)
  (h1 : m.total_wings = 8)
  (h2 : total_artifacts m = 4 * total_paintings m)
  (h3 : m.large_painting_wings = 1)
  (h4 : m.small_painting_wings = 2)
  (h5 : m.paintings_per_small_wing = 12)
  (h6 : m.artifacts_per_wing = 20) :
  painting_wings m = 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_wings_count_l3862_386257


namespace NUMINAMATH_CALUDE_race_pace_cristina_pace_l3862_386276

/-- The race between Nicky and Cristina -/
theorem race_pace (nicky_pace : ℝ) (race_time : ℝ) (head_start : ℝ) : ℝ :=
  let nicky_distance := nicky_pace * race_time
  let cristina_distance := nicky_distance + head_start
  cristina_distance / race_time

/-- Cristina's pace in the race -/
theorem cristina_pace : race_pace 3 36 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_race_pace_cristina_pace_l3862_386276


namespace NUMINAMATH_CALUDE_inequalities_proof_l3862_386214

theorem inequalities_proof (a b : ℝ) (h : 1/a > 1/b ∧ 1/b > 0) : 
  a^3 < b^3 ∧ Real.sqrt b - Real.sqrt a < Real.sqrt (b - a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3862_386214


namespace NUMINAMATH_CALUDE_min_value_expression_l3862_386232

theorem min_value_expression (a b : ℝ) (h1 : a * b - 2 * a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 2 * x - y + 1 = 0 → x > 1 → (x + 3) * (y + 2) ≥ 25 ∧
  ∃ x y : ℝ, x * y - 2 * x - y + 1 = 0 ∧ x > 1 ∧ (x + 3) * (y + 2) = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3862_386232


namespace NUMINAMATH_CALUDE_polynomial_factors_l3862_386283

theorem polynomial_factors (x : ℝ) : 
  ∃ (a b c : ℝ), 8*x^3 + 14*x^2 - 17*x + 6 = (x + 1/2) * (x - 2) * (a*x + b) ∧ c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factors_l3862_386283


namespace NUMINAMATH_CALUDE_roots_and_coefficients_l3862_386250

theorem roots_and_coefficients (θ : Real) (m : Real) :
  0 < θ ∧ θ < 2 * Real.pi →
  (2 * Real.sin θ ^ 2 - (Real.sqrt 3 + 1) * Real.sin θ + m = 0) ∧
  (2 * Real.cos θ ^ 2 - (Real.sqrt 3 + 1) * Real.cos θ + m = 0) →
  (Real.sin θ ^ 2 / (Real.sin θ - Real.cos θ) + Real.cos θ ^ 2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2) ∧
  (m = Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_roots_and_coefficients_l3862_386250


namespace NUMINAMATH_CALUDE_consecutive_divisible_numbers_l3862_386211

theorem consecutive_divisible_numbers :
  ∃ (n : ℕ),
    (5 ∣ n) ∧
    (4 ∣ n + 1) ∧
    (3 ∣ n + 2) ∧
    (∀ (m : ℕ), (5 ∣ m) ∧ (4 ∣ m + 1) ∧ (3 ∣ m + 2) → n ≤ m) ∧
    n = 55 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_divisible_numbers_l3862_386211


namespace NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l3862_386219

theorem square_fraction_is_perfect_square (a b k : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : k > 0) 
  (h4 : (a^2 + b^2 : ℕ) = k * (a * b + 1)) : 
  ∃ (n : ℕ), k = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l3862_386219


namespace NUMINAMATH_CALUDE_total_combinations_eq_twelve_l3862_386297

/-- The number of paint colors available. -/
def num_colors : ℕ := 4

/-- The number of painting methods available. -/
def num_methods : ℕ := 3

/-- The total number of combinations of paint color and painting method. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 12. -/
theorem total_combinations_eq_twelve : total_combinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_eq_twelve_l3862_386297


namespace NUMINAMATH_CALUDE_square_difference_equals_24_l3862_386251

theorem square_difference_equals_24 (x y : ℝ) (h1 : x + y = 4) (h2 : x - y = 6) :
  x^2 - y^2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_24_l3862_386251


namespace NUMINAMATH_CALUDE_complex_magnitude_l3862_386229

theorem complex_magnitude (z : ℂ) (h : (z + 2) / (z - 2) = Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3862_386229


namespace NUMINAMATH_CALUDE_frog_riverbank_probability_l3862_386258

/-- The probability of reaching the riverbank from stone N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- The number of stones -/
def num_stones : ℕ := 7

theorem frog_riverbank_probability :
  -- The frog starts on stone 2
  -- There are 7 stones labeled from 0 to 6
  -- For stone N (0 < N < 6), the frog jumps to N-1 with probability N/6 and to N+1 with probability 1 - N/6
  -- If the frog reaches stone 0, it falls into the water (probability 0)
  -- If the frog reaches stone 6, it safely reaches the riverbank (probability 1)
  (∀ N, 0 < N → N < 6 → P N = (N / 6 : ℝ) * P (N - 1) + (1 - N / 6 : ℝ) * P (N + 1)) →
  P 0 = 0 →
  P 6 = 1 →
  P 2 = 4/9 :=
sorry

end NUMINAMATH_CALUDE_frog_riverbank_probability_l3862_386258


namespace NUMINAMATH_CALUDE_root_product_l3862_386240

theorem root_product (r b c : ℝ) : 
  r^2 = r + 1 → r^6 = b*r + c → b*c = 40 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l3862_386240


namespace NUMINAMATH_CALUDE_venus_meal_cost_is_35_l3862_386290

/-- The cost per meal at Venus Hall -/
def venus_meal_cost : ℚ := 35

/-- The room rental cost at Caesar's -/
def caesars_rental : ℚ := 800

/-- The cost per meal at Caesar's -/
def caesars_meal_cost : ℚ := 30

/-- The room rental cost at Venus Hall -/
def venus_rental : ℚ := 500

/-- The number of guests at which the total costs are equal -/
def num_guests : ℚ := 60

theorem venus_meal_cost_is_35 :
  caesars_rental + caesars_meal_cost * num_guests =
  venus_rental + venus_meal_cost * num_guests := by
  sorry

end NUMINAMATH_CALUDE_venus_meal_cost_is_35_l3862_386290


namespace NUMINAMATH_CALUDE_triangle_centroid_distances_l3862_386246

/-- Given a triangle DEF with centroid G, prove that if the sum of squared distances
    from G to the vertices is 72, then the sum of squared side lengths is 216. -/
theorem triangle_centroid_distances (D E F G : ℝ × ℝ) : 
  G = ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3) →  -- G is the centroid
  (G.1 - D.1)^2 + (G.2 - D.2)^2 +    -- GD^2
  (G.1 - E.1)^2 + (G.2 - E.2)^2 +    -- GE^2
  (G.1 - F.1)^2 + (G.2 - F.2)^2 = 72 →  -- GF^2
  (D.1 - E.1)^2 + (D.2 - E.2)^2 +    -- DE^2
  (D.1 - F.1)^2 + (D.2 - F.2)^2 +    -- DF^2
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 216  -- EF^2
:= by sorry

end NUMINAMATH_CALUDE_triangle_centroid_distances_l3862_386246


namespace NUMINAMATH_CALUDE_cathys_money_ratio_is_two_to_one_l3862_386291

/-- The ratio of the amount Cathy's mom sent her to the amount her dad sent her -/
def cathys_money_ratio (initial_amount dad_amount mom_amount final_amount : ℚ) : ℚ :=
  mom_amount / dad_amount

/-- Proves that the ratio of the amount Cathy's mom sent her to the amount her dad sent her is 2:1 -/
theorem cathys_money_ratio_is_two_to_one 
  (initial_amount : ℚ) 
  (dad_amount : ℚ) 
  (mom_amount : ℚ) 
  (final_amount : ℚ) 
  (h1 : initial_amount = 12)
  (h2 : dad_amount = 25)
  (h3 : final_amount = 87)
  (h4 : initial_amount + dad_amount + mom_amount = final_amount) :
  cathys_money_ratio initial_amount dad_amount mom_amount final_amount = 2 := by
sorry

#eval cathys_money_ratio 12 25 50 87

end NUMINAMATH_CALUDE_cathys_money_ratio_is_two_to_one_l3862_386291


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3862_386208

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_dot_product : m * 1 + 1 * (n - 1) = 0) :
  ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_dot : x * 1 + 1 * (y - 1) = 0), 
    (1 / m + 1 / n ≥ 1 / x + 1 / y) ∧ (1 / x + 1 / y = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3862_386208


namespace NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l3862_386206

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13]

def remove_number (l : List ℕ) (n : ℕ) : List ℕ :=
  l.filter (· ≠ n)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem remove_one_gives_average_seven_point_five :
  average (remove_number original_list 1) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l3862_386206


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3862_386295

-- Define sets A and B
def A : Set ℝ := {x | (x - 1) * (x - 3) < 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3862_386295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l3862_386284

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2015th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a5 : a 5 = 6) :
  a 2015 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l3862_386284


namespace NUMINAMATH_CALUDE_min_value_product_sum_l3862_386223

theorem min_value_product_sum (p q r s t u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) 
  (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    0 < p' ∧ 0 < q' ∧ 0 < r' ∧ 0 < s' ∧
    0 < t' ∧ 0 < u' ∧ 0 < v' ∧ 0 < w' ∧
    p' * q' * r' * s' = 16 ∧
    t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_product_sum_l3862_386223


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3862_386221

/-- Represents the number of students in each grade --/
structure StudentCounts where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ

/-- Calculates the total number of students --/
def total_students (counts : StudentCounts) : ℕ :=
  counts.freshman + counts.sophomore + counts.senior

/-- Calculates the sample size based on the number of sampled freshmen --/
def sample_size (counts : StudentCounts) (sampled_freshmen : ℕ) : ℕ :=
  sampled_freshmen * (total_students counts) / counts.freshman

/-- Theorem stating that for the given student counts and sampled freshmen, the sample size is 30 --/
theorem stratified_sample_size 
  (counts : StudentCounts) 
  (h1 : counts.freshman = 700)
  (h2 : counts.sophomore = 500)
  (h3 : counts.senior = 300)
  (h4 : sampled_freshmen = 14) :
  sample_size counts sampled_freshmen = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3862_386221


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3862_386268

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = -10 + 10 * Real.sqrt 2) ∧ 
              (x₂ = -10 - 10 * Real.sqrt 2) ∧ 
              (∀ x : ℝ, (10 - x)^2 = 2*x^2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3862_386268


namespace NUMINAMATH_CALUDE_product_of_roots_l3862_386203

theorem product_of_roots (x : ℝ) : 
  (∃ α β : ℝ, α * β = -21 ∧ -α^2 + 4*α = -21 ∧ -β^2 + 4*β = -21) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3862_386203


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3862_386222

theorem min_value_sum_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = x * y) :
  x + y ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3862_386222


namespace NUMINAMATH_CALUDE_triangle_angles_l3862_386242

/-- Given a triangle with sides a, b, and c, where a = b = 3 and c = √7 - √3,
    prove that the angles of the triangle are as follows:
    - Angle C (opposite side c) = arccos((4 + √21) / 9)
    - Angles A and B = (180° - arccos((4 + √21) / 9)) / 2 -/
theorem triangle_angles (a b c : ℝ) (h1 : a = 3) (h2 : b = 3) (h3 : c = Real.sqrt 7 - Real.sqrt 3) :
  let angle_c := Real.arccos ((4 + Real.sqrt 21) / 9)
  let angle_a := (π - angle_c) / 2
  ∃ (A B C : ℝ),
    A = angle_a ∧
    B = angle_a ∧
    C = angle_c ∧
    A + B + C = π :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l3862_386242


namespace NUMINAMATH_CALUDE_georgia_muffins_l3862_386272

theorem georgia_muffins (students : ℕ) (muffins_per_batch : ℕ) (months : ℕ) :
  students = 24 →
  muffins_per_batch = 6 →
  months = 9 →
  (students / muffins_per_batch) * months = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_georgia_muffins_l3862_386272


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l3862_386271

theorem unique_solution_floor_equation :
  ∃! c : ℝ, c + ⌊c⌋ = 25.6 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l3862_386271
