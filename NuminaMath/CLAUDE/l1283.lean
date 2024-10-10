import Mathlib

namespace increasing_quadratic_parameter_range_l1283_128316

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem increasing_quadratic_parameter_range (a : ℝ) :
  (∀ x ≥ 2, ∀ y ≥ 2, x < y → f a x < f a y) →
  a ∈ Set.Ici (-2 : ℝ) :=
sorry

end increasing_quadratic_parameter_range_l1283_128316


namespace white_squares_47th_row_l1283_128397

/-- Represents the number of squares in a row of the stair-step figure -/
def totalSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n - 1) / 2

/-- Theorem stating that the 47th row of the stair-step figure contains 46 white squares -/
theorem white_squares_47th_row :
  whiteSquares 47 = 46 := by
  sorry


end white_squares_47th_row_l1283_128397


namespace probability_not_red_is_three_fifths_l1283_128396

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of not seeing the red light -/
def probability_not_red (d : TrafficLightDurations) : ℚ :=
  (d.yellow + d.green : ℚ) / (d.red + d.yellow + d.green)

/-- Theorem stating the probability of not seeing the red light is 3/5 -/
theorem probability_not_red_is_three_fifths :
  let d : TrafficLightDurations := ⟨30, 5, 40⟩
  probability_not_red d = 3/5 := by sorry

end probability_not_red_is_three_fifths_l1283_128396


namespace unique_solution_condition_l1283_128352

/-- The equation (x - 3)(x - 5) = k - 4x has exactly one real solution if and only if k = 11 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 := by
  sorry

end unique_solution_condition_l1283_128352


namespace stamps_per_ounce_l1283_128313

/-- Given a letter with 8 pieces of paper each weighing 1/5 ounce,
    an envelope weighing 2/5 ounce, and requiring 2 stamps total,
    prove that 1 stamp is needed per ounce. -/
theorem stamps_per_ounce (paper_weight : ℚ) (envelope_weight : ℚ) (total_stamps : ℕ) :
  paper_weight = 1/5
  → envelope_weight = 2/5
  → total_stamps = 2
  → (total_stamps : ℚ) / (8 * paper_weight + envelope_weight) = 1 := by
  sorry

end stamps_per_ounce_l1283_128313


namespace money_division_l1283_128359

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total →
  3 * p = 7 * q →
  7 * q = 12 * r →
  q - p = 4500 →
  r - q = 4500 →
  q - p = 3600 := by
sorry

end money_division_l1283_128359


namespace baker_problem_l1283_128306

def verify_cake_info (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) : Prop :=
  initial_cakes - sold_cakes = remaining_cakes

def can_determine_initial_pastries (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) 
    (sold_pastries : ℕ) : Prop :=
  false

theorem baker_problem (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) 
    (sold_pastries : ℕ) :
  initial_cakes = 149 →
  sold_cakes = 10 →
  remaining_cakes = 139 →
  sold_pastries = 90 →
  verify_cake_info initial_cakes sold_cakes remaining_cakes ∧
  ¬can_determine_initial_pastries initial_cakes sold_cakes remaining_cakes sold_pastries :=
by
  sorry

end baker_problem_l1283_128306


namespace no_number_divisible_by_1998_with_small_digit_sum_l1283_128325

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem no_number_divisible_by_1998_with_small_digit_sum :
  ∀ n : ℕ, n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end no_number_divisible_by_1998_with_small_digit_sum_l1283_128325


namespace swimming_time_against_current_l1283_128374

theorem swimming_time_against_current 
  (swimming_speed : ℝ) 
  (water_speed : ℝ) 
  (time_with_current : ℝ) 
  (h1 : swimming_speed = 4) 
  (h2 : water_speed = 2) 
  (h3 : time_with_current = 4) : 
  (swimming_speed + water_speed) * time_with_current / (swimming_speed - water_speed) = 12 := by
  sorry

end swimming_time_against_current_l1283_128374


namespace complementary_angle_of_35_30_l1283_128370

-- Define the angle in degrees and minutes
def angle_alpha : ℚ := 35 + 30 / 60

-- Define the complementary angle function
def complementary_angle (α : ℚ) : ℚ := 90 - α

-- Theorem statement
theorem complementary_angle_of_35_30 :
  let result := complementary_angle angle_alpha
  ⌊result⌋ = 54 ∧ (result - ⌊result⌋) * 60 = 30 := by
  sorry

#eval complementary_angle angle_alpha

end complementary_angle_of_35_30_l1283_128370


namespace tetrahedron_edge_assignment_exists_l1283_128312

/-- Represents a tetrahedron with face areas -/
structure Tetrahedron where
  s : ℝ  -- smallest face area
  S : ℝ  -- largest face area
  a : ℝ  -- another face area
  b : ℝ  -- another face area
  h_s_smallest : s ≤ S ∧ s ≤ a ∧ s ≤ b
  h_S_largest : S ≥ s ∧ S ≥ a ∧ S ≥ b
  h_positive : s > 0 ∧ S > 0 ∧ a > 0 ∧ b > 0

/-- Represents the assignment of numbers to the edges of a tetrahedron -/
structure EdgeAssignment (t : Tetrahedron) where
  e1 : ℝ  -- edge common to smallest and largest face
  e2 : ℝ  -- edge of smallest face
  e3 : ℝ  -- edge of smallest face
  e4 : ℝ  -- edge of largest face
  e5 : ℝ  -- edge of largest face
  e6 : ℝ  -- remaining edge
  h_non_negative : e1 ≥ 0 ∧ e2 ≥ 0 ∧ e3 ≥ 0 ∧ e4 ≥ 0 ∧ e5 ≥ 0 ∧ e6 ≥ 0

/-- The theorem stating that a valid edge assignment exists for any tetrahedron -/
theorem tetrahedron_edge_assignment_exists (t : Tetrahedron) :
  ∃ (ea : EdgeAssignment t),
    ea.e1 + ea.e2 + ea.e3 = t.s ∧
    ea.e1 + ea.e4 + ea.e5 = t.S ∧
    ea.e2 + ea.e5 + ea.e6 = t.a ∧
    ea.e3 + ea.e4 + ea.e6 = t.b :=
  sorry

end tetrahedron_edge_assignment_exists_l1283_128312


namespace blake_bucket_water_l1283_128305

theorem blake_bucket_water (poured_out water_left : ℝ) 
  (h1 : poured_out = 0.2)
  (h2 : water_left = 0.6) :
  poured_out + water_left = 0.8 := by
sorry

end blake_bucket_water_l1283_128305


namespace sales_tax_difference_l1283_128395

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 30 → tax_rate1 = 0.0675 → tax_rate2 = 0.055 → 
  price * tax_rate1 - price * tax_rate2 = 0.375 := by
  sorry

#eval (30 * 0.0675 - 30 * 0.055)

end sales_tax_difference_l1283_128395


namespace alex_candles_used_l1283_128327

/-- The number of candles Alex used -/
def candles_used (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem stating that Alex used 32 candles -/
theorem alex_candles_used :
  let initial : ℕ := 44
  let remaining : ℕ := 12
  candles_used initial remaining = 32 := by
  sorry

end alex_candles_used_l1283_128327


namespace problem_statement_l1283_128368

theorem problem_statement (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) :
  a^2006 + (a + b)^2007 = 2 := by
  sorry

end problem_statement_l1283_128368


namespace fourth_term_is_six_l1283_128333

/-- An increasing sequence of positive integers satisfying a_{a_n} = 2n + 1 -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, n < m → a n < a m) ∧ 
  (∀ n : ℕ+, a (a n) = 2*n + 1)

/-- The fourth term of the special sequence is 6 -/
theorem fourth_term_is_six (a : ℕ → ℕ) (h : SpecialSequence a) : a 4 = 6 := by
  sorry

end fourth_term_is_six_l1283_128333


namespace range_of_m_l1283_128315

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀ + (1/3) * m = Real.exp x₀

def q (m : ℝ) : Prop :=
  let a := m
  let b := 5
  let e := Real.sqrt ((a - b) / a)
  (1/2) < e ∧ e < (2/3)

-- Define the theorem
theorem range_of_m (m : ℝ) (h : p m ∧ q m) :
  (20/3 < m ∧ m < 9) ∨ (3 ≤ m ∧ m < 15/4) := by
  sorry

end range_of_m_l1283_128315


namespace expression_evaluation_l1283_128388

theorem expression_evaluation : (120 / 6 * 2 / 3 : ℚ) = 40 / 3 := by sorry

end expression_evaluation_l1283_128388


namespace ellipse_m_values_l1283_128339

/-- An ellipse with equation x²/5 + y²/m = 1 and eccentricity √10/5 has m equal to 3 or 25/3 -/
theorem ellipse_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2/5 + y^2/m = 1) →  -- Ellipse equation
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (x^2/a^2 + y^2/b^2 = 1 ↔ x^2/5 + y^2/m = 1) ∧  -- Standard form of ellipse
    c^2/a^2 = 10/25) →  -- Eccentricity condition
  m = 3 ∨ m = 25/3 := by
sorry

end ellipse_m_values_l1283_128339


namespace integer_double_root_theorem_l1283_128311

/-- A polynomial with integer coefficients of the form x^4 + b_3x^3 + b_2x^2 + b_1x + 48 -/
def IntPolynomial (b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 48

/-- The set of possible integer double roots -/
def PossibleRoots : Set ℤ := {-4, -2, -1, 1, 2, 4}

theorem integer_double_root_theorem (b₃ b₂ b₁ s : ℤ) :
  (∃ k : ℤ, IntPolynomial b₃ b₂ b₁ x = (x - s)^2 * (x^2 + kx + m)) →
  s ∈ PossibleRoots := by
  sorry

end integer_double_root_theorem_l1283_128311


namespace pure_imaginary_fraction_l1283_128338

theorem pure_imaginary_fraction (a : ℝ) :
  (∃ b : ℝ, (a^2 + Complex.I) / (1 - Complex.I) = Complex.I * b) →
  a = 1 ∨ a = -1 := by
  sorry

end pure_imaginary_fraction_l1283_128338


namespace tan_ratio_given_sin_condition_l1283_128392

theorem tan_ratio_given_sin_condition (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (π / 180))) : 
  Real.tan (α + π / 180) / Real.tan (α - π / 180) = -3/2 := by
  sorry

end tan_ratio_given_sin_condition_l1283_128392


namespace total_fish_weight_l1283_128361

/-- Calculates the total weight of fish in James' three tanks -/
theorem total_fish_weight (goldfish_weight guppy_weight angelfish_weight : ℝ)
  (goldfish_count_1 guppy_count_1 : ℕ)
  (goldfish_count_2 guppy_count_2 : ℕ)
  (goldfish_count_3 guppy_count_3 angelfish_count_3 : ℕ)
  (h1 : goldfish_weight = 0.08)
  (h2 : guppy_weight = 0.05)
  (h3 : angelfish_weight = 0.14)
  (h4 : goldfish_count_1 = 15)
  (h5 : guppy_count_1 = 12)
  (h6 : goldfish_count_2 = 2 * goldfish_count_1)
  (h7 : guppy_count_2 = 3 * guppy_count_1)
  (h8 : goldfish_count_3 = 3 * goldfish_count_1)
  (h9 : guppy_count_3 = 2 * guppy_count_1)
  (h10 : angelfish_count_3 = 5) :
  goldfish_weight * (goldfish_count_1 + goldfish_count_2 + goldfish_count_3 : ℝ) +
  guppy_weight * (guppy_count_1 + guppy_count_2 + guppy_count_3 : ℝ) +
  angelfish_weight * angelfish_count_3 = 11.5 := by
  sorry


end total_fish_weight_l1283_128361


namespace village_blocks_l1283_128308

/-- The number of blocks in a village, given the number of children per block and the total number of children. -/
def number_of_blocks (children_per_block : ℕ) (total_children : ℕ) : ℕ :=
  total_children / children_per_block

/-- Theorem: Given 6 children per block and 54 total children, there are 9 blocks in the village. -/
theorem village_blocks :
  number_of_blocks 6 54 = 9 := by
  sorry

end village_blocks_l1283_128308


namespace quadratic_root_product_l1283_128303

theorem quadratic_root_product (b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + b*x + 8
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ * x₂ = 8 :=
by sorry

end quadratic_root_product_l1283_128303


namespace instantaneous_velocity_at_4_l1283_128344

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_4 : v 4 = 7 := by
  sorry

end instantaneous_velocity_at_4_l1283_128344


namespace basketball_lineup_combinations_l1283_128378

theorem basketball_lineup_combinations (total_players : ℕ) (lineup_size : ℕ) (guaranteed_players : ℕ) :
  total_players = 15 →
  lineup_size = 6 →
  guaranteed_players = 2 →
  Nat.choose (total_players - guaranteed_players) (lineup_size - guaranteed_players) = 715 :=
by sorry

end basketball_lineup_combinations_l1283_128378


namespace product_x_y_is_32_l1283_128353

/-- A parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  is_parallelogram : EF = GH ∧ FG = HE

/-- The product of x and y in the given parallelogram is 32 -/
theorem product_x_y_is_32 (p : Parallelogram)
  (h1 : p.EF = 42)
  (h2 : ∃ y, p.FG = 4 * y^3)
  (h3 : ∃ x, p.GH = 2 * x + 10)
  (h4 : p.HE = 32) :
  ∃ x y, x * y = 32 ∧ p.FG = 4 * y^3 ∧ p.GH = 2 * x + 10 := by
  sorry

end product_x_y_is_32_l1283_128353


namespace total_cost_proof_l1283_128393

def price_AVN : ℝ := 12
def price_TheDark : ℝ := 2 * price_AVN
def num_TheDark : ℕ := 2
def num_AVN : ℕ := 1
def ratio_90s : ℝ := 0.4
def num_90s : ℕ := 5

theorem total_cost_proof :
  let cost_main := price_TheDark * num_TheDark + price_AVN * num_AVN
  let cost_90s := ratio_90s * cost_main * num_90s
  cost_main + cost_90s = 180 := by sorry

end total_cost_proof_l1283_128393


namespace B_equals_zero_one_l1283_128336

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_zero_one : B = {0, 1} := by
  sorry

end B_equals_zero_one_l1283_128336


namespace custom_mul_four_three_l1283_128371

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 - a*b + b^2

/-- Theorem stating that 4*3 = 13 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 13 := by
  sorry

end custom_mul_four_three_l1283_128371


namespace base_power_zero_l1283_128321

theorem base_power_zero (b : ℝ) (x y : ℝ) (h1 : 3^x * b^y = 59049) (h2 : x - y = 10) (h3 : x = 10) : y = 0 := by
  sorry

end base_power_zero_l1283_128321


namespace min_perimeter_area_l1283_128358

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the right focus F
def rightFocus : ℝ × ℝ := (3, 0)

-- Define point A
def A : ℝ × ℝ := (0, 4)

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Define the perimeter of triangle APF
def perimeter (p : ℝ × ℝ) : ℝ := sorry

-- Define the area of triangle APF
def area (p : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_perimeter_area :
  ∃ (p : ℝ × ℝ), 
    hyperbola p.1 p.2 ∧ 
    p.1 < 0 ∧ 
    (∀ q : ℝ × ℝ, hyperbola q.1 q.2 ∧ q.1 < 0 → perimeter p ≤ perimeter q) ∧
    area p = 36/7 :=
sorry

end min_perimeter_area_l1283_128358


namespace product_inequality_l1283_128319

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a * b) ^ (1/4 : ℝ) + (b * c) ^ (1/4 : ℝ) + (c * a) ^ (1/4 : ℝ) < 1/4 := by
  sorry

end product_inequality_l1283_128319


namespace first_car_mpg_l1283_128301

/-- Proves that the average miles per gallon of the first car is 27.5 given the conditions --/
theorem first_car_mpg (total_miles : ℝ) (total_gallons : ℝ) (second_car_mpg : ℝ) (first_car_gallons : ℝ) :
  total_miles = 1825 →
  total_gallons = 55 →
  second_car_mpg = 40 →
  first_car_gallons = 30 →
  (first_car_gallons * (total_miles - second_car_mpg * (total_gallons - first_car_gallons))) / 
    (first_car_gallons * total_miles) = 27.5 := by
sorry

end first_car_mpg_l1283_128301


namespace equation_solution_l1283_128386

theorem equation_solution : ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2/11 := by
  sorry

end equation_solution_l1283_128386


namespace smallest_f_one_l1283_128324

/-- A cubic polynomial f(x) with specific properties -/
noncomputable def f (r s : ℝ) (x : ℝ) : ℝ := (x - r) * (x - s) * (x - (r + s) / 2)

/-- The theorem stating the smallest value of f(1) -/
theorem smallest_f_one (r s : ℝ) :
  (r ≠ s) →
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f r s (f r s x₁) = 0 ∧ f r s (f r s x₂) = 0 ∧ f r s (f r s x₃) = 0) →
  (∀ (x : ℝ), f r s (f r s x) = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∀ (r' s' : ℝ), r' ≠ s' → f r' s' 1 ≥ 3/8) ∧
  (∃ (r₀ s₀ : ℝ), r₀ ≠ s₀ ∧ f r₀ s₀ 1 = 3/8) := by
  sorry


end smallest_f_one_l1283_128324


namespace angle_between_vectors_l1283_128343

/-- Given vectors a and b in ℝ², prove that the angle between them is π -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a + 2 • b = (2, -4) → 
  3 • a - b = (-8, 16) → 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π :=
by sorry

end angle_between_vectors_l1283_128343


namespace book_has_fifty_pages_l1283_128307

/-- Calculates the number of pages in a book based on reading speed and book structure -/
def book_pages (sentences_per_hour : ℕ) (paragraphs_per_page : ℕ) (sentences_per_paragraph : ℕ) (total_reading_hours : ℕ) : ℕ :=
  (sentences_per_hour * total_reading_hours) / (sentences_per_paragraph * paragraphs_per_page)

/-- Theorem stating that given the specific conditions, the book has 50 pages -/
theorem book_has_fifty_pages :
  book_pages 200 20 10 50 = 50 := by
  sorry

end book_has_fifty_pages_l1283_128307


namespace parallel_lines_condition_l1283_128320

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (2*a - 1)*x - a*y - 1 = 0) ↔ (a = 0 ∨ a = 1/4) :=
by sorry

end parallel_lines_condition_l1283_128320


namespace coefficient_of_x_is_17_l1283_128389

def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (9 - 3 * x^2 + 7 * x) - 10 * (3 * x - 2)

theorem coefficient_of_x_is_17 :
  ∃ (a b c : ℝ), expression = λ x => a * x^2 + 17 * x + c :=
by sorry

end coefficient_of_x_is_17_l1283_128389


namespace dorothy_initial_money_l1283_128362

-- Define the family members
inductive FamilyMember
| Dorothy
| Brother
| Parent1
| Parent2
| Grandfather

-- Define the age of a family member
def age (member : FamilyMember) : ℕ :=
  match member with
  | .Dorothy => 15
  | .Brother => 0  -- We don't know exact age, but younger than 18
  | .Parent1 => 18 -- We don't know exact age, but at least 18
  | .Parent2 => 18 -- We don't know exact age, but at least 18
  | .Grandfather => 18 -- We don't know exact age, but at least 18

-- Define the regular ticket price
def regularTicketPrice : ℕ := 10

-- Define the discount rate for young people
def youngDiscount : ℚ := 0.3

-- Define the discounted ticket price function
def ticketPrice (member : FamilyMember) : ℚ :=
  if age member ≤ 18 then
    regularTicketPrice * (1 - youngDiscount)
  else
    regularTicketPrice

-- Define the total cost of tickets for the family
def totalTicketCost : ℚ :=
  ticketPrice FamilyMember.Dorothy +
  ticketPrice FamilyMember.Brother +
  ticketPrice FamilyMember.Parent1 +
  ticketPrice FamilyMember.Parent2 +
  ticketPrice FamilyMember.Grandfather

-- Define Dorothy's remaining money after the trip
def moneyLeftAfterTrip : ℚ := 26

-- Theorem: Dorothy's initial money was $70
theorem dorothy_initial_money :
  totalTicketCost + moneyLeftAfterTrip = 70 := by
  sorry

end dorothy_initial_money_l1283_128362


namespace pau_total_chicken_l1283_128390

def kobe_order : ℕ := 5

def pau_order (kobe : ℕ) : ℕ := 2 * kobe

def total_pau_order (initial : ℕ) : ℕ := 2 * initial

theorem pau_total_chicken :
  total_pau_order (pau_order kobe_order) = 20 := by
  sorry

end pau_total_chicken_l1283_128390


namespace orange_packing_problem_l1283_128332

/-- Given a total number of oranges and the capacity of each box, 
    calculate the number of boxes needed. -/
def boxes_needed (total_oranges : ℕ) (oranges_per_box : ℕ) : ℕ :=
  total_oranges / oranges_per_box

/-- Theorem stating that 265 boxes are needed to pack 2650 oranges 
    when each box holds 10 oranges. -/
theorem orange_packing_problem : 
  boxes_needed 2650 10 = 265 := by
  sorry

end orange_packing_problem_l1283_128332


namespace fencing_requirement_l1283_128318

theorem fencing_requirement (length width : ℝ) (h1 : length = 30) (h2 : length * width = 810) :
  2 * width + length = 84 := by
  sorry

end fencing_requirement_l1283_128318


namespace square_sum_inequality_equality_condition_l1283_128342

theorem square_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end square_sum_inequality_equality_condition_l1283_128342


namespace connie_tickets_l1283_128391

/-- The number of tickets Connie redeemed -/
def total_tickets : ℕ := 50

/-- The number of tickets spent on earbuds -/
def earbuds_tickets : ℕ := 10

/-- The number of tickets spent on glow bracelets -/
def glow_bracelets_tickets : ℕ := 15

/-- Theorem stating that Connie redeemed 50 tickets -/
theorem connie_tickets : 
  (total_tickets / 2 : ℚ) + earbuds_tickets + glow_bracelets_tickets = total_tickets := by
  sorry

#check connie_tickets

end connie_tickets_l1283_128391


namespace adiabatic_compression_work_l1283_128357

/-- Adiabatic compression work in a cylindrical vessel -/
theorem adiabatic_compression_work
  (V₀ V₁ p₀ k : ℝ)
  (h₀ : V₀ > 0)
  (h₁ : V₁ > 0)
  (h₂ : p₀ > 0)
  (h₃ : k > 1)
  (h₄ : V₁ < V₀) :
  ∃ W : ℝ, W = (p₀ * V₀ / (k - 1)) * ((V₀ / V₁) ^ (k - 1) - 1) :=
by sorry

end adiabatic_compression_work_l1283_128357


namespace village_population_l1283_128369

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 80 / 100 →
  partial_population = 32000 →
  percentage * (total_population : ℚ) = partial_population →
  total_population = 40000 :=
by
  sorry

end village_population_l1283_128369


namespace f_properties_l1283_128372

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 2*(x + 1) + 7

-- Theorem statement
theorem f_properties :
  (f 2 = 10) ∧
  (∀ a, f a = a^2 + 6) ∧
  (∀ x, f x = x^2 + 6) ∧
  (∀ x, f (x + 1) = x^2 + 2*x + 7) ∧
  (∀ y, y ∈ Set.range (λ x => f (x + 1)) ↔ y ≥ 6) :=
by sorry

end f_properties_l1283_128372


namespace acute_angle_relation_l1283_128366

theorem acute_angle_relation (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = (1/2) * Real.sin (α + β)) : α < β := by
  sorry

end acute_angle_relation_l1283_128366


namespace sand_overflow_l1283_128398

/-- Represents the capacity of a bucket -/
structure BucketCapacity where
  value : ℚ
  positive : 0 < value

/-- Represents the amount of sand in a bucket -/
structure SandAmount where
  amount : ℚ
  nonnegative : 0 ≤ amount

/-- Theorem stating the overflow amount when pouring sand between buckets -/
theorem sand_overflow
  (CA : BucketCapacity) -- Capacity of Bucket A
  (sand_A : SandAmount) -- Initial sand in Bucket A
  (sand_B : SandAmount) -- Initial sand in Bucket B
  (sand_C : SandAmount) -- Initial sand in Bucket C
  (h1 : sand_A.amount = (1 : ℚ) / 4 * CA.value) -- Bucket A is 1/4 full
  (h2 : sand_B.amount = (3 : ℚ) / 8 * (CA.value / 2)) -- Bucket B is 3/8 full
  (h3 : sand_C.amount = (1 : ℚ) / 3 * (2 * CA.value)) -- Bucket C is 1/3 full
  : ∃ (overflow : ℚ), overflow = (17 : ℚ) / 48 * CA.value :=
by sorry

end sand_overflow_l1283_128398


namespace exam_mean_score_l1283_128382

theorem exam_mean_score (score_below mean score_above : ℝ) 
  (h1 : score_below = mean - 2 * (score_above - mean) / 5)
  (h2 : score_above = mean + 3 * (score_above - mean) / 5)
  (h3 : score_below = 60)
  (h4 : score_above = 100) : 
  mean = 76 := by
sorry

end exam_mean_score_l1283_128382


namespace ladder_problem_l1283_128329

theorem ladder_problem (ladder_length height_on_wall base_distance : ℝ) : 
  ladder_length = 13 ∧ height_on_wall = 12 ∧ 
  ladder_length^2 = height_on_wall^2 + base_distance^2 → 
  base_distance = 5 := by
sorry

end ladder_problem_l1283_128329


namespace sum_of_net_gains_l1283_128350

def initial_revenue : ℝ := 4.7
def revenue_increase_A : ℝ := 0.1326
def revenue_increase_B : ℝ := 0.0943
def revenue_increase_C : ℝ := 0.7731
def tax_rate : ℝ := 0.235

def net_gain (initial_rev : ℝ) (rev_increase : ℝ) (tax : ℝ) : ℝ :=
  (initial_rev * (1 + rev_increase)) * (1 - tax)

theorem sum_of_net_gains :
  let net_gain_A := net_gain initial_revenue revenue_increase_A tax_rate
  let net_gain_B := net_gain initial_revenue revenue_increase_B tax_rate
  let net_gain_C := net_gain initial_revenue revenue_increase_C tax_rate
  net_gain_A + net_gain_B + net_gain_C = 14.38214 := by sorry

end sum_of_net_gains_l1283_128350


namespace no_circular_arrangement_with_conditions_l1283_128302

theorem no_circular_arrangement_with_conditions : ¬ ∃ (a : Fin 9 → ℕ),
  (∀ i, a i ∈ Finset.range 9) ∧
  (∀ i, a i ≠ 0) ∧
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
  (∀ i, a i + a ((i + 1) % 9) + a ((i + 2) % 9) > 12) :=
by sorry

end no_circular_arrangement_with_conditions_l1283_128302


namespace uncle_bobs_age_l1283_128331

theorem uncle_bobs_age (anna_age brianna_age caitlin_age bob_age : ℕ) : 
  anna_age = 48 →
  brianna_age = anna_age / 2 →
  caitlin_age = brianna_age - 6 →
  bob_age = 3 * caitlin_age →
  bob_age = 54 := by
sorry

end uncle_bobs_age_l1283_128331


namespace arctan_of_tan_difference_l1283_128354

-- Define the problem parameters
def angle₁ : Real := 80
def angle₂ : Real := 30

-- Define the theorem
theorem arctan_of_tan_difference (h : 0 ≤ angle₁ ∧ angle₁ ≤ 180 ∧ 0 ≤ angle₂ ∧ angle₂ ≤ 180) :
  Real.arctan (Real.tan (angle₁ * π / 180) - 3 * Real.tan (angle₂ * π / 180)) * 180 / π = angle₁ := by
  sorry


end arctan_of_tan_difference_l1283_128354


namespace even_function_implies_b_zero_solution_set_inequality_l1283_128376

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

-- Theorem 1: If f is even, then b = 0
theorem even_function_implies_b_zero (b : ℝ) :
  (∀ x : ℝ, f b x = f b (-x)) → b = 0 := by sorry

-- Define the specific function f with b = 0
def f_zero (x : ℝ) : ℝ := x^2 + 1

-- Theorem 2: Solution set of f(x-1) < |x|
theorem solution_set_inequality :
  {x : ℝ | f_zero (x - 1) < |x|} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end even_function_implies_b_zero_solution_set_inequality_l1283_128376


namespace sum_of_circle_areas_l1283_128340

/-- Given a 5-12-13 right triangle with vertices as centers of mutually externally tangent circles,
    where the radius of the circle at the right angle is half that of the circle opposite the shortest side,
    prove that the sum of the areas of these circles is 105π. -/
theorem sum_of_circle_areas (r s t : ℝ) : 
  r > 0 ∧ s > 0 ∧ t > 0 →  -- radii are positive
  r + s = 13 →  -- sum of radii equals hypotenuse
  s + t = 5 →   -- sum of radii equals one side
  r + t = 12 →  -- sum of radii equals other side
  t = r / 2 →   -- radius at right angle is half of radius opposite shortest side
  π * (r^2 + s^2 + t^2) = 105 * π := by
  sorry

end sum_of_circle_areas_l1283_128340


namespace first_step_error_l1283_128365

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (x - 1) / 2 + 1 = (2 * x + 1) / 3

-- Define the incorrect step 1 result
def incorrect_step1 (x : ℝ) : Prop :=
  3 * (x - 1) + 2 = 2 * x + 1

-- Define the correct step 1 result
def correct_step1 (x : ℝ) : Prop :=
  3 * (x - 1) + 6 = 2 * x + 1

-- Theorem stating that the first step is erroneous
theorem first_step_error :
  ∃ x : ℝ, original_equation x ∧ ¬(incorrect_step1 x ↔ correct_step1 x) :=
sorry

end first_step_error_l1283_128365


namespace family_income_problem_l1283_128323

/-- The number of initial earning members in a family -/
def initial_members : ℕ := 4

/-- The initial average monthly income -/
def initial_average : ℚ := 735

/-- The new average monthly income after one member's death -/
def new_average : ℚ := 650

/-- The income of the deceased member -/
def deceased_income : ℚ := 990

theorem family_income_problem :
  initial_members * initial_average - (initial_members - 1) * new_average = deceased_income :=
by sorry

end family_income_problem_l1283_128323


namespace negation_exactly_one_even_l1283_128381

/-- Represents the property of a natural number being even -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Represents the property that exactly one of three natural numbers is even -/
def ExactlyOneEven (a b c : ℕ) : Prop :=
  (IsEven a ∧ ¬IsEven b ∧ ¬IsEven c) ∨
  (¬IsEven a ∧ IsEven b ∧ ¬IsEven c) ∨
  (¬IsEven a ∧ ¬IsEven b ∧ IsEven c)

/-- The main theorem stating that the negation of "exactly one even" is equivalent to "all odd or at least two even" -/
theorem negation_exactly_one_even (a b c : ℕ) :
  ¬(ExactlyOneEven a b c) ↔ (¬IsEven a ∧ ¬IsEven b ∧ ¬IsEven c) ∨ (IsEven a ∧ IsEven b) ∨ (IsEven a ∧ IsEven c) ∨ (IsEven b ∧ IsEven c) :=
sorry


end negation_exactly_one_even_l1283_128381


namespace inverse_prop_t_times_function_no_linear_2k_times_function_quadratic_5_times_function_l1283_128387

/-- Definition of a "t times function" on [a,b] -/
def is_t_times_function (f : ℝ → ℝ) (t a b : ℝ) : Prop :=
  a < b ∧ t > 0 ∧ ∀ x ∈ Set.Icc a b, t * a ≤ f x ∧ f x ≤ t * b

/-- Part 1: Inverse proportional function -/
theorem inverse_prop_t_times_function :
  ∀ t > 0, is_t_times_function (fun x ↦ 2023 / x) t 1 2023 ↔ t = 1 := by sorry

/-- Part 2: Non-existence of linear "2k times function" -/
theorem no_linear_2k_times_function :
  ∀ k > 0, ∀ a b : ℝ, a < b →
    ¬∃ (c : ℝ), is_t_times_function (fun x ↦ k * x + c) (2 * k) a b := by sorry

/-- Part 3: Quadratic "5 times function" -/
theorem quadratic_5_times_function :
  ∀ a b : ℝ, is_t_times_function (fun x ↦ x^2 - 4*x - 7) 5 a b ↔
    (a = -2 ∧ b = 1) ∨ (a = -11/5 ∧ b = (9 + Real.sqrt 109) / 2) := by sorry

end inverse_prop_t_times_function_no_linear_2k_times_function_quadratic_5_times_function_l1283_128387


namespace upstream_speed_is_eight_l1283_128322

/-- Represents the speed of a man in a stream -/
structure StreamSpeed where
  downstream : ℝ
  stream : ℝ

/-- Calculates the upstream speed given downstream and stream speeds -/
def upstreamSpeed (s : StreamSpeed) : ℝ :=
  s.downstream - 2 * s.stream

/-- Theorem stating that for given downstream and stream speeds, the upstream speed is 8 -/
theorem upstream_speed_is_eight (s : StreamSpeed) 
  (h1 : s.downstream = 15) 
  (h2 : s.stream = 3.5) : 
  upstreamSpeed s = 8 := by
  sorry

end upstream_speed_is_eight_l1283_128322


namespace shinyoung_candy_problem_l1283_128349

theorem shinyoung_candy_problem (initial_candies : ℕ) : 
  (initial_candies / 2 - (initial_candies / 2 / 3 + 5) = 5) → 
  initial_candies = 30 :=
by
  sorry

end shinyoung_candy_problem_l1283_128349


namespace cube_sum_divisible_by_nine_l1283_128347

theorem cube_sum_divisible_by_nine (n : ℕ+) : 
  9 ∣ (n.val^3 + (n.val + 1)^3 + (n.val + 2)^3) := by
sorry

end cube_sum_divisible_by_nine_l1283_128347


namespace statue_weight_proof_l1283_128355

/-- The weight of a statue after three successive cuts -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_first_cut := initial_weight * (1 - 0.25)
  let weight_after_second_cut := weight_after_first_cut * (1 - 0.15)
  let weight_after_third_cut := weight_after_second_cut * (1 - 0.10)
  weight_after_third_cut

/-- Theorem stating that the final weight of the statue is 109.0125 kg -/
theorem statue_weight_proof :
  final_statue_weight 190 = 109.0125 := by
  sorry

end statue_weight_proof_l1283_128355


namespace cone_curved_surface_area_l1283_128330

/-- The curved surface area of a cone with given slant height and base radius -/
theorem cone_curved_surface_area 
  (slant_height : ℝ) 
  (base_radius : ℝ) 
  (h1 : slant_height = 10) 
  (h2 : base_radius = 5) : 
  π * base_radius * slant_height = 50 * π := by
sorry

end cone_curved_surface_area_l1283_128330


namespace distribute_teachers_count_l1283_128360

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the number of teachers --/
def num_teachers : ℕ := 5

/-- Represents the constraint that each school must have at least one teacher --/
def min_teachers_per_school : ℕ := 1

/-- The function that calculates the number of ways to distribute teachers --/
def distribute_teachers : ℕ := sorry

/-- The theorem stating that the number of ways to distribute teachers is 150 --/
theorem distribute_teachers_count : distribute_teachers = 150 := by sorry

end distribute_teachers_count_l1283_128360


namespace point_on_line_k_l1283_128351

/-- A line passing through the origin with slope 1/5 -/
def line_k (x y : ℝ) : Prop := y = (1/5) * x

theorem point_on_line_k (x y : ℝ) :
  line_k x 1 →
  line_k 5 y →
  y = 1 := by
  sorry

end point_on_line_k_l1283_128351


namespace incorrect_observation_value_l1283_128317

theorem incorrect_observation_value
  (n : ℕ)
  (initial_mean correct_mean correct_value : ℚ)
  (h_n : n = 50)
  (h_initial_mean : initial_mean = 36)
  (h_correct_mean : correct_mean = 365/10)
  (h_correct_value : correct_value = 45)
  : (n : ℚ) * initial_mean + correct_value - ((n : ℚ) * correct_mean) = 20 := by
  sorry

end incorrect_observation_value_l1283_128317


namespace smallest_n_divisible_by_1991_l1283_128309

theorem smallest_n_divisible_by_1991 : ∃ (n : ℕ),
  (∀ (m : ℕ), m < n → ∃ (S : Finset ℤ), S.card = m ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → ¬(1991 ∣ (a + b)) ∧ ¬(1991 ∣ (a - b))) ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1991 ∣ (a + b) ∨ 1991 ∣ (a - b))) ∧
  n = 997 :=
sorry

end smallest_n_divisible_by_1991_l1283_128309


namespace kyuhyung_candies_l1283_128379

theorem kyuhyung_candies :
  ∀ (k d : ℕ), -- k for Kyuhyung's candies, d for Dongmin's candies
  d = k + 5 →   -- Dongmin has 5 more candies than Kyuhyung
  k + d = 43 → -- The sum of their candies is 43
  k = 19       -- Kyuhyung has 19 candies
  := by sorry

end kyuhyung_candies_l1283_128379


namespace sum_of_four_squares_express_689_as_sum_of_squares_l1283_128348

theorem sum_of_four_squares (m n : ℕ) (h : m ≠ n) :
  ∃ (a b c d : ℕ), m^4 + 4*n^4 = a^2 + b^2 + c^2 + d^2 :=
sorry

theorem express_689_as_sum_of_squares :
  ∃ (a b c d : ℕ), 689 = a^2 + b^2 + c^2 + d^2 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end sum_of_four_squares_express_689_as_sum_of_squares_l1283_128348


namespace arithmetic_sequence_length_l1283_128341

theorem arithmetic_sequence_length 
  (a₁ : ℤ) 
  (d : ℤ) 
  (aₙ : ℤ) 
  (h1 : a₁ = 2) 
  (h2 : d = 3) 
  (h3 : aₙ = 110) :
  ∃ n : ℕ, n = 37 ∧ aₙ = a₁ + (n - 1) * d :=
sorry

end arithmetic_sequence_length_l1283_128341


namespace consecutive_fibonacci_coprime_l1283_128364

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem consecutive_fibonacci_coprime (n : ℕ) (h : n ≥ 1) : 
  Nat.gcd (fib n) (fib (n - 1)) = 1 := by
  sorry

end consecutive_fibonacci_coprime_l1283_128364


namespace biased_coin_probability_l1283_128328

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) :
  (Nat.choose 6 2 : ℝ) * p^2 * (1 - p)^4 = 1/8 → p = 1/5 := by
  sorry

end biased_coin_probability_l1283_128328


namespace p_is_converse_of_r_l1283_128383

-- Define propositions as functions from some type α to Prop
variable {α : Type}
variable (p q r : α → Prop)

-- Define the relationships between p, q, and r
axiom contrapositive : (∀ x, p x → q x) ↔ (∀ x, ¬q x → ¬p x)
axiom negation : (∀ x, q x) ↔ (∀ x, ¬r x)

-- Theorem to prove
theorem p_is_converse_of_r : (∀ x, p x → r x) ↔ (∀ x, r x → p x) := by sorry

end p_is_converse_of_r_l1283_128383


namespace sum_of_four_numbers_l1283_128346

theorem sum_of_four_numbers : 1357 + 7531 + 3175 + 5713 = 17776 := by
  sorry

end sum_of_four_numbers_l1283_128346


namespace tangent_product_l1283_128345

theorem tangent_product (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := by sorry

end tangent_product_l1283_128345


namespace sum_of_constants_l1283_128304

theorem sum_of_constants (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / (x^2 + 1)) →
  (3 = a + b / (1^2 + 1)) →
  (2 = a + b / (0^2 + 1)) →
  a + b = 2 := by
sorry

end sum_of_constants_l1283_128304


namespace playground_total_l1283_128363

/-- The number of people on a playground --/
structure Playground where
  girls : ℕ
  boys : ℕ
  thirdGradeGirls : ℕ
  thirdGradeBoys : ℕ
  teachers : ℕ
  maleTeachers : ℕ
  femaleTeachers : ℕ

/-- The total number of people on the playground is 67 --/
theorem playground_total (p : Playground)
  (h1 : p.girls = 28)
  (h2 : p.boys = 35)
  (h3 : p.thirdGradeGirls = 15)
  (h4 : p.thirdGradeBoys = 18)
  (h5 : p.teachers = 4)
  (h6 : p.maleTeachers = 2)
  (h7 : p.femaleTeachers = 2)
  (h8 : p.teachers = p.maleTeachers + p.femaleTeachers) :
  p.girls + p.boys + p.teachers = 67 := by
  sorry

#check playground_total

end playground_total_l1283_128363


namespace product_neg_seventeen_sum_l1283_128335

theorem product_neg_seventeen_sum (a b c : ℤ) : 
  a * b * c = -17 → (a + b + c = -17 ∨ a + b + c = 15) :=
by
  sorry

end product_neg_seventeen_sum_l1283_128335


namespace zack_traveled_20_countries_l1283_128356

def alex_countries : ℕ := 30

def george_countries (alex : ℕ) : ℚ := (3 : ℚ) / 5 * alex

def joseph_countries (george : ℚ) : ℚ := (1 : ℚ) / 3 * george

def patrick_countries (joseph : ℚ) : ℚ := (4 : ℚ) / 3 * joseph

def zack_countries (patrick : ℚ) : ℚ := (5 : ℚ) / 2 * patrick

theorem zack_traveled_20_countries :
  zack_countries (patrick_countries (joseph_countries (george_countries alex_countries))) = 20 := by
  sorry

end zack_traveled_20_countries_l1283_128356


namespace reflected_quad_area_l1283_128377

/-- A convex quadrilateral in the plane -/
structure ConvexQuadrilateral where
  -- We don't need to define the specifics of the quadrilateral,
  -- just that it exists and has an area
  area : ℝ
  area_pos : area > 0

/-- The quadrilateral formed by reflecting a point inside a convex quadrilateral 
    with respect to the midpoints of its sides -/
def reflectedQuadrilateral (Q : ConvexQuadrilateral) : ConvexQuadrilateral where
  -- We don't need to define how this quadrilateral is constructed,
  -- just that it exists and is related to the original quadrilateral
  area := 2 * Q.area
  area_pos := by
    -- The proof that the area is positive
    sorry

/-- Theorem stating that the area of the reflected quadrilateral 
    is twice the area of the original quadrilateral -/
theorem reflected_quad_area (Q : ConvexQuadrilateral) :
  (reflectedQuadrilateral Q).area = 2 * Q.area := by
  -- The proof of the theorem
  sorry

end reflected_quad_area_l1283_128377


namespace complement_union_problem_l1283_128375

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 3}

theorem complement_union_problem : 
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end complement_union_problem_l1283_128375


namespace total_seashells_l1283_128380

theorem total_seashells (sam mary lucy : ℕ) 
  (h_sam : sam = 18) 
  (h_mary : mary = 47) 
  (h_lucy : lucy = 32) : 
  sam + mary + lucy = 97 := by
  sorry

end total_seashells_l1283_128380


namespace arithmetic_sequence_a7_l1283_128367

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 = 2 → a 5 = 7 → a 7 = 12 := by
  sorry

end arithmetic_sequence_a7_l1283_128367


namespace complex_power_difference_l1283_128385

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_power_difference_l1283_128385


namespace remove_four_gives_desired_average_l1283_128384

def original_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def remove_number (list : List Nat) (n : Nat) : List Nat :=
  list.filter (· ≠ n)

def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem remove_four_gives_desired_average :
  average (remove_number original_list 4) = 29/4 := by
sorry

end remove_four_gives_desired_average_l1283_128384


namespace salary_change_result_l1283_128326

def initial_salary : ℝ := 2500

def raise_percentage : ℝ := 0.10

def cut_percentage : ℝ := 0.25

def final_salary : ℝ := initial_salary * (1 + raise_percentage) * (1 - cut_percentage)

theorem salary_change_result :
  final_salary = 2062.5 := by sorry

end salary_change_result_l1283_128326


namespace inequality_system_solution_l1283_128399

theorem inequality_system_solution (x : ℝ) :
  (5 * x - 2 < 3 * (x + 2)) ∧
  ((2 * x - 1) / 3 - (5 * x + 1) / 2 ≤ 1) →
  -1 ≤ x ∧ x < 4 := by
sorry

end inequality_system_solution_l1283_128399


namespace passing_marks_l1283_128334

/-- Proves that the passing marks is 120 given the conditions of the problem -/
theorem passing_marks (T : ℝ) (P : ℝ) 
  (h1 : 0.30 * T = P - 30)
  (h2 : 0.45 * T = P + 15) : P = 120 := by
  sorry

end passing_marks_l1283_128334


namespace area_of_graph_region_l1283_128394

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  |x - 80| + |y| = |x / 5|

/-- The region enclosed by the graph -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph_equation p.1 p.2}

/-- The area of the enclosed region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_graph_region :
  area_of_region = 800 :=
sorry

end area_of_graph_region_l1283_128394


namespace circle_area_l1283_128373

/-- The area of the circle defined by the equation 3x^2 + 3y^2 + 12x - 9y - 27 = 0 is 49/4 * π -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 + 12 * x - 9 * y - 27 = 0) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = r^2) ∧ 
    (π * r^2 = 49/4 * π)) := by
  sorry

end circle_area_l1283_128373


namespace computer_table_cost_price_l1283_128337

/-- The cost price of a computer table given its selling price and markup percentage. -/
def cost_price (selling_price : ℚ) (markup_percent : ℚ) : ℚ :=
  selling_price / (1 + markup_percent / 100)

/-- Theorem stating that the cost price of a computer table is 6525 
    given a selling price of 8091 and a markup of 24%. -/
theorem computer_table_cost_price :
  cost_price 8091 24 = 6525 := by
  sorry

end computer_table_cost_price_l1283_128337


namespace max_guests_l1283_128314

/-- Represents a menu choice as a quadruple of integers -/
structure MenuChoice (n : ℕ) where
  starter : Fin n
  main : Fin n
  dessert : Fin n
  wine : Fin n

/-- The set of all valid menu choices -/
def validMenus (n : ℕ) : Finset (MenuChoice n) :=
  sorry

theorem max_guests (n : ℕ) (h : n > 0) :
  (Finset.card (validMenus n) : ℕ) = n^4 - n^3 ∧
  ∀ (S : Finset (MenuChoice n)), Finset.card S > n^4 - n^3 →
    ∃ (T : Finset (MenuChoice n)), Finset.card T = n ∧ T ⊆ S ∧
      (∃ (i : Fin 4), ∀ (x y : MenuChoice n), x ∈ T → y ∈ T → x ≠ y →
        (i.val = 0 → x.starter = y.starter) ∧
        (i.val = 1 → x.main = y.main) ∧
        (i.val = 2 → x.dessert = y.dessert) ∧
        (i.val = 3 → x.wine = y.wine)) :=
  sorry

end max_guests_l1283_128314


namespace cubic_function_properties_l1283_128310

/-- The cubic function f(x) with specific properties -/
def f (x : ℝ) : ℝ := 2*x^3 - 9*x^2 + 12*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6*x^2 - 18*x + 12

theorem cubic_function_properties :
  (f 0 = -4) ∧ 
  (∀ x, f' 0 * x - (f x - f 0) - 4 = 0) ∧
  (f 2 = 0) ∧ 
  (f' 2 = 0) ∧
  (∀ x, x < 1 ∨ x > 2 → f' x > 0) :=
sorry


end cubic_function_properties_l1283_128310


namespace playground_area_l1283_128300

/-- Given a rectangular landscape with specific dimensions and a playground, 
    prove that the playground area is 3200 square meters. -/
theorem playground_area (length breadth : ℝ) (playground_area : ℝ) : 
  breadth = 8 * length →
  breadth = 480 →
  playground_area = (1 / 9) * (length * breadth) →
  playground_area = 3200 := by
  sorry

end playground_area_l1283_128300
