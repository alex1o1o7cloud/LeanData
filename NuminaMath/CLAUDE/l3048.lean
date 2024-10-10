import Mathlib

namespace line_x_intercept_l3048_304869

/-- Given a line passing through the point (3, 4) with slope 2, its x-intercept is 1. -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = 2 * x + (4 - 2 * 3)) →  -- Line equation derived from point-slope form
  f 4 = 3 →                           -- Line passes through (3, 4)
  f 0 = 1 :=                          -- x-intercept is at (1, 0)
by
  sorry

end line_x_intercept_l3048_304869


namespace ice_cream_bill_l3048_304815

/-- The final bill for four ice cream sundaes with a 20% tip -/
theorem ice_cream_bill (alicia_cost brant_cost josh_cost yvette_cost : ℚ) 
  (h1 : alicia_cost = 7.5)
  (h2 : brant_cost = 10)
  (h3 : josh_cost = 8.5)
  (h4 : yvette_cost = 9)
  (tip_rate : ℚ)
  (h5 : tip_rate = 0.2) :
  alicia_cost + brant_cost + josh_cost + yvette_cost + 
  (alicia_cost + brant_cost + josh_cost + yvette_cost) * tip_rate = 42 := by
  sorry

end ice_cream_bill_l3048_304815


namespace cube_root_function_l3048_304857

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (64 : ℝ)^(1/3) ∧ y = 8) →
  k * (27 : ℝ)^(1/3) = 6 := by
  sorry

end cube_root_function_l3048_304857


namespace angle_range_theorem_l3048_304883

theorem angle_range_theorem (θ : Real) 
  (h1 : 0 ≤ θ) (h2 : θ < 2 * Real.pi) 
  (h3 : Real.sin θ ^ 3 - Real.cos θ ^ 3 ≥ Real.cos θ - Real.sin θ) : 
  Real.pi / 4 ≤ θ ∧ θ ≤ 5 * Real.pi / 4 := by
  sorry

end angle_range_theorem_l3048_304883


namespace circle_area_equivalence_l3048_304899

theorem circle_area_equivalence (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 33) (h₂ : r₂ = 24) : 
  (π * r₁^2 - π * r₂^2 = π * r₃^2) → r₃ = 3 * Real.sqrt 57 := by
  sorry

end circle_area_equivalence_l3048_304899


namespace consecutive_numbers_digit_sum_l3048_304870

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sum_of_digits_range (start : ℕ) (count : ℕ) : ℕ :=
  List.range count |>.map (fun i => sum_of_digits (start + i)) |>.sum

theorem consecutive_numbers_digit_sum :
  ∃! start : ℕ, sum_of_digits_range start 10 = 145 ∧ start ≥ 100 ∧ start < 1000 := by
  sorry

end consecutive_numbers_digit_sum_l3048_304870


namespace female_officers_count_l3048_304820

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) 
  (female_duty_percentage : ℚ) (h1 : total_on_duty = 300) 
  (h2 : female_on_duty_ratio = 1/2) (h3 : female_duty_percentage = 15/100) : 
  ∃ (total_female : ℕ), total_female = 1000 ∧ 
  (total_on_duty : ℚ) * female_on_duty_ratio * (1/female_duty_percentage) = total_female := by
sorry

end female_officers_count_l3048_304820


namespace fixed_point_quadratic_fixed_point_satisfies_equation_l3048_304829

/-- The fixed point on the graph of y = 9x^2 + kx - 5k -/
theorem fixed_point_quadratic (k : ℝ) : 
  9 * (-3)^2 + k * (-3) - 5 * k = 81 := by
  sorry

/-- The fixed point (-3, 81) satisfies the equation for all k -/
theorem fixed_point_satisfies_equation (k : ℝ) :
  ∃ (x y : ℝ), x = -3 ∧ y = 81 ∧ y = 9 * x^2 + k * x - 5 * k := by
  sorry

end fixed_point_quadratic_fixed_point_satisfies_equation_l3048_304829


namespace sum_of_cubes_equality_l3048_304808

def original_equation (x : ℝ) : Prop :=
  x * Real.rpow x (1/3) + 4*x - 9 * Real.rpow x (1/3) + 2 = 0

def transformed_equation (y : ℝ) : Prop :=
  y^4 + 4*y^3 - 9*y + 2 = 0

def roots_original : Set ℝ :=
  {x : ℝ | original_equation x ∧ x ≥ 0}

def roots_transformed : Set ℝ :=
  {y : ℝ | transformed_equation y ∧ y ≥ 0}

theorem sum_of_cubes_equality :
  ∀ (x₁ x₂ x₃ x₄ : ℝ) (y₁ y₂ y₃ y₄ : ℝ),
  roots_original = {x₁, x₂, x₃, x₄} →
  roots_transformed = {y₁, y₂, y₃, y₄} →
  x₁^3 + x₂^3 + x₃^3 + x₄^3 = y₁^9 + y₂^9 + y₃^9 + y₄^9 :=
by sorry

end sum_of_cubes_equality_l3048_304808


namespace initial_number_of_people_l3048_304865

theorem initial_number_of_people (avg_weight_increase : ℝ) (weight_difference : ℝ) : 
  avg_weight_increase = 2.5 →
  weight_difference = 20 →
  avg_weight_increase * (weight_difference / avg_weight_increase) = weight_difference →
  (weight_difference / avg_weight_increase : ℝ) = 8 := by
  sorry

end initial_number_of_people_l3048_304865


namespace only_setB_proportional_l3048_304842

/-- A set of four line segments --/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if a set of line segments is proportional --/
def isProportional (s : LineSegmentSet) : Prop :=
  s.a * s.d = s.b * s.c

/-- The given sets of line segments --/
def setA : LineSegmentSet := ⟨3, 4, 5, 6⟩
def setB : LineSegmentSet := ⟨5, 15, 2, 6⟩
def setC : LineSegmentSet := ⟨4, 8, 3, 5⟩
def setD : LineSegmentSet := ⟨8, 4, 1, 3⟩

/-- Theorem stating that only set B is proportional --/
theorem only_setB_proportional :
  ¬ isProportional setA ∧
  isProportional setB ∧
  ¬ isProportional setC ∧
  ¬ isProportional setD :=
sorry

end only_setB_proportional_l3048_304842


namespace unique_albums_count_l3048_304841

/-- Represents a music collection -/
structure MusicCollection where
  total : ℕ
  shared : ℕ
  unique : ℕ

/-- Theorem about the number of unique albums in two collections -/
theorem unique_albums_count
  (andrew : MusicCollection)
  (john : MusicCollection)
  (h1 : andrew.total = 23)
  (h2 : andrew.shared = 11)
  (h3 : john.shared = 11)
  (h4 : john.unique = 8)
  : andrew.unique + john.unique = 20 := by
  sorry

end unique_albums_count_l3048_304841


namespace shina_probability_l3048_304810

def word : Finset Char := {'М', 'А', 'Ш', 'И', 'Н', 'А'}

def draw_probability (word : Finset Char) (target : List Char) : ℚ :=
  let n := word.card
  let prob := target.foldl (λ acc c =>
    acc * (word.filter (λ x => x = c)).card / n) 1
  prob * (n - 1) * (n - 2) * (n - 3) / n

theorem shina_probability :
  draw_probability word ['Ш', 'И', 'Н', 'А'] = 1 / 180 := by
  sorry

end shina_probability_l3048_304810


namespace brianna_cd_purchase_l3048_304851

theorem brianna_cd_purchase (total_money : ℚ) (total_cds : ℚ) (h : total_money > 0) (h' : total_cds > 0) :
  (1 / 4 : ℚ) * total_money = (1 / 4 : ℚ) * (total_cds * (total_money / total_cds)) →
  total_money - (total_cds * (total_money / total_cds)) = 0 := by
sorry

end brianna_cd_purchase_l3048_304851


namespace daily_class_schedule_l3048_304879

theorem daily_class_schedule (n m : ℕ) (hn : n = 10) (hm : m = 6) :
  (n.factorial / (n - m).factorial) = 151200 :=
sorry

end daily_class_schedule_l3048_304879


namespace ten_thousand_equals_10000_l3048_304894

theorem ten_thousand_equals_10000 : (10 * 1000 : ℕ) = 10000 := by
  sorry

end ten_thousand_equals_10000_l3048_304894


namespace odometer_sum_squares_l3048_304821

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds + tens + ones ≤ 9

/-- Represents a car trip -/
structure CarTrip where
  hours : Nat
  speed : Nat
  initial : OdometerReading
  final : OdometerReading
  valid : speed = 65 ∧
          final.hundreds = initial.ones ∧
          final.tens = initial.tens ∧
          final.ones = initial.hundreds

theorem odometer_sum_squares (trip : CarTrip) :
  trip.initial.hundreds ^ 2 + trip.initial.tens ^ 2 + trip.initial.ones ^ 2 = 41 :=
sorry

end odometer_sum_squares_l3048_304821


namespace units_digit_of_2009_pow_2008_plus_2013_l3048_304807

theorem units_digit_of_2009_pow_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 := by
  sorry

end units_digit_of_2009_pow_2008_plus_2013_l3048_304807


namespace no_integer_solutions_l3048_304814

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^4 + y^2 = 4*y + 4 := by
  sorry

end no_integer_solutions_l3048_304814


namespace line_cannot_contain_point_l3048_304806

theorem line_cannot_contain_point (m b : ℝ) (h : m * b < 0) :
  ¬∃ x y : ℝ, x = -2022 ∧ y = 0 ∧ y = m * x + b :=
by sorry

end line_cannot_contain_point_l3048_304806


namespace quadratic_function_properties_l3048_304831

/-- Given a quadratic function y = -x^2 + 8x - 7 -/
def f (x : ℝ) : ℝ := -x^2 + 8*x - 7

theorem quadratic_function_properties :
  /- (1) y increases as x increases for x < 4 -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 4 → f x₁ < f x₂) ∧
  /- (2) y < 0 for x < 1 or x > 7 -/
  (∀ x : ℝ, (x < 1 ∨ x > 7) → f x < 0) :=
sorry


end quadratic_function_properties_l3048_304831


namespace paiges_pencils_l3048_304880

/-- Paige's pencil problem -/
theorem paiges_pencils (P : ℕ) : 
  P - (P - 15) / 4 + 16 - 12 + 23 = 84 → P = 71 := by
  sorry

end paiges_pencils_l3048_304880


namespace math_class_students_count_l3048_304884

theorem math_class_students_count :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 6 ∧ n % 5 = 1 ∧ n = 46 := by
  sorry

end math_class_students_count_l3048_304884


namespace pure_imaginary_m_value_l3048_304887

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 3) (m - 1)

theorem pure_imaginary_m_value :
  ∀ m : ℝ, IsPureImaginary (z m) → m = -3 := by
  sorry

end pure_imaginary_m_value_l3048_304887


namespace max_pages_proof_l3048_304830

/-- The cost in cents to copy 5 pages -/
def cost_per_5_pages : ℚ := 8

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 1 / 10

/-- The available money in dollars -/
def available_money : ℚ := 30

/-- The maximum number of pages that can be copied -/
def max_pages : ℕ := 1687

theorem max_pages_proof :
  let discounted_money : ℚ := available_money * 100 * (1 - discount_rate)
  let pages_per_cent : ℚ := 5 / cost_per_5_pages
  ⌊discounted_money * pages_per_cent⌋ = max_pages := by
  sorry

end max_pages_proof_l3048_304830


namespace professor_count_proof_l3048_304833

/-- The number of professors in the first year -/
def initial_professors : ℕ := 5

/-- The number of failing grades given in the first year -/
def first_year_grades : ℕ := 6480

/-- The number of failing grades given in the second year -/
def second_year_grades : ℕ := 11200

/-- The increase in the number of professors in the second year -/
def professor_increase : ℕ := 3

theorem professor_count_proof :
  (first_year_grades % initial_professors = 0) ∧
  (second_year_grades % (initial_professors + professor_increase) = 0) ∧
  (first_year_grades / initial_professors < second_year_grades / (initial_professors + professor_increase)) ∧
  (∀ p : ℕ, p < initial_professors →
    (first_year_grades % p = 0 ∧ second_year_grades % (p + professor_increase) = 0) →
    (first_year_grades / p < second_year_grades / (p + professor_increase)) →
    False) :=
by
  sorry

end professor_count_proof_l3048_304833


namespace sum_to_all_ones_implies_digit_five_or_greater_l3048_304819

/-- A function that checks if a natural number has no zero digits -/
def hasNoZeroDigits (n : ℕ) : Prop := sorry

/-- A function that generates all digit permutations of a natural number -/
def digitPermutations (n : ℕ) : Finset ℕ := sorry

/-- A function that checks if a natural number consists only of digit 1 -/
def isAllOnes (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has at least one digit 5 or greater -/
def hasDigitFiveOrGreater (n : ℕ) : Prop := sorry

/-- Theorem stating that if a number without zero digits and three of its permutations sum to all ones, it must have a digit 5 or greater -/
theorem sum_to_all_ones_implies_digit_five_or_greater (n : ℕ) :
  hasNoZeroDigits n →
  ∃ (p q r : ℕ), p ∈ digitPermutations n ∧ q ∈ digitPermutations n ∧ r ∈ digitPermutations n ∧
  isAllOnes (n + p + q + r) →
  hasDigitFiveOrGreater n :=
sorry

end sum_to_all_ones_implies_digit_five_or_greater_l3048_304819


namespace two_tangent_lines_l3048_304890

/-- A line that intersects a parabola at exactly one point -/
structure TangentLine where
  slope : ℝ
  y_intercept : ℝ

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The point M(2, 4) -/
def M : Point := ⟨2, 4⟩

/-- A line passes through a point -/
def passes_through (l : TangentLine) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- A line intersects the parabola at exactly one point -/
def intersects_once (l : TangentLine) : Prop :=
  ∃! (p : Point), passes_through l p ∧ parabola p.x p.y

/-- There are exactly two lines passing through M(2, 4) that intersect the parabola at exactly one point -/
theorem two_tangent_lines : ∃! (l1 l2 : TangentLine), 
  l1 ≠ l2 ∧ 
  passes_through l1 M ∧ 
  passes_through l2 M ∧ 
  intersects_once l1 ∧ 
  intersects_once l2 :=
sorry

end two_tangent_lines_l3048_304890


namespace dog_catches_rabbit_l3048_304855

/-- Represents the chase scenario between a dog and a rabbit -/
structure ChaseScenario where
  rabbit_head_start : ℕ
  rabbit_distance_ratio : ℕ
  dog_distance_ratio : ℕ
  rabbit_time_ratio : ℕ
  dog_time_ratio : ℕ

/-- Calculates the minimum number of steps the dog must run to catch the rabbit -/
def min_steps_to_catch (scenario : ChaseScenario) : ℕ :=
  sorry

/-- Theorem stating that given the specific chase scenario, the dog needs 240 steps to catch the rabbit -/
theorem dog_catches_rabbit :
  let scenario : ChaseScenario := {
    rabbit_head_start := 100,
    rabbit_distance_ratio := 8,
    dog_distance_ratio := 3,
    rabbit_time_ratio := 9,
    dog_time_ratio := 4
  }
  min_steps_to_catch scenario = 240 := by
  sorry

end dog_catches_rabbit_l3048_304855


namespace mars_bars_count_l3048_304816

theorem mars_bars_count (total : ℕ) (snickers : ℕ) (butterfingers : ℕ) 
  (h1 : total = 12)
  (h2 : snickers = 3)
  (h3 : butterfingers = 7) :
  total - snickers - butterfingers = 2 := by
  sorry

end mars_bars_count_l3048_304816


namespace two_hundred_fiftieth_letter_l3048_304802

def repeating_pattern : ℕ → Char
  | n => match n % 3 with
         | 0 => 'C'
         | 1 => 'A'
         | _ => 'B'

theorem two_hundred_fiftieth_letter : repeating_pattern 250 = 'A' := by
  sorry

end two_hundred_fiftieth_letter_l3048_304802


namespace set_D_forms_triangle_l3048_304877

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem set_D_forms_triangle :
  can_form_triangle 10 10 5 := by
  sorry

end set_D_forms_triangle_l3048_304877


namespace parabola_x_intercepts_l3048_304850

theorem parabola_x_intercepts :
  let f (x : ℝ) := 3 * x^2 + 5 * x - 8
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∀ (x₁ x₂ x₃ : ℝ), f x₁ = 0 → f x₂ = 0 → f x₃ = 0 → x₁ = x₂ ∨ x₁ = x₃ ∨ x₂ = x₃) := by
  sorry

end parabola_x_intercepts_l3048_304850


namespace green_pill_cost_l3048_304882

theorem green_pill_cost (weeks : ℕ) (daily_green : ℕ) (daily_pink : ℕ) 
  (green_pink_diff : ℚ) (total_cost : ℚ) :
  weeks = 3 →
  daily_green = 1 →
  daily_pink = 1 →
  green_pink_diff = 3 →
  total_cost = 819 →
  ∃ (green_cost : ℚ), 
    green_cost = 21 ∧ 
    (weeks * 7 * (green_cost + (green_cost - green_pink_diff))) = total_cost :=
by sorry

end green_pill_cost_l3048_304882


namespace prob_green_face_specific_cube_l3048_304848

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  yellow_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def prob_green_face (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a cube with 5 green faces and 1 yellow face is 5/6 -/
theorem prob_green_face_specific_cube :
  let cube : ColoredCube := ⟨6, 5, 1⟩
  prob_green_face cube = 5 / 6 := by
  sorry

end prob_green_face_specific_cube_l3048_304848


namespace principal_is_12000_l3048_304858

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  interest / (rate * time.cast / 100)

/-- Theorem stating that given the specified conditions, the principal amount is $12000. -/
theorem principal_is_12000 (rate : ℚ) (time : ℕ) (interest : ℚ) 
  (h_rate : rate = 12)
  (h_time : time = 3)
  (h_interest : interest = 4320) :
  calculate_principal rate time interest = 12000 := by
  sorry

#eval calculate_principal 12 3 4320

end principal_is_12000_l3048_304858


namespace fourth_circle_radius_l3048_304827

theorem fourth_circle_radius (r₁ r₂ r : ℝ) (h₁ : r₁ = 17) (h₂ : r₂ = 27) :
  π * r^2 = π * (r₂^2 - r₁^2) → r = 2 * Real.sqrt 110 := by
  sorry

end fourth_circle_radius_l3048_304827


namespace concentric_circles_k_value_l3048_304863

/-- Two concentric circles with center at the origin --/
structure ConcentricCircles where
  largeRadius : ℝ
  smallRadius : ℝ

/-- The point P on the larger circle --/
def P : ℝ × ℝ := (10, 6)

/-- The point S on the smaller circle --/
def S (k : ℝ) : ℝ × ℝ := (0, k)

/-- The distance QR --/
def QR : ℝ := 4

theorem concentric_circles_k_value (circles : ConcentricCircles) 
  (h1 : circles.largeRadius ^ 2 = P.1 ^ 2 + P.2 ^ 2)
  (h2 : circles.smallRadius = circles.largeRadius - QR)
  (h3 : (S k).2 = circles.smallRadius) :
  k = 2 * Real.sqrt 34 - 4 := by
  sorry

end concentric_circles_k_value_l3048_304863


namespace bobby_has_more_books_l3048_304836

/-- Given that Bobby has 142 books and Kristi has 78 books, 
    prove that Bobby has 64 more books than Kristi. -/
theorem bobby_has_more_books : 
  let bobby_books : ℕ := 142
  let kristi_books : ℕ := 78
  bobby_books - kristi_books = 64 := by sorry

end bobby_has_more_books_l3048_304836


namespace congruence_systems_solvability_l3048_304878

theorem congruence_systems_solvability :
  (∃ x : ℤ, x ≡ 2 [ZMOD 3] ∧ x ≡ 6 [ZMOD 14]) ∧
  (¬ ∃ x : ℤ, x ≡ 5 [ZMOD 12] ∧ x ≡ 7 [ZMOD 15]) ∧
  (∃ x : ℤ, x ≡ 10 [ZMOD 12] ∧ x ≡ 16 [ZMOD 21]) :=
by sorry

end congruence_systems_solvability_l3048_304878


namespace fraction_ratio_equality_l3048_304809

theorem fraction_ratio_equality : ∃ (x y : ℚ), 
  (x / y) / (7 / 15) = ((5 / 3) / ((2 / 3) - (1 / 4))) / ((1 / 3 + 1 / 6) / (1 / 2 - 1 / 3)) ∧
  x / y = 28 / 45 := by
  sorry

end fraction_ratio_equality_l3048_304809


namespace quadratic_properties_l3048_304845

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Theorem statement
theorem quadratic_properties :
  (∀ x, f x = (x - 2)^2 - 1) ∧
  (∀ x, f x ≥ f 2) ∧
  (f 2 = -1) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, ∀ y ∈ Set.Ioc 2 3, f x > f y) ∧
  (∀ y ∈ Set.Icc (-1 : ℝ) 8, ∃ x ∈ Set.Ico (-1 : ℝ) 3, f x = y) :=
by sorry

end quadratic_properties_l3048_304845


namespace pure_imaginary_condition_l3048_304824

theorem pure_imaginary_condition (i a : ℂ) : 
  i^2 = -1 →
  (((i^2 + a*i) / (1 + i)).re = 0) →
  a = 1 := by
sorry

end pure_imaginary_condition_l3048_304824


namespace negative_two_hash_negative_seven_l3048_304838

/-- The # operation for rational numbers -/
def hash (a b : ℚ) : ℚ := a * b + 1

/-- Theorem stating that (-2) # (-7) = 15 -/
theorem negative_two_hash_negative_seven :
  hash (-2) (-7) = 15 := by
  sorry

end negative_two_hash_negative_seven_l3048_304838


namespace trigonometric_identities_l3048_304823

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 3) :
  (sin α + 3 * cos α) / (2 * sin α + 5 * cos α) = 6 / 11 ∧
  sin α ^ 2 + sin α * cos α + 3 * cos α ^ 2 = 3 / 2 := by
sorry

end trigonometric_identities_l3048_304823


namespace infinite_series_not_computable_l3048_304801

/-- An infinite series of natural numbers -/
def infinite_series (n : ℕ) : ℕ := n

/-- A predicate indicating whether a series can be computed algorithmically -/
def is_algorithmically_computable (f : ℕ → ℕ) : Prop :=
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → f n = 0

/-- The theorem stating that the infinite series cannot be computed algorithmically -/
theorem infinite_series_not_computable :
  ¬ (is_algorithmically_computable infinite_series) := by
  sorry

end infinite_series_not_computable_l3048_304801


namespace jesse_pencils_l3048_304817

theorem jesse_pencils (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  given = 44 → remaining = 34 → initial = given + remaining :=
by
  sorry

end jesse_pencils_l3048_304817


namespace correct_average_l3048_304897

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 18 ∧ incorrect_num = 26 ∧ correct_num = 66 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 22 := by
  sorry

end correct_average_l3048_304897


namespace decimal_20_equals_base4_110_l3048_304839

/-- Converts a decimal number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem: The decimal number 20 is equivalent to 110 in base 4 -/
theorem decimal_20_equals_base4_110 : toBase4 20 = [1, 1, 0] := by
  sorry

end decimal_20_equals_base4_110_l3048_304839


namespace total_minutes_played_l3048_304867

/-- The number of days in 2 weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of gigs Mark does in 2 weeks -/
def gigs_in_two_weeks : ℕ := days_in_two_weeks / 2

/-- The number of songs Mark plays in each gig -/
def songs_per_gig : ℕ := 3

/-- The duration of the first two songs in minutes -/
def short_song_duration : ℕ := 5

/-- The duration of the last song in minutes -/
def long_song_duration : ℕ := 2 * short_song_duration

/-- The total duration of all songs in one gig in minutes -/
def duration_per_gig : ℕ := 2 * short_song_duration + long_song_duration

/-- The theorem stating the total number of minutes Mark played in 2 weeks -/
theorem total_minutes_played : gigs_in_two_weeks * duration_per_gig = 140 := by
  sorry

end total_minutes_played_l3048_304867


namespace model2_best_fit_l3048_304891

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  name : String
  r_squared : Float

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

/-- The list of regression models with their R² values -/
def regression_models : List RegressionModel := [
  ⟨"Model 1", 0.78⟩,
  ⟨"Model 2", 0.85⟩,
  ⟨"Model 3", 0.61⟩,
  ⟨"Model 4", 0.31⟩
]

/-- Theorem stating that Model 2 has the best fitting effect -/
theorem model2_best_fit :
  ∃ model ∈ regression_models, model.name = "Model 2" ∧ has_best_fit model regression_models :=
by
  sorry

end model2_best_fit_l3048_304891


namespace bert_spending_l3048_304895

theorem bert_spending (n : ℚ) : 
  (1/2 * ((3/4 * n) - 9)) = 15 → n = 52 := by
  sorry

end bert_spending_l3048_304895


namespace water_tank_problem_l3048_304853

/-- A water tank problem during the rainy season -/
theorem water_tank_problem (tank_capacity : ℝ) (initial_fill_fraction : ℝ) 
  (day1_collection : ℝ) (day2_extra : ℝ) :
  tank_capacity = 100 →
  initial_fill_fraction = 2/5 →
  day1_collection = 15 →
  day2_extra = 5 →
  let initial_water := initial_fill_fraction * tank_capacity
  let day1_total := initial_water + day1_collection
  let day2_collection := day1_collection + day2_extra
  let day2_total := day1_total + day2_collection
  let day3_collection := tank_capacity - day2_total
  day3_collection = 25 := by
sorry


end water_tank_problem_l3048_304853


namespace inequality_always_true_l3048_304840

theorem inequality_always_true (a b : ℝ) (h : a * b > 0) :
  b / a + a / b ≥ 2 := by sorry

end inequality_always_true_l3048_304840


namespace floor_equality_l3048_304854

theorem floor_equality (n : ℤ) (h : n > 2) :
  ⌊(n * (n + 1) : ℚ) / (4 * n - 2)⌋ = ⌊(n + 1 : ℚ) / 4⌋ := by
  sorry

end floor_equality_l3048_304854


namespace trout_division_l3048_304813

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 52 → num_people = 4 → trout_per_person = total_trout / num_people → trout_per_person = 13 := by
  sorry

end trout_division_l3048_304813


namespace evaluate_expression_l3048_304876

theorem evaluate_expression : (0.5^4 / 0.05^3) + 3 = 503 := by
  sorry

end evaluate_expression_l3048_304876


namespace sum_odd_numbers_less_than_20_l3048_304868

theorem sum_odd_numbers_less_than_20 : 
  (Finset.range 10).sum (fun n => 2 * n + 1) = 100 := by
  sorry

end sum_odd_numbers_less_than_20_l3048_304868


namespace total_bottles_l3048_304844

theorem total_bottles (regular : ℕ) (diet : ℕ) (lite : ℕ)
  (h1 : regular = 57)
  (h2 : diet = 26)
  (h3 : lite = 27) :
  regular + diet + lite = 110 := by
  sorry

end total_bottles_l3048_304844


namespace karen_average_speed_l3048_304874

/-- Calculates the time difference in hours between two times given in hours and minutes -/
def timeDifference (start_hour start_minute end_hour end_minute : ℕ) : ℚ :=
  (end_hour - start_hour : ℚ) + (end_minute - start_minute : ℚ) / 60

/-- Calculates the average speed given distance and time -/
def averageSpeed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

theorem karen_average_speed :
  let start_time : ℕ × ℕ := (9, 40)  -- (hour, minute)
  let end_time : ℕ × ℕ := (13, 20)   -- (hour, minute)
  let distance : ℚ := 198
  let time := timeDifference start_time.1 start_time.2 end_time.1 end_time.2
  averageSpeed distance time = 54 := by sorry

end karen_average_speed_l3048_304874


namespace uncle_welly_roses_l3048_304818

/-- The number of roses Uncle Welly planted two days ago -/
def roses_two_days_ago : ℕ := 50

/-- The number of roses Uncle Welly planted yesterday -/
def roses_yesterday : ℕ := roses_two_days_ago + 20

/-- The number of roses Uncle Welly planted today -/
def roses_today : ℕ := 2 * roses_two_days_ago

/-- The total number of roses Uncle Welly planted in his vacant lot -/
def total_roses : ℕ := roses_two_days_ago + roses_yesterday + roses_today

theorem uncle_welly_roses : total_roses = 220 := by
  sorry

end uncle_welly_roses_l3048_304818


namespace square_perimeter_ratio_l3048_304846

theorem square_perimeter_ratio (s S : ℝ) (hs : s > 0) (hS : S > 0) : 
  S * Real.sqrt 2 = 3 * (s * Real.sqrt 2) → 4 * S / (4 * s) = 3 := by
  sorry

end square_perimeter_ratio_l3048_304846


namespace green_balloons_l3048_304864

theorem green_balloons (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by sorry

end green_balloons_l3048_304864


namespace arithmetic_calculation_l3048_304826

theorem arithmetic_calculation : (20 * 24) / (2 * 0 + 2 * 4) = 60 := by
  sorry

end arithmetic_calculation_l3048_304826


namespace odd_function_sum_l3048_304804

def f (x a b : ℝ) : ℝ := (x - 1)^2 + a * x^2 + b

theorem odd_function_sum (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) → a + b = -2 :=
by
  sorry

end odd_function_sum_l3048_304804


namespace time_per_furniture_piece_l3048_304856

theorem time_per_furniture_piece (chairs tables total_time : ℕ) 
  (h1 : chairs = 4)
  (h2 : tables = 2)
  (h3 : total_time = 48) : 
  total_time / (chairs + tables) = 8 := by
  sorry

end time_per_furniture_piece_l3048_304856


namespace cary_calorie_deficit_l3048_304811

/-- Calculates the net calorie deficit for Cary's grocery store trip -/
theorem cary_calorie_deficit :
  let miles_walked : ℕ := 3
  let calories_per_mile : ℕ := 150
  let candy_bar_calories : ℕ := 200
  let total_calories_burned := miles_walked * calories_per_mile
  let net_deficit := total_calories_burned - candy_bar_calories
  net_deficit = 250 := by sorry

end cary_calorie_deficit_l3048_304811


namespace notebook_cost_per_page_l3048_304866

/-- Calculates the cost per page in cents given the number of notebooks, pages per notebook, and total cost in dollars. -/
def cost_per_page (notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (notebooks * pages_per_notebook)

/-- Proves that for 2 notebooks with 50 pages each, purchased for $5, the cost per page is 5 cents. -/
theorem notebook_cost_per_page :
  cost_per_page 2 50 5 = 5 := by
  sorry

end notebook_cost_per_page_l3048_304866


namespace machines_count_l3048_304885

/-- Given that n machines produce x units in 6 days and 12 machines produce 3x units in 6 days,
    where all machines work at an identical constant rate, prove that n = 4. -/
theorem machines_count (n : ℕ) (x : ℝ) (h1 : x > 0) :
  (n * x / 6 = x / 6) →
  (12 * (3 * x) / 6 = 3 * x / 6) →
  (n * x / (6 * n) = 12 * (3 * x) / (6 * 12)) →
  n = 4 :=
sorry

end machines_count_l3048_304885


namespace gcd_digits_bound_l3048_304835

theorem gcd_digits_bound (a b : ℕ) (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7)
  (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 := by
  sorry

end gcd_digits_bound_l3048_304835


namespace calculator_cost_ratio_l3048_304862

theorem calculator_cost_ratio :
  ∀ (basic scientific graphing : ℝ),
  basic = 8 →
  graphing = 3 * scientific →
  100 - (basic + scientific + graphing) = 28 →
  scientific / basic = 2 := by
sorry

end calculator_cost_ratio_l3048_304862


namespace jump_rope_challenge_l3048_304849

structure Jumper where
  initialRate : ℝ
  breakPatterns : List (ℝ × ℝ)
  speedChanges : List ℝ

def calculateSkips (j : Jumper) (totalTime : ℝ) : ℝ :=
  sorry

theorem jump_rope_challenge :
  let leah : Jumper := {
    initialRate := 5,
    breakPatterns := [(120, 20), (120, 25), (120, 30)],
    speedChanges := [0.5, 0.5, 0.5]
  }
  let matt : Jumper := {
    initialRate := 3,
    breakPatterns := [(180, 15), (180, 15)],
    speedChanges := [-0.25, -0.25]
  }
  let linda : Jumper := {
    initialRate := 4,
    breakPatterns := [(240, 10), (240, 15)],
    speedChanges := [-0.1, 0.2]
  }
  let totalTime : ℝ := 600
  (calculateSkips leah totalTime = 3540) ∧
  (calculateSkips matt totalTime = 1635) ∧
  (calculateSkips linda totalTime = 2412) ∧
  (calculateSkips leah totalTime + calculateSkips matt totalTime + calculateSkips linda totalTime = 7587) :=
by
  sorry

end jump_rope_challenge_l3048_304849


namespace min_value_theorem_min_value_achievable_l3048_304886

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 / Real.sqrt 5 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 9 / Real.sqrt 5 := by
  sorry

end min_value_theorem_min_value_achievable_l3048_304886


namespace locus_all_importance_l3048_304881

/-- Definition of a locus --/
def Locus (P : Type*) (condition : P → Prop) : Set P :=
  {p : P | condition p}

/-- Property of comprehensiveness --/
def Comprehensive (S : Set P) (condition : P → Prop) : Prop :=
  ∀ p, condition p → p ∈ S

/-- Property of exclusivity --/
def Exclusive (S : Set P) (condition : P → Prop) : Prop :=
  ∀ p, p ∈ S → condition p

/-- Theorem: The definition of locus ensures both comprehensiveness and exclusivity --/
theorem locus_all_importance {P : Type*} (condition : P → Prop) :
  let L := Locus P condition
  Comprehensive L condition ∧ Exclusive L condition := by
  sorry

end locus_all_importance_l3048_304881


namespace june_science_book_price_l3048_304898

/-- Calculates the price of each science book given June's school supply purchases. -/
theorem june_science_book_price (total_budget : ℕ) (math_book_price : ℕ) (math_book_count : ℕ)
  (art_book_price : ℕ) (music_book_cost : ℕ) :
  total_budget = 500 →
  math_book_price = 20 →
  math_book_count = 4 →
  art_book_price = 20 →
  music_book_cost = 160 →
  let science_book_count := math_book_count + 6
  let art_book_count := 2 * math_book_count
  let total_spent := math_book_price * math_book_count +
                     art_book_price * art_book_count +
                     music_book_cost
  let remaining_budget := total_budget - total_spent
  remaining_budget / science_book_count = 10 := by
  sorry

end june_science_book_price_l3048_304898


namespace thirteen_to_six_div_three_l3048_304843

theorem thirteen_to_six_div_three (x : ℕ) : 13^6 / 13^3 = 2197 := by
  sorry

end thirteen_to_six_div_three_l3048_304843


namespace inverse_variation_problem_l3048_304860

theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k) 
  (h4 : 2^3 * 8 = 64) : 
  (x^3 * 64 = 64) → x = 1 := by
sorry

end inverse_variation_problem_l3048_304860


namespace sqrt_sum_equals_seven_l3048_304896

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_equals_seven_l3048_304896


namespace consecutive_even_integers_sum_l3048_304805

theorem consecutive_even_integers_sum (a : ℤ) : 
  (∃ b c d : ℤ, 
    b = a + 2 ∧ 
    c = a + 4 ∧ 
    d = a + 6 ∧ 
    a % 2 = 0 ∧ 
    a + c = 146) →
  a + (a + 2) + (a + 4) + (a + 6) = 296 := by
sorry

end consecutive_even_integers_sum_l3048_304805


namespace bobs_remaining_amount_l3048_304803

/-- Calculates the remaining amount after Bob's spending over three days. -/
def remaining_amount (initial : ℚ) (mon_frac : ℚ) (tue_frac : ℚ) (wed_frac : ℚ) : ℚ :=
  let after_mon := initial * (1 - mon_frac)
  let after_tue := after_mon * (1 - tue_frac)
  after_tue * (1 - wed_frac)

/-- Theorem stating that Bob's remaining amount is $20 after three days of spending. -/
theorem bobs_remaining_amount :
  remaining_amount 80 (1/2) (1/5) (3/8) = 20 := by
  sorry

end bobs_remaining_amount_l3048_304803


namespace greatest_value_quadratic_inequality_l3048_304852

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, a^2 - 12*a + 35 ≤ 0 → a ≤ 7 ∧
  ∃ a : ℝ, a^2 - 12*a + 35 ≤ 0 ∧ a = 7 :=
by sorry

end greatest_value_quadratic_inequality_l3048_304852


namespace certain_number_problem_l3048_304832

theorem certain_number_problem (N x : ℝ) (h1 : N / x * 2 = 12) (h2 : x = 0.1) : N = 0.6 := by
  sorry

end certain_number_problem_l3048_304832


namespace function_equation_implies_constant_l3048_304893

/-- A function satisfying the given functional equation is constant -/
theorem function_equation_implies_constant
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end function_equation_implies_constant_l3048_304893


namespace walnut_trees_in_park_l3048_304871

theorem walnut_trees_in_park (initial_trees new_trees : ℕ) : 
  initial_trees = 4 → new_trees = 6 → initial_trees + new_trees = 10 := by
  sorry

end walnut_trees_in_park_l3048_304871


namespace speed_increase_for_time_reduction_car_speed_increase_l3048_304861

/-- Calculates the required speed increase for a car to reduce its travel time --/
theorem speed_increase_for_time_reduction 
  (initial_speed : ℝ) 
  (distance : ℝ) 
  (time_reduction : ℝ) : ℝ :=
  let initial_time := distance / initial_speed
  let final_time := initial_time - time_reduction
  let final_speed := distance / final_time
  final_speed - initial_speed

/-- Proves that a car traveling at 60 km/h needs to increase its speed by 60 km/h
    to travel 1 km in half a minute less time --/
theorem car_speed_increase : 
  speed_increase_for_time_reduction 60 1 (1/120) = 60 := by
  sorry

end speed_increase_for_time_reduction_car_speed_increase_l3048_304861


namespace infinite_cube_differences_l3048_304875

theorem infinite_cube_differences (n : ℕ+) : 
  (∃ p : ℕ+, 3 * p + 1 = (n + 1)^3 - n^3) ∧ 
  (∃ q : ℕ+, 5 * q + 1 = (5 * n + 1)^3 - (5 * n)^3) := by
  sorry

end infinite_cube_differences_l3048_304875


namespace ratio_problem_l3048_304892

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 2) (h2 : c/b = 3) : (a + b) / (b + c) = 3/8 := by
  sorry

end ratio_problem_l3048_304892


namespace quadratic_equation_solution_l3048_304872

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -5
  let c : ℝ := 3
  let x₁ : ℝ := 3/2
  let x₂ : ℝ := 1
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end quadratic_equation_solution_l3048_304872


namespace sum_of_coefficients_l3048_304812

theorem sum_of_coefficients : 
  let p (x : ℝ) := -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)
  p 1 = 45 := by sorry

end sum_of_coefficients_l3048_304812


namespace min_dot_product_planar_vectors_l3048_304847

/-- Given planar vectors a and b satisfying |2a - b| ≤ 3, 
    the minimum value of a · b is -9/8 -/
theorem min_dot_product_planar_vectors 
  (a b : ℝ × ℝ) 
  (h : ‖(2 : ℝ) • a - b‖ ≤ 3) : 
  ∃ (m : ℝ), m = -9/8 ∧ ∀ (x : ℝ), x = a.1 * b.1 + a.2 * b.2 → m ≤ x :=
sorry

end min_dot_product_planar_vectors_l3048_304847


namespace krista_hens_count_l3048_304873

def egg_price_per_dozen : ℚ := 3
def total_sales : ℚ := 120
def weeks : ℕ := 4
def eggs_per_hen_per_week : ℕ := 12

def num_hens : ℕ := 10

theorem krista_hens_count :
  (egg_price_per_dozen * (total_sales / egg_price_per_dozen) = 
   ↑num_hens * ↑eggs_per_hen_per_week * ↑weeks) := by sorry

end krista_hens_count_l3048_304873


namespace system_solution_l3048_304889

theorem system_solution (x y : ℝ) (eq1 : 2 * x - y = -1) (eq2 : x + 4 * y = 22) : x + y = 7 := by
  sorry

end system_solution_l3048_304889


namespace binomial_half_variance_l3048_304834

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

/-- The main theorem -/
theorem binomial_half_variance (X : BinomialVariable) 
  (h2 : X.n = 8) (h3 : X.p = 3/5) : 
  variance X * (1/2)^2 = 12/25 := by sorry

end binomial_half_variance_l3048_304834


namespace det_of_specific_matrix_l3048_304859

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]
  Matrix.det A = 32 := by
sorry

end det_of_specific_matrix_l3048_304859


namespace integral_even_function_l3048_304888

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem integral_even_function 
  (f : ℝ → ℝ) 
  (h1 : EvenFunction f) 
  (h2 : ∫ x in (0:ℝ)..6, f x = 8) : 
  ∫ x in (-6:ℝ)..6, f x = 16 := by
  sorry

end integral_even_function_l3048_304888


namespace min_value_of_f_in_interval_l3048_304822

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-1, 2]
def interval : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem min_value_of_f_in_interval : 
  ∃ (x : ℝ), x ∈ interval ∧ f x = -2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end min_value_of_f_in_interval_l3048_304822


namespace elephant_pig_equivalence_l3048_304825

variable (P Q : Prop)

theorem elephant_pig_equivalence :
  (P → Q) →
  ((P → Q) ↔ (¬Q → ¬P)) ∧
  ((P → Q) ↔ (¬P ∨ Q)) ∧
  ¬((P → Q) ↔ (Q → P)) :=
by sorry

end elephant_pig_equivalence_l3048_304825


namespace percentage_of_75_to_125_l3048_304800

theorem percentage_of_75_to_125 : (75 : ℝ) / 125 * 100 = 60 := by
  sorry

end percentage_of_75_to_125_l3048_304800


namespace pure_imaginary_complex_number_l3048_304828

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (2 * m^2 + m - 1) (-m^2 - 2*m - 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = 1/2 := by
  sorry

end pure_imaginary_complex_number_l3048_304828


namespace equation_real_solution_l3048_304837

theorem equation_real_solution (x : ℝ) :
  (∀ y : ℝ, ∃ z : ℝ, x^2 + y^2 + z^2 + 2*x*y*z = 1) ↔ (x = 1 ∨ x = -1) := by
  sorry

end equation_real_solution_l3048_304837
