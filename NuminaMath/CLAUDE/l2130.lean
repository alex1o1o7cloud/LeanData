import Mathlib

namespace least_subtraction_for_divisibility_l2130_213052

theorem least_subtraction_for_divisibility (n m : ℕ) (h : n = 45678 ∧ m = 47) :
  ∃ k : ℕ, k ≤ m - 1 ∧ (n - k) % m = 0 ∧ ∀ j : ℕ, j < k → (n - j) % m ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_l2130_213052


namespace geometric_sequence_fifth_term_l2130_213043

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given conditions for the geometric sequence -/
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 = 10 ∧ a 2 + a 4 = 5

theorem geometric_sequence_fifth_term (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_cond : sequence_conditions a) : 
  a 5 = 1/2 := by
sorry

end geometric_sequence_fifth_term_l2130_213043


namespace max_value_theorem_l2130_213056

theorem max_value_theorem (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h1 : a^2 + b^2 - c^2 - d^2 = 0)
  (h2 : a^2 - b^2 - c^2 + d^2 = 56/53 * (b*c + a*d)) :
  (∀ x y z w : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 
    x^2 + y^2 - z^2 - w^2 = 0 ∧
    x^2 - y^2 - z^2 + w^2 = 56/53 * (y*z + x*w) →
    (x*y + z*w) / (y*z + x*w) ≤ (a*b + c*d) / (b*c + a*d)) ∧
  (a*b + c*d) / (b*c + a*d) = 45/53 :=
sorry

end max_value_theorem_l2130_213056


namespace min_sum_squares_min_sum_squares_attained_l2130_213001

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 4*x₂ + 6*x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 350 := by
  sorry

theorem min_sum_squares_attained (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 4*x₂ + 6*x₃ = 120) : 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 2*y₁ + 4*y₂ + 6*y₃ = 120 ∧ 
  y₁^2 + y₂^2 + y₃^2 = 350 := by
  sorry

end min_sum_squares_min_sum_squares_attained_l2130_213001


namespace conditional_prob_is_two_thirds_l2130_213025

/-- The sample space for two coin flips -/
def S : Finset (Fin 2 × Fin 2) := Finset.univ

/-- Event A: at least one tail shows up -/
def A : Finset (Fin 2 × Fin 2) := {(0, 1), (1, 0), (1, 1)}

/-- Event B: exactly one head shows up -/
def B : Finset (Fin 2 × Fin 2) := {(0, 1), (1, 0)}

/-- The probability measure for the sample space -/
def P (E : Finset (Fin 2 × Fin 2)) : ℚ := (E.card : ℚ) / (S.card : ℚ)

/-- The conditional probability of B given A -/
def conditional_prob : ℚ := P (A ∩ B) / P A

theorem conditional_prob_is_two_thirds : conditional_prob = 2/3 := by
  sorry

end conditional_prob_is_two_thirds_l2130_213025


namespace anniversary_sale_cost_l2130_213029

def original_ice_cream_price : ℚ := 12
def ice_cream_discount : ℚ := 2
def juice_price_per_5_cans : ℚ := 2
def ice_cream_tubs : ℕ := 2
def juice_cans : ℕ := 10

theorem anniversary_sale_cost : 
  (ice_cream_tubs * (original_ice_cream_price - ice_cream_discount)) + 
  (juice_cans / 5 * juice_price_per_5_cans) = 24 := by
  sorry

end anniversary_sale_cost_l2130_213029


namespace chime_1500_date_l2130_213012

/-- Represents a date with year, month, and day. -/
structure Date :=
  (year : Nat) (month : Nat) (day : Nat)

/-- Represents a time with hour and minute. -/
structure Time :=
  (hour : Nat) (minute : Nat)

/-- Represents the chiming pattern of the clock. -/
def chime_pattern (hour : Nat) (minute : Nat) : Nat :=
  if minute == 0 then hour
  else if minute == 15 || minute == 30 then 1
  else 0

/-- Calculates the number of chimes from a given start date and time to an end date and time. -/
def count_chimes (start_date : Date) (start_time : Time) (end_date : Date) (end_time : Time) : Nat :=
  sorry

/-- The theorem to be proved. -/
theorem chime_1500_date :
  let start_date := Date.mk 2003 2 28
  let start_time := Time.mk 18 30
  let end_date := Date.mk 2003 3 13
  count_chimes start_date start_time end_date (Time.mk 23 59) ≥ 1500 ∧
  count_chimes start_date start_time end_date (Time.mk 0 0) < 1500 :=
sorry

end chime_1500_date_l2130_213012


namespace import_tax_threshold_l2130_213094

/-- Proves that the amount in excess of which a 7% import tax was applied is $1,000,
    given that the tax paid was $111.30 on an item with a total value of $2,590. -/
theorem import_tax_threshold (tax_rate : ℝ) (tax_paid : ℝ) (total_value : ℝ) :
  tax_rate = 0.07 →
  tax_paid = 111.30 →
  total_value = 2590 →
  ∃ (threshold : ℝ), threshold = 1000 ∧ tax_rate * (total_value - threshold) = tax_paid :=
by sorry

end import_tax_threshold_l2130_213094


namespace milford_lake_algae_increase_l2130_213063

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original current : ℕ) : ℕ :=
  current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end milford_lake_algae_increase_l2130_213063


namespace christine_needs_32_tablespoons_l2130_213097

/-- Represents the number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- Represents the number of cakes Christine is making -/
def num_cakes : ℕ := 2

/-- Represents the number of egg whites required for each cake -/
def egg_whites_per_cake : ℕ := 8

/-- Calculates the total number of tablespoons of aquafaba needed -/
def aquafaba_needed : ℕ := aquafaba_per_egg * num_cakes * egg_whites_per_cake

/-- Proves that Christine needs 32 tablespoons of aquafaba -/
theorem christine_needs_32_tablespoons : aquafaba_needed = 32 := by
  sorry

end christine_needs_32_tablespoons_l2130_213097


namespace imaginary_part_of_1_minus_2i_l2130_213031

theorem imaginary_part_of_1_minus_2i :
  Complex.im (1 - 2 * Complex.I) = -2 := by sorry

end imaginary_part_of_1_minus_2i_l2130_213031


namespace range_of_a_l2130_213080

def f (a x : ℝ) : ℝ := a^2 * x - 2*a + 1

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f a x ≤ 0) → a ≥ 1/2 := by sorry

end range_of_a_l2130_213080


namespace height_ratio_l2130_213068

def sara_height : ℝ := 120 - 82
def joe_height : ℝ := 82

axiom combined_height : sara_height + joe_height = 120
axiom joe_height_relation : ∃ k : ℝ, joe_height = k * sara_height + 6

theorem height_ratio : (joe_height / sara_height) = 41 / 19 := by
  sorry

end height_ratio_l2130_213068


namespace quadratic_two_members_l2130_213060

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | m * x^2 + 2 * x + 1 = 0}

-- Define the property that A has only two members
def has_two_members (S : Set ℝ) : Prop := ∃ (a b : ℝ), a ≠ b ∧ S = {a, b}

-- Theorem statement
theorem quadratic_two_members :
  ∀ m : ℝ, has_two_members (A m) ↔ (m = 0 ∨ m = 1) :=
by sorry

end quadratic_two_members_l2130_213060


namespace max_and_dog_same_age_in_dog_years_l2130_213005

/-- Conversion rate from human years to dog years -/
def human_to_dog_years : ℕ → ℕ := (· * 7)

/-- Max's age in human years -/
def max_age : ℕ := 3

/-- Max's dog's age in human years -/
def dog_age : ℕ := 3

/-- Theorem: Max and his dog have the same age when expressed in dog years -/
theorem max_and_dog_same_age_in_dog_years :
  human_to_dog_years max_age = human_to_dog_years dog_age :=
by sorry

end max_and_dog_same_age_in_dog_years_l2130_213005


namespace construction_rate_calculation_l2130_213098

/-- Represents the hourly rate for construction work -/
def construction_rate : ℝ := 14.67

/-- Represents the total weekly earnings -/
def total_earnings : ℝ := 300

/-- Represents the hourly rate for library work -/
def library_rate : ℝ := 8

/-- Represents the total weekly work hours -/
def total_hours : ℝ := 25

/-- Represents the weekly hours worked at the library -/
def library_hours : ℝ := 10

theorem construction_rate_calculation :
  construction_rate = (total_earnings - library_rate * library_hours) / (total_hours - library_hours) :=
by sorry

#check construction_rate_calculation

end construction_rate_calculation_l2130_213098


namespace expression_value_l2130_213048

theorem expression_value (a b c d x : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : x^2 = 9)  -- distance from x to origin is 3
  : (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end expression_value_l2130_213048


namespace box_height_proof_l2130_213041

/-- Proves that a box with given dimensions and cube requirements has a specific height -/
theorem box_height_proof (length width : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) (height : ℝ) : 
  length = 10 →
  width = 13 →
  cube_volume = 5 →
  min_cubes = 130 →
  height = (min_cubes : ℝ) * cube_volume / (length * width) →
  height = 5 := by
sorry

end box_height_proof_l2130_213041


namespace geometric_sequence_sum_ratio_l2130_213083

/-- Given a geometric sequence {a_n} with sum S_n of the first n terms,
    if 8a_2 + a_5 = 0, then S_3 / a_3 = 3/4 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) →
  (8 * (a 2) + (a 5) = 0) →
  (S 3) / (a 3) = 3/4 := by
  sorry

end geometric_sequence_sum_ratio_l2130_213083


namespace monic_cubic_polynomial_uniqueness_l2130_213013

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let q := MonicCubicPolynomial a b c
  (q (2 - 3*I) = 0 ∧ q 0 = -30) →
  (a = -82/13 ∧ b = 277/13 ∧ c = -390/13) :=
by sorry

end monic_cubic_polynomial_uniqueness_l2130_213013


namespace backyard_max_area_l2130_213003

theorem backyard_max_area (P : ℝ) (h : P > 0) :
  let A : ℝ → ℝ → ℝ := λ l w => l * w
  let perimeter : ℝ → ℝ → ℝ := λ l w => l + 2 * w
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ perimeter l w = P ∧
    ∀ (l' w' : ℝ), l' > 0 → w' > 0 → perimeter l' w' = P →
      A l w ≥ A l' w' ∧
      A l w = (P / 4) ^ 2 ∧
      w = P / 4 :=
by sorry

end backyard_max_area_l2130_213003


namespace product_equals_square_l2130_213038

theorem product_equals_square : 100 * 19.98 * 1.998 * 1000 = (1998 : ℝ)^2 := by
  sorry

end product_equals_square_l2130_213038


namespace marina_extra_parks_l2130_213092

/-- The number of theme parks in Jamestown -/
def jamestown_parks : ℕ := 20

/-- The number of additional theme parks Venice has compared to Jamestown -/
def venice_extra_parks : ℕ := 25

/-- The total number of theme parks in all three towns -/
def total_parks : ℕ := 135

/-- The number of theme parks in Venice -/
def venice_parks : ℕ := jamestown_parks + venice_extra_parks

/-- The number of theme parks in Marina Del Ray -/
def marina_parks : ℕ := total_parks - (jamestown_parks + venice_parks)

/-- The difference in theme parks between Marina Del Ray and Jamestown -/
def marina_jamestown_difference : ℕ := marina_parks - jamestown_parks

theorem marina_extra_parks :
  marina_jamestown_difference = 50 := by sorry

end marina_extra_parks_l2130_213092


namespace parking_lot_length_l2130_213085

/-- Proves that given the conditions of the parking lot problem, the length is 500 feet -/
theorem parking_lot_length
  (width : ℝ)
  (usable_percentage : ℝ)
  (area_per_car : ℝ)
  (total_cars : ℝ)
  (h1 : width = 400)
  (h2 : usable_percentage = 0.8)
  (h3 : area_per_car = 10)
  (h4 : total_cars = 16000)
  : ∃ (length : ℝ), length = 500 ∧ width * length * usable_percentage = total_cars * area_per_car :=
by
  sorry

end parking_lot_length_l2130_213085


namespace max_sum_of_squares_of_roots_l2130_213011

/-- The quadratic equation in question -/
def quadratic (a x : ℝ) : ℝ := x^2 + 2*a*x + 2*a^2 + 4*a + 3

/-- The sum of squares of roots of the quadratic equation -/
def sumOfSquaresOfRoots (a : ℝ) : ℝ := -8*a - 6

/-- The theorem stating the maximum sum of squares of roots and when it occurs -/
theorem max_sum_of_squares_of_roots :
  (∃ (a : ℝ), ∀ (b : ℝ), sumOfSquaresOfRoots b ≤ sumOfSquaresOfRoots a) ∧
  (sumOfSquaresOfRoots (-3) = 18) := by
  sorry

#check max_sum_of_squares_of_roots

end max_sum_of_squares_of_roots_l2130_213011


namespace square_area_on_parabola_l2130_213055

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end square_area_on_parabola_l2130_213055


namespace at_least_two_babies_speak_l2130_213042

def probability_baby_speaks : ℚ := 1 / 5

def number_of_babies : ℕ := 7

theorem at_least_two_babies_speak :
  let p := probability_baby_speaks
  let n := number_of_babies
  (1 : ℚ) - (1 - p)^n - n * p * (1 - p)^(n-1) = 50477 / 78125 :=
by sorry

end at_least_two_babies_speak_l2130_213042


namespace rhombus_area_l2130_213002

/-- The area of a rhombus with diagonals satisfying a specific equation --/
theorem rhombus_area (a b : ℝ) (h : (a - 1)^2 + Real.sqrt (b - 4) = 0) :
  (1/2 : ℝ) * a * b = 2 := by
  sorry

end rhombus_area_l2130_213002


namespace min_perimeter_triangle_l2130_213022

/-- Given a triangle ABC where a + b = 10 and cos C is a root of 2x^2 - 3x - 2 = 0,
    prove that the minimum perimeter of the triangle is 10 + 5√3 -/
theorem min_perimeter_triangle (a b c : ℝ) (C : ℝ) :
  a + b = 10 →
  2 * (Real.cos C)^2 - 3 * (Real.cos C) - 2 = 0 →
  ∃ (p : ℝ), p = a + b + c ∧ p ≥ 10 + 5 * Real.sqrt 3 ∧
  ∀ (a' b' c' : ℝ), a' + b' = 10 →
    2 * (Real.cos C)^2 - 3 * (Real.cos C) - 2 = 0 →
    a' + b' + c' ≥ p :=
by sorry

end min_perimeter_triangle_l2130_213022


namespace triangle_properties_l2130_213010

/-- Triangle ABC with vertices A(3,0), B(4,6), and C(0,8) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from point B to side AC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  fun p => 2 * p.1 - p.2 - 6 = 0

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := 13

theorem triangle_properties :
  let t : Triangle := { A := (3, 0), B := (4, 6), C := (0, 8) }
  (∀ p, altitude t p ↔ 2 * p.1 - p.2 - 6 = 0) ∧
  area t = 13 := by sorry

end triangle_properties_l2130_213010


namespace child_grandmother_weight_ratio_l2130_213066

/-- Represents the weights of family members and their relationships -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  child : ℝ
  total_weight : grandmother + daughter + child = 130
  daughter_child_weight : daughter + child = 60
  daughter_weight : daughter = 46

/-- The ratio of the child's weight to the grandmother's weight is 1:5 -/
theorem child_grandmother_weight_ratio (fw : FamilyWeights) :
  fw.child / fw.grandmother = 1 / 5 := by
  sorry

end child_grandmother_weight_ratio_l2130_213066


namespace equation_solution_l2130_213082

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^10 = (10 * x)^5 ↔ x = 2/5 := by
  sorry

end equation_solution_l2130_213082


namespace light_bulb_replacement_l2130_213057

def month_number (m : String) : Nat :=
  match m with
  | "January" => 1
  | "February" => 2
  | "March" => 3
  | "April" => 4
  | "May" => 5
  | "June" => 6
  | "July" => 7
  | "August" => 8
  | "September" => 9
  | "October" => 10
  | "November" => 11
  | "December" => 12
  | _ => 0

def cycle_length : Nat := 7
def start_month : String := "January"
def replacement_count : Nat := 12

theorem light_bulb_replacement :
  (cycle_length * (replacement_count - 1)) % 12 + month_number start_month = month_number "June" :=
by sorry

end light_bulb_replacement_l2130_213057


namespace age_difference_is_ten_l2130_213070

/-- The age difference between Declan's elder son and younger son -/
def age_difference : ℕ → ℕ → ℕ
  | elder_age, younger_age => elder_age - younger_age

/-- The current age of Declan's elder son -/
def elder_son_age : ℕ := 40

/-- The age of Declan's younger son 30 years from now -/
def younger_son_future_age : ℕ := 60

/-- The number of years in the future when the younger son's age is known -/
def years_in_future : ℕ := 30

theorem age_difference_is_ten :
  age_difference elder_son_age (younger_son_future_age - years_in_future) = 10 := by
  sorry

end age_difference_is_ten_l2130_213070


namespace equilateral_triangle_perimeter_equilateral_triangle_perimeter_is_60_l2130_213050

/-- The perimeter of an equilateral triangle, given its relationship with an isosceles triangle -/
theorem equilateral_triangle_perimeter : ℝ :=
  let equilateral_side : ℝ := sorry
  let isosceles_base : ℝ := 10
  let isosceles_perimeter : ℝ := 50
  have h1 : isosceles_perimeter = 2 * equilateral_side + isosceles_base := by sorry
  have h2 : equilateral_side = (isosceles_perimeter - isosceles_base) / 2 := by sorry
  3 * equilateral_side

/-- Proof that the perimeter of the equilateral triangle is 60 -/
theorem equilateral_triangle_perimeter_is_60 :
  equilateral_triangle_perimeter = 60 := by sorry

end equilateral_triangle_perimeter_equilateral_triangle_perimeter_is_60_l2130_213050


namespace vector_sum_example_l2130_213081

theorem vector_sum_example :
  let v1 : Fin 3 → ℝ := ![3, -2, 7]
  let v2 : Fin 3 → ℝ := ![-1, 5, -3]
  v1 + v2 = ![2, 3, 4] := by
sorry

end vector_sum_example_l2130_213081


namespace track_circumference_jogging_track_circumference_l2130_213006

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (speed1 speed2 : ℝ) (meeting_time : ℝ) (h1 : speed1 = 4.2)
    (h2 : speed2 = 3.8) (h3 : meeting_time = 4.8 / 60) : ℝ :=
  let distance1 := speed1 * meeting_time
  let distance2 := speed2 * meeting_time
  let total_distance := distance1 + distance2
  total_distance

/-- The circumference of the jogging track is 0.63984 km -/
theorem jogging_track_circumference :
    track_circumference 4.2 3.8 (4.8 / 60) rfl rfl rfl = 0.63984 := by
  sorry

end track_circumference_jogging_track_circumference_l2130_213006


namespace smallest_positive_integer_with_remainders_l2130_213065

theorem smallest_positive_integer_with_remainders : ∃ M : ℕ+,
  (M : ℕ) % 3 = 2 ∧
  (M : ℕ) % 4 = 3 ∧
  (M : ℕ) % 5 = 4 ∧
  (M : ℕ) % 6 = 5 ∧
  (M : ℕ) % 7 = 6 ∧
  (∀ n : ℕ+, n < M →
    (n : ℕ) % 3 ≠ 2 ∨
    (n : ℕ) % 4 ≠ 3 ∨
    (n : ℕ) % 5 ≠ 4 ∨
    (n : ℕ) % 6 ≠ 5 ∨
    (n : ℕ) % 7 ≠ 6) :=
by
  sorry

end smallest_positive_integer_with_remainders_l2130_213065


namespace shortest_distance_is_zero_l2130_213084

/-- Define a 3D vector -/
def Vector3D := Fin 3 → ℝ

/-- Define the first line -/
def line1 (t : ℝ) : Vector3D := fun i => 
  match i with
  | 0 => 4 + 3*t
  | 1 => 1 - t
  | 2 => 3 + 2*t

/-- Define the second line -/
def line2 (s : ℝ) : Vector3D := fun i =>
  match i with
  | 0 => 1 + 2*s
  | 1 => 2 + 3*s
  | 2 => 5 - 2*s

/-- Calculate the square of the distance between two points -/
def distanceSquared (v w : Vector3D) : ℝ :=
  (v 0 - w 0)^2 + (v 1 - w 1)^2 + (v 2 - w 2)^2

/-- Theorem: The shortest distance between the two lines is 0 -/
theorem shortest_distance_is_zero :
  ∃ (t s : ℝ), distanceSquared (line1 t) (line2 s) = 0 := by
  sorry

end shortest_distance_is_zero_l2130_213084


namespace breakfast_time_is_39_minutes_l2130_213046

def sausage_count : ℕ := 3
def egg_count : ℕ := 6
def sausage_time : ℕ := 5
def egg_time : ℕ := 4

def total_breakfast_time : ℕ := sausage_count * sausage_time + egg_count * egg_time

theorem breakfast_time_is_39_minutes : total_breakfast_time = 39 := by
  sorry

end breakfast_time_is_39_minutes_l2130_213046


namespace grandson_height_prediction_l2130_213016

/-- Predicts the height of the next generation using linear regression -/
def predict_next_height (heights : List ℝ) : ℝ :=
  sorry

theorem grandson_height_prediction 
  (heights : List ℝ) 
  (h1 : heights = [173, 170, 176, 182]) : 
  predict_next_height heights = 185 := by
  sorry

end grandson_height_prediction_l2130_213016


namespace triangle_area_with_median_l2130_213076

/-- Given a triangle DEF with side lengths and median, calculate its area using Heron's formula -/
theorem triangle_area_with_median (DE DF DM : ℝ) (h1 : DE = 8) (h2 : DF = 17) (h3 : DM = 11) :
  ∃ (EF : ℝ), let a := DE
               let b := DF
               let c := EF
               let s := (a + b + c) / 2
               (s * (s - a) * (s - b) * (s - c)).sqrt = DM * EF / 2 := by
  sorry

#check triangle_area_with_median

end triangle_area_with_median_l2130_213076


namespace inequality_proof_l2130_213034

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a * (1 - b) * (1 - c)) + Real.sqrt (b * (1 - a) * (1 - c)) + 
  Real.sqrt (c * (1 - a) * (1 - b)) ≤ 1 + Real.sqrt (a * b * c) := by
  sorry

end inequality_proof_l2130_213034


namespace stratified_sampling_junior_count_l2130_213087

theorem stratified_sampling_junior_count 
  (total_employees : ℕ) 
  (junior_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 150) 
  (h2 : junior_employees = 90) 
  (h3 : sample_size = 30) :
  (junior_employees : ℚ) * (sample_size : ℚ) / (total_employees : ℚ) = 18 := by
  sorry

end stratified_sampling_junior_count_l2130_213087


namespace geometric_sequence_fourth_term_l2130_213069

/-- Given a geometric sequence with first term 512 and sixth term 32, 
    the fourth term is 64. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℝ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence property
  a 0 = 512 →                                  -- First term is 512
  a 5 = 32 →                                   -- Sixth term is 32
  a 3 = 64 :=                                  -- Fourth term is 64
by
  sorry

end geometric_sequence_fourth_term_l2130_213069


namespace train_length_l2130_213072

/-- Given a train traveling at 72 km/hr that crosses a pole in 9 seconds, prove that its length is 180 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 72 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = (speed * 1000 / 3600) * time →
  length = 180 := by sorry

end train_length_l2130_213072


namespace drug_use_percentage_is_four_percent_l2130_213078

/-- Warner's Random Response Technique for surveying athletes --/
structure WarnerSurvey where
  total_athletes : ℕ
  yes_answers : ℕ
  prob_odd_roll : ℚ
  prob_even_birthday : ℚ

/-- Calculate the percentage of athletes who have used performance-enhancing drugs --/
def calculate_drug_use_percentage (survey : WarnerSurvey) : ℚ :=
  2 * (survey.yes_answers / survey.total_athletes - 1/4)

/-- Theorem stating that the drug use percentage is 4% for the given survey --/
theorem drug_use_percentage_is_four_percent (survey : WarnerSurvey) 
  (h1 : survey.total_athletes = 200)
  (h2 : survey.yes_answers = 54)
  (h3 : survey.prob_odd_roll = 1/2)
  (h4 : survey.prob_even_birthday = 1/2) :
  calculate_drug_use_percentage survey = 4/100 := by
  sorry

end drug_use_percentage_is_four_percent_l2130_213078


namespace faye_coloring_books_faye_coloring_books_proof_l2130_213000

theorem faye_coloring_books : ℕ :=
  let initial : ℕ := sorry
  let given_away : ℕ := 3
  let bought : ℕ := 48
  let final : ℕ := 79
  have h : initial - given_away + bought = final := by sorry
  initial

-- The proof
theorem faye_coloring_books_proof : faye_coloring_books = 34 := by sorry

end faye_coloring_books_faye_coloring_books_proof_l2130_213000


namespace problem_statement_l2130_213040

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (3 * x + 1)) :
  (x - y + 3 * x * y) / (x * y) = 6 := by
sorry

end problem_statement_l2130_213040


namespace heptagon_exterior_angle_sum_l2130_213061

/-- The exterior angle sum of a heptagon is 360 degrees. -/
theorem heptagon_exterior_angle_sum : ℝ :=
  360

#check heptagon_exterior_angle_sum

end heptagon_exterior_angle_sum_l2130_213061


namespace unique_square_divisible_by_six_l2130_213017

theorem unique_square_divisible_by_six : ∃! x : ℕ,
  (∃ n : ℕ, x = n^2) ∧ 
  (∃ k : ℕ, x = 6 * k) ∧
  50 ≤ x ∧
  x ≤ 150 := by
  sorry

end unique_square_divisible_by_six_l2130_213017


namespace variance_linear_transform_l2130_213024

variable {α : Type*} [LinearOrderedField α]
variable (x : Finset ℕ → α)
variable (n : ℕ)

def variance (x : Finset ℕ → α) (n : ℕ) : α := sorry

theorem variance_linear_transform 
  (h : variance x n = 2) : 
  variance (fun i => 3 * x i + 2) n = 18 := by
  sorry

end variance_linear_transform_l2130_213024


namespace count_eight_to_thousand_l2130_213064

/-- Count of digit 8 in a single integer -/
def count_eight (n : ℕ) : ℕ := sorry

/-- Sum of count_eight for integers from 1 to n -/
def sum_count_eight (n : ℕ) : ℕ := sorry

/-- The count of digit 8 in integers from 1 to 1000 is 300 -/
theorem count_eight_to_thousand : sum_count_eight 1000 = 300 := by sorry

end count_eight_to_thousand_l2130_213064


namespace solution_set_f_geq_3_range_of_a_for_f_geq_a_squared_minus_a_l2130_213086

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for part I
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_geq_a_squared_minus_a :
  {a : ℝ | ∀ x : ℝ, f x ≥ a^2 - a} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} :=
sorry

end solution_set_f_geq_3_range_of_a_for_f_geq_a_squared_minus_a_l2130_213086


namespace morgan_change_is_eleven_l2130_213062

/-- The change Morgan receives after buying lunch -/
def morgan_change (hamburger_cost onion_rings_cost smoothie_cost bill_amount : ℕ) : ℕ :=
  bill_amount - (hamburger_cost + onion_rings_cost + smoothie_cost)

/-- Theorem stating that Morgan receives $11 in change -/
theorem morgan_change_is_eleven :
  morgan_change 4 2 3 20 = 11 := by
  sorry

end morgan_change_is_eleven_l2130_213062


namespace sodium_chloride_percentage_l2130_213089

theorem sodium_chloride_percentage
  (tank_capacity : ℝ)
  (fill_ratio : ℝ)
  (evaporation_rate : ℝ)
  (time : ℝ)
  (final_water_concentration : ℝ)
  (h1 : tank_capacity = 24)
  (h2 : fill_ratio = 1/4)
  (h3 : evaporation_rate = 0.4)
  (h4 : time = 6)
  (h5 : final_water_concentration = 1/2) :
  let initial_volume := tank_capacity * fill_ratio
  let evaporated_water := evaporation_rate * time
  let final_volume := initial_volume - evaporated_water
  let initial_sodium_chloride_percentage := 
    100 * (initial_volume - (final_volume * final_water_concentration)) / initial_volume
  initial_sodium_chloride_percentage = 30 := by
sorry

end sodium_chloride_percentage_l2130_213089


namespace team_reading_balance_l2130_213090

/-- The number of pages in the novel --/
def total_pages : ℕ := 820

/-- Alice's reading speed in seconds per page --/
def alice_speed : ℕ := 25

/-- Bob's reading speed in seconds per page --/
def bob_speed : ℕ := 50

/-- Chandra's reading speed in seconds per page --/
def chandra_speed : ℕ := 35

/-- The number of pages Chandra should read --/
def chandra_pages : ℕ := 482

theorem team_reading_balance :
  bob_speed * (total_pages - chandra_pages) = chandra_speed * chandra_pages := by
  sorry

#check team_reading_balance

end team_reading_balance_l2130_213090


namespace quadratic_root_sum_squares_l2130_213039

theorem quadratic_root_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 + 4 * h * x + 6 = 0 ∧ 
               2 * y^2 + 4 * h * y + 6 = 0 ∧ 
               x^2 + y^2 = 34) → 
  |h| = Real.sqrt 10 := by
sorry

end quadratic_root_sum_squares_l2130_213039


namespace equality_of_cyclic_sum_powers_l2130_213079

theorem equality_of_cyclic_sum_powers (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (h_prime : Nat.Prime p)
  (h_cycle : a^n.val + p * b = b^n.val + p * c ∧ b^n.val + p * c = c^n.val + p * a) :
  a = b ∧ b = c := by sorry

end equality_of_cyclic_sum_powers_l2130_213079


namespace arithmetic_sequence_10th_term_l2130_213045

/-- An arithmetic sequence with given first term and third term -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 8) :
  a 10 = 29 := by
sorry

end arithmetic_sequence_10th_term_l2130_213045


namespace quadratic_solution_for_b_l2130_213033

theorem quadratic_solution_for_b (a b c m : ℝ) (h1 : m = c * a * (b - 1) / (a - b^2)) 
  (h2 : c * a ≠ 0) : m * b^2 + c * a * b - m * a - c * a = 0 := by
  sorry

end quadratic_solution_for_b_l2130_213033


namespace mall_sales_problem_l2130_213019

/-- Represents the cost price of the item in yuan -/
def cost_price : ℝ := 500

/-- Represents the markup percentage in the first month -/
def markup1 : ℝ := 0.2

/-- Represents the markup percentage in the second month -/
def markup2 : ℝ := 0.1

/-- Represents the profit in the first month in yuan -/
def profit1 : ℝ := 6000

/-- Represents the increase in profit in the second month in yuan -/
def profit_increase : ℝ := 2000

/-- Represents the increase in sales volume in the second month -/
def sales_increase : ℕ := 100

/-- Theorem stating the cost price and second month sales volume -/
theorem mall_sales_problem :
  (cost_price * markup1 * (profit1 / (cost_price * markup1)) +
   cost_price * markup2 * ((profit1 + profit_increase) / (cost_price * markup2)) -
   cost_price * markup1 * (profit1 / (cost_price * markup1))) / cost_price = sales_increase ∧
  (profit1 + profit_increase) / (cost_price * markup2) = 160 :=
by sorry

end mall_sales_problem_l2130_213019


namespace smallest_multiples_sum_l2130_213059

theorem smallest_multiples_sum (x y : ℕ) : 
  (x ≥ 10 ∧ x < 100 ∧ x % 2 = 0 ∧ ∀ z : ℕ, (z ≥ 10 ∧ z < 100 ∧ z % 2 = 0) → x ≤ z) ∧
  (y ≥ 100 ∧ y < 1000 ∧ y % 5 = 0 ∧ ∀ w : ℕ, (w ≥ 100 ∧ w < 1000 ∧ w % 5 = 0) → y ≤ w) →
  2 * (x + y) = 220 :=
by sorry

end smallest_multiples_sum_l2130_213059


namespace square_equals_four_digit_l2130_213023

theorem square_equals_four_digit : ∃ (M N : ℕ), 
  10 ≤ M ∧ M < 100 ∧ 
  1000 ≤ N ∧ N < 10000 ∧ 
  M^2 = N :=
sorry

end square_equals_four_digit_l2130_213023


namespace sqrt_fraction_simplification_l2130_213027

theorem sqrt_fraction_simplification :
  Real.sqrt ((25 : ℝ) / 49 + (16 : ℝ) / 81) = (53 : ℝ) / 63 := by
  sorry

end sqrt_fraction_simplification_l2130_213027


namespace workshop_day_probability_l2130_213018

/-- The probability of a student being absent on a normal day -/
def normal_absence_rate : ℚ := 1/20

/-- The probability of a student being absent on the workshop day -/
def workshop_absence_rate : ℚ := min (2 * normal_absence_rate) 1

/-- The probability of a student being present on the workshop day -/
def workshop_presence_rate : ℚ := 1 - workshop_absence_rate

/-- The probability of one student being absent and one being present on the workshop day -/
def one_absent_one_present : ℚ := 
  workshop_absence_rate * workshop_presence_rate * 2

theorem workshop_day_probability : one_absent_one_present = 18/100 := by
  sorry

end workshop_day_probability_l2130_213018


namespace cot_150_degrees_l2130_213088

theorem cot_150_degrees : Real.cos (150 * π / 180) / Real.sin (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end cot_150_degrees_l2130_213088


namespace randy_initial_amount_l2130_213096

def initial_amount (spend_per_visit : ℕ) (visits_per_month : ℕ) (months : ℕ) (remaining : ℕ) : ℕ :=
  spend_per_visit * visits_per_month * months + remaining

theorem randy_initial_amount :
  initial_amount 2 4 12 104 = 200 :=
by sorry

end randy_initial_amount_l2130_213096


namespace geometric_sequence_property_l2130_213053

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h1 : a 5 * a 6 = 3) 
  (h2 : a 9 * a 10 = 9) : 
  a 7 * a 8 = 3 * Real.sqrt 3 := by
  sorry

end geometric_sequence_property_l2130_213053


namespace max_notebooks_is_14_l2130_213044

/-- Represents the pricing options for notebooks -/
structure NotebookPricing where
  single_price : ℕ
  pack3_price : ℕ
  pack7_price : ℕ

/-- Calculates the maximum number of notebooks that can be bought with a given budget and pricing -/
def max_notebooks (budget : ℕ) (pricing : NotebookPricing) : ℕ :=
  sorry

/-- The specific pricing and budget from the problem -/
def problem_pricing : NotebookPricing :=
  { single_price := 2
  , pack3_price := 5
  , pack7_price := 10 }

def problem_budget : ℕ := 20

/-- Theorem stating that the maximum number of notebooks that can be bought is 14 -/
theorem max_notebooks_is_14 : 
  max_notebooks problem_budget problem_pricing = 14 := by sorry

end max_notebooks_is_14_l2130_213044


namespace circumcircle_equation_correct_l2130_213054

/-- The circumcircle of a triangle AOB, where O is the origin (0, 0), A is at (4, 0), and B is at (0, 3) --/
def CircumcircleAOB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 3*p.2 = 0}

/-- Point O is the origin --/
def O : ℝ × ℝ := (0, 0)

/-- Point A has coordinates (4, 0) --/
def A : ℝ × ℝ := (4, 0)

/-- Point B has coordinates (0, 3) --/
def B : ℝ × ℝ := (0, 3)

/-- The circumcircle equation is correct for the given triangle AOB --/
theorem circumcircle_equation_correct :
  O ∈ CircumcircleAOB ∧ A ∈ CircumcircleAOB ∧ B ∈ CircumcircleAOB :=
sorry

end circumcircle_equation_correct_l2130_213054


namespace gcd_lcm_sum_l2130_213009

theorem gcd_lcm_sum : Nat.gcd 42 98 + Nat.lcm 60 15 = 74 := by
  sorry

end gcd_lcm_sum_l2130_213009


namespace big_dig_nickel_output_l2130_213028

/-- Represents the daily mining output of Big Dig Mining Company -/
structure MiningOutput where
  copper : ℝ
  iron : ℝ
  nickel : ℝ

/-- Calculates the total daily output -/
def totalOutput (output : MiningOutput) : ℝ :=
  output.copper + output.iron + output.nickel

theorem big_dig_nickel_output :
  ∀ output : MiningOutput,
  output.copper = 360 ∧
  output.iron = 0.6 * totalOutput output ∧
  output.nickel = 0.1 * totalOutput output →
  output.nickel = 120 := by
sorry


end big_dig_nickel_output_l2130_213028


namespace cody_caramel_boxes_l2130_213077

-- Define the given conditions
def chocolate_boxes : ℕ := 7
def pieces_per_box : ℕ := 8
def total_pieces : ℕ := 80

-- Define the function to calculate the number of caramel boxes
def caramel_boxes : ℕ :=
  (total_pieces - chocolate_boxes * pieces_per_box) / pieces_per_box

-- Theorem statement
theorem cody_caramel_boxes :
  caramel_boxes = 3 := by
  sorry

end cody_caramel_boxes_l2130_213077


namespace marble_difference_l2130_213093

theorem marble_difference (connie_marbles juan_marbles : ℕ) 
  (h1 : connie_marbles = 323)
  (h2 : juan_marbles = 498)
  (h3 : juan_marbles > connie_marbles) : 
  juan_marbles - connie_marbles = 175 := by
  sorry

end marble_difference_l2130_213093


namespace alex_pictures_l2130_213036

/-- The number of pictures Alex has, given processing time per picture and total processing time. -/
def number_of_pictures (minutes_per_picture : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours * 60) / minutes_per_picture

/-- Theorem stating that Alex has 960 pictures. -/
theorem alex_pictures : number_of_pictures 2 32 = 960 := by
  sorry

end alex_pictures_l2130_213036


namespace sqrt_200_simplification_l2130_213091

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end sqrt_200_simplification_l2130_213091


namespace volume_S_form_prism_ratio_l2130_213007

/-- A right rectangular prism with given edge lengths -/
structure RectPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The set of points within distance r of a point in the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

theorem volume_S_form (B : RectPrism) :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  ∀ r : ℝ, r ≥ 0 → volume_S B r = a * r^3 + b * r^2 + c * r + d :=
sorry

theorem prism_ratio (B : RectPrism) (a b c d : ℝ) :
  B.length = 2 ∧ B.width = 3 ∧ B.height = 5 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  (∀ r : ℝ, r ≥ 0 → volume_S B r = a * r^3 + b * r^2 + c * r + d) →
  b * c / (a * d) = 15.5 :=
sorry

end volume_S_form_prism_ratio_l2130_213007


namespace hyperbola_equation_l2130_213047

theorem hyperbola_equation (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  (∃ e : ℝ, e = k * Real.sqrt 5 ∧ 
   (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = k * x)) →
  (∃ x y : ℝ, x^2 / (4 * b^2) - y^2 / b^2 = 1) :=
by sorry

end hyperbola_equation_l2130_213047


namespace teairra_closet_count_l2130_213058

/-- The number of shirts and pants that are neither plaid nor purple -/
def non_plaid_purple_count (total_shirts : ℕ) (total_pants : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) : ℕ :=
  (total_shirts - plaid_shirts) + (total_pants - purple_pants)

theorem teairra_closet_count :
  non_plaid_purple_count 5 24 3 5 = 21 := by
  sorry

end teairra_closet_count_l2130_213058


namespace brand_preference_l2130_213067

theorem brand_preference (total : ℕ) (ratio : ℚ) (brand_x : ℕ) : 
  total = 180 →
  ratio = 5 / 1 →
  brand_x * (1 + 1 / ratio) = total →
  brand_x = 150 :=
by sorry

end brand_preference_l2130_213067


namespace smallest_m_satisfying_conditions_l2130_213074

theorem smallest_m_satisfying_conditions : ∃ m : ℕ+,
  (∀ k : ℕ+, (∃ n : ℕ, 5 * k = n^5) ∧
             (∃ n : ℕ, 6 * k = n^6) ∧
             (∃ n : ℕ, 7 * k = n^7) →
   m ≤ k) ∧
  (∃ n : ℕ, 5 * m = n^5) ∧
  (∃ n : ℕ, 6 * m = n^6) ∧
  (∃ n : ℕ, 7 * m = n^7) ∧
  m = 2^35 * 3^35 * 5^84 * 7^90 :=
by sorry

end smallest_m_satisfying_conditions_l2130_213074


namespace terry_commute_time_l2130_213037

/-- Calculates the total daily driving time for Terry's commute -/
theorem terry_commute_time : 
  let segment1_distance : ℝ := 15
  let segment1_speed : ℝ := 30
  let segment2_distance : ℝ := 35
  let segment2_speed : ℝ := 50
  let segment3_distance : ℝ := 10
  let segment3_speed : ℝ := 40
  let total_time := 
    (segment1_distance / segment1_speed + 
     segment2_distance / segment2_speed + 
     segment3_distance / segment3_speed) * 2
  total_time = 2.9 := by sorry

end terry_commute_time_l2130_213037


namespace roberto_outfits_l2130_213021

/-- Calculates the number of different outfits given the number of options for each clothing item. -/
def calculate_outfits (trousers shirts jackets ties : ℕ) : ℕ :=
  trousers * shirts * jackets * ties

/-- Theorem stating that Roberto can create 240 different outfits. -/
theorem roberto_outfits :
  calculate_outfits 5 6 4 2 = 240 := by
  sorry

#eval calculate_outfits 5 6 4 2

end roberto_outfits_l2130_213021


namespace geometric_sequence_formula_l2130_213051

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (p : ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, a (n + 1) = p * a n + 2^n) →
  geometric_sequence a →
  ∀ n : ℕ, a n = 2^n :=
sorry

end geometric_sequence_formula_l2130_213051


namespace equivalent_discount_l2130_213035

/-- Proves that a single discount of 32.5% before taxes is equivalent to a series of discounts
    (25% followed by 10%) and a 5% sales tax, given an original price of $50. -/
theorem equivalent_discount (original_price : ℝ) (first_discount second_discount tax : ℝ)
  (single_discount : ℝ) :
  original_price = 50 →
  first_discount = 0.25 →
  second_discount = 0.10 →
  tax = 0.05 →
  single_discount = 0.325 →
  original_price * (1 - single_discount) * (1 + tax) =
  original_price * (1 - first_discount) * (1 - second_discount) * (1 + tax) :=
by sorry

end equivalent_discount_l2130_213035


namespace keiko_walking_speed_l2130_213008

/-- Keiko's walking speed around two rectangular tracks with semicircular ends -/
theorem keiko_walking_speed :
  ∀ (speed : ℝ) (width_A width_B time_diff_A time_diff_B : ℝ),
  width_A = 4 →
  width_B = 8 →
  time_diff_A = 48 →
  time_diff_B = 72 →
  (2 * π * width_A) / speed = time_diff_A →
  (2 * π * width_B) / speed = time_diff_B →
  speed = 2 * π / 5 :=
by sorry

end keiko_walking_speed_l2130_213008


namespace meeting_point_theorem_l2130_213014

/-- The distance between two points A and B, where two people walk towards each other
    under specific conditions. -/
def distance_AB : ℝ := 2800

theorem meeting_point_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := distance_AB
  let meeting_point := 1200
  let B_double_speed := 2 * y
  S / 2 / x + (S / 2 - meeting_point) / x = 
    S / 2 / y + (meeting_point - S * y / (2 * x)) / B_double_speed ∧
  S - meeting_point = S / 2 →
  S = 2800 := by sorry

end meeting_point_theorem_l2130_213014


namespace total_ears_is_500_l2130_213075

/-- Calculates the total number of ears for a given number of puppies -/
def total_ears (total_puppies droopy_eared_puppies pointed_eared_puppies : ℕ) : ℕ :=
  2 * total_puppies

/-- Theorem stating that the total number of ears is 500 given the problem conditions -/
theorem total_ears_is_500 :
  let total_puppies : ℕ := 250
  let droopy_eared_puppies : ℕ := 150
  let pointed_eared_puppies : ℕ := 100
  total_ears total_puppies droopy_eared_puppies pointed_eared_puppies = 500 := by
  sorry


end total_ears_is_500_l2130_213075


namespace kramer_packing_theorem_l2130_213030

/-- The number of boxes Kramer can pack per minute -/
def boxes_per_minute : ℕ := 10

/-- The number of cases Kramer can pack in 2 hours -/
def cases_in_two_hours : ℕ := 240

/-- The number of minutes in 2 hours -/
def minutes_in_two_hours : ℕ := 2 * 60

/-- The number of boxes of cigarettes in one case -/
def boxes_per_case : ℕ := (boxes_per_minute * minutes_in_two_hours) / cases_in_two_hours

theorem kramer_packing_theorem : boxes_per_case = 5 := by
  sorry

end kramer_packing_theorem_l2130_213030


namespace smallest_multiple_l2130_213004

theorem smallest_multiple (x : ℕ) : x = 32 ↔ 
  (x > 0 ∧ 900 * x % 1152 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ y < x) → 900 * y % 1152 ≠ 0) :=
sorry

end smallest_multiple_l2130_213004


namespace team_a_games_played_l2130_213095

theorem team_a_games_played (team_a_win_ratio : ℚ) (team_b_win_ratio : ℚ) 
  (team_b_extra_wins : ℕ) (team_b_extra_losses : ℕ) :
  team_a_win_ratio = 3/4 →
  team_b_win_ratio = 2/3 →
  team_b_extra_wins = 5 →
  team_b_extra_losses = 3 →
  ∃ (a : ℕ), 
    a = 4 ∧
    team_b_win_ratio * (a + team_b_extra_wins + team_b_extra_losses) = 
      team_a_win_ratio * a + team_b_extra_wins :=
by sorry

end team_a_games_played_l2130_213095


namespace complex_fraction_equals_neg_i_l2130_213026

theorem complex_fraction_equals_neg_i : (1 + 2*Complex.I) / (Complex.I - 2) = -Complex.I := by
  sorry

end complex_fraction_equals_neg_i_l2130_213026


namespace roper_lawn_cutting_l2130_213099

/-- The number of times Mr. Roper cuts his lawn per month from April to September -/
def summer_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn per month from October to March -/
def winter_cuts : ℕ := 3

/-- The number of months in each season (summer and winter) -/
def months_per_season : ℕ := 6

/-- The total number of months in a year -/
def months_in_year : ℕ := 12

/-- The average number of times Mr. Roper cuts his lawn per month -/
def average_cuts : ℚ := (summer_cuts * months_per_season + winter_cuts * months_per_season) / months_in_year

theorem roper_lawn_cutting :
  average_cuts = 9 := by sorry

end roper_lawn_cutting_l2130_213099


namespace max_shadow_distance_l2130_213015

/-- 
Given a projectile motion with:
- v: initial velocity
- t: time of flight
- y: vertical displacement
- g: gravitational acceleration
- a: constant horizontal acceleration due to air resistance

The maximum horizontal distance L of the projectile's shadow is 0.75 m.
-/
theorem max_shadow_distance 
  (v : ℝ) 
  (t : ℝ) 
  (y : ℝ) 
  (g : ℝ) 
  (a : ℝ) 
  (h1 : v = 5)
  (h2 : t = 1)
  (h3 : y = -1)
  (h4 : g = 10)
  (h5 : y = v * Real.sin α * t - (g * t^2) / 2)
  (h6 : 0 = v * Real.cos α * t - (a * t^2) / 2)
  (h7 : α = Real.arcsin (4/5))
  : ∃ L : ℝ, L = 0.75 ∧ L = (v^2 * (Real.cos α)^2) / (2 * a) := by
  sorry

end max_shadow_distance_l2130_213015


namespace exists_divisible_by_digit_sum_l2130_213073

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 18 consecutive integers ≤ 2016, one is divisible by its digit sum -/
theorem exists_divisible_by_digit_sum :
  ∀ (start : ℕ), start + 17 ≤ 2016 →
  ∃ n ∈ Finset.range 18, (start + n).mod (sum_of_digits (start + n)) = 0 := by
  sorry

end exists_divisible_by_digit_sum_l2130_213073


namespace divisor_of_p_l2130_213020

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 75)
  (h4 : 120 < Nat.gcd s p ∧ Nat.gcd s p < 180) :
  5 ∣ p := by
  sorry

end divisor_of_p_l2130_213020


namespace fraction_proof_l2130_213049

theorem fraction_proof (n : ℝ) (f : ℝ) (h1 : n / 2 = 945.0000000000013) 
  (h2 : (4/15 * 5/7 * n) - (4/9 * f * n) = 24) : f = 0.4 := by
  sorry

end fraction_proof_l2130_213049


namespace ellipse_foci_distance_l2130_213032

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

-- Define the conditions of the problem
def problem_ellipse : Ellipse :=
  { center := (5, 2)
  , a := 5
  , b := 2 }

-- Define the point that the ellipse passes through
def point_on_ellipse : ℝ × ℝ := (3, 1)

-- Theorem statement
theorem ellipse_foci_distance :
  let e := problem_ellipse
  let (x, y) := point_on_ellipse
  let (cx, cy) := e.center
  (((x - cx) / e.a) ^ 2 + ((y - cy) / e.b) ^ 2 ≤ 1) →
  (2 * Real.sqrt (e.a ^ 2 - e.b ^ 2) = 2 * Real.sqrt 21) :=
by sorry

end ellipse_foci_distance_l2130_213032


namespace sum_of_prime_divisors_of_N_l2130_213071

/-- The number of ways to choose a committee from 11 men and 12 women,
    where the number of women is always one more than the number of men. -/
def N : ℕ := (Finset.range 12).sum (λ k => Nat.choose 11 k * Nat.choose 12 (k + 1))

/-- The sum of prime numbers that divide N -/
def sum_of_prime_divisors (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range (n + 1))).sum (λ p => if p ∣ n then p else 0)

theorem sum_of_prime_divisors_of_N : sum_of_prime_divisors N = 79 := by
  sorry

end sum_of_prime_divisors_of_N_l2130_213071
